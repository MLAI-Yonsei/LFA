import os.path as osp
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, Registry, check_availability
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import LayerNorm
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

CUSTOM_DATASETS = {
    "OxfordPets": "oxford_pets", 
    "OxfordFlowers": "oxford_flowers",
    "FGVCAircraft": "fgvc_aircraft",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat", 
    "StanfordCars": "stanford_cars",
    "Food101": "food101",
    "SUN397": "sun397",
    "Caltech101": "caltech101",
    "UCF101": "ucf101",
    "ImageNet": "imagenet",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
}

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model

def load_pretrained_model(cfg, model_name):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(model_name, avai_trainers)
    model = TRAINER_REGISTRY.get(model_name)(cfg)
    _dataset = CUSTOM_DATASETS[cfg.DATASET.NAME]
    if 'imagenet' in _dataset:
        model.load_model(
            f"/path/to/output_dir/imagenet/CoOp/vit_b16_16shots/nctx16_cscFalse_ctpend/seed1",
            epoch=200) # NOTE!!!
    else:
        model.load_model(
            f"/path/to/output_dir/{_dataset}/CoOp/vit_b16_16shots/nctx16_cscFalse_ctpend/seed1",
            epoch=200) # NOTE!!!
    # print(model, model.model)
    # model.load_model(cfg.MODEL_DIR, epoch=cfg.LOAD_EPOCH)
    print('='*20+"Loaded Model!")
    return model.model
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model, width, output_dim):
        super().__init__()    
        scale = width ** -0.5
        self.transformer = clip_model.transformer.float()   
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype
        # self.textual_projection = clip_model.text_projection

        self.Q_self = nn.Linear(width, output_dim, bias=False)
        self.K_self = nn.Linear(width, output_dim, bias=False)
        self.V_self = nn.Linear(width, output_dim, bias=False)

        self.K_cross = nn.Linear(width, output_dim, bias=False)
        self.V_cross = nn.Linear(width, output_dim, bias=False)

        self.V_self.weight = nn.Parameter(clip_model.text_projection.t())
        self.V_cross.weight = nn.Parameter(clip_model.text_projection.t())

        self.softmax_cross = nn.Softmax(dim=2)
        self.softmax_self = nn.Softmax(dim=1)
        self.scaler = np.sqrt(output_dim)
        
    def forward(self, prompt):
        out_list = []
        # x = self.token_embedding(tokenized_text).type(self.dtype)
        x = prompt + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.transformer.layers):
            x = self.transformer.resblocks[i](x.float())    
            # print(f"LOG::TYPEs {x.dtype}")
            tmp = x.permute(1, 0, 2)            # LND -> NLD
            out_list.append(tmp)
        text_features = torch.stack(out_list)
        text_features = self.ln_final(text_features).type(self.dtype)
        return text_features
    
    def cross_attention(self, text_features, image_Q):
        # n_layer, n_prompt, width
        x = text_features.permute(1, 0, 2)  # n_prompt, n_layer, width
        Q = image_Q                  # batch_size, d_model
        # print(f"LOG::text_Q.shape: {Q.shape}")
        K = self.K_cross(x).permute(0, 2, 1)      # n_prompt, d_model, n_layer 
        # print(f"LOG::text_K.shape: {K.shape}")
        V = self.V_cross(x)                       # n_prompt, n_layer, d_model
        # print(f"LOG::text_V.shape: {V.shape}")
        attn_score = torch.einsum('ab,cbd->acd',Q,K) / self.scaler    # batch_size, n_prompt, n_layer
        attn_prob = self.softmax_cross(attn_score.float())
        attn_value = torch.einsum('abc,bcd->abd',attn_prob,V)         # batch_size, n_prompt, d_model
        return attn_value
    
    def self_attention(self, text_features):
        x = text_features.permute(1, 0, 2)      # n_prompt, n_layer, width
        Q = self.Q_self(x)[:,-1,:].squeeze(1)        # n_prompt, 1, d_model -> batch_size, d_model
        # print(f"LOG::image_Q.shape: {Q.shape}")
        K = self.K_self(x).permute(0, 2, 1)          # n_prompt, d_model, n_layer
        # print(f"LOG::image_K.shape: {K.shape}")
        V = self.V_self(x)                           # n_prompt, n_layer, d_model
        # print(f"LOG::image_V.shape: {V.shape}")
        attn_score = torch.einsum('ab,abd->ad',Q,K) / self.scaler   # n_prompt, n_layer
        attn_prob = self.softmax_self(attn_score.float())
        attn_value = torch.einsum('ab,abc->ac',attn_prob,V)         # n_prompt, d_model
        return attn_value

class ImageEncoder(nn.Module):
    def __init__(self, clip_model, width, output_dim):
        super().__init__()
        scale = width ** -0.5
        self.conv1 = clip_model.visual.conv1
        self.transformer = clip_model.visual.transformer
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.ln_post = clip_model.visual.ln_post
        self.dtype = clip_model.dtype
        # self.visual_projection = clip_model.visual.proj
        self.Q = nn.Linear(width, output_dim, bias=False)
        self.K = nn.Linear(width, output_dim, bias=False)
        self.V = nn.Linear(width, output_dim, bias=False)

        self.V.weight = nn.Parameter(clip_model.visual.proj.t())

        self.softmax = nn.Softmax(dim=1)
        self.scaler = np.sqrt(output_dim)

    def forward(self, x: torch.Tensor):
        out_list = []
        
        x = self.conv1(x.float())
        x = x.reshape(x.shape[0], x.shape[1], -1)   # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)                      # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + 
                    torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)          # NLD -> LND

        for i in range(self.transformer.layers):
            x = self.transformer.resblocks[i](x.float())
            tmp = x.permute(1, 0, 2)    # LND -> NLD
            tmp = tmp[:, 0, :]
            out_list.append(self.ln_post(tmp))
        image_features = torch.stack(out_list)  
        # image_features = torch.einsum('abc,cd->abd', image_features, self.visual_projection)     
        return image_features
    
    def self_attention(self, image_features):
        x = image_features.permute(1, 0, 2)     # batch_size, n_layer, width
        Q = self.Q(x)[:,-1,:].squeeze(1)        # batch_size, 1, d_model -> batch_size, d_model
        # print(f"LOG::image_Q.shape: {Q.shape}")
        K = self.K(x).permute(0, 2, 1)          # batch_size, d_model, n_layer
        # print(f"LOG::image_K.shape: {K.shape}")
        V = self.V(x)                           # batch_size, n_layer, d_model
        # print(f"LOG::image_V.shape: {V.shape}")
        attn_score = torch.einsum('ab,abd->ad',Q,K) / self.scaler   # batch_size, n_layer
        attn_prob = self.softmax(attn_score.float())
        attn_value = torch.einsum('ab,abc->ac',attn_prob,V)         # batch_size, d_model
        return attn_value, Q


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, pretrained_model):
        super().__init__()
        # NOTE: CoOp
        self.prompt_learner = pretrained_model.prompt_learner
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # NOTE: LFA
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype   # float16
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dataset = cfg.DATASET.NAME

        # if dataset == "ImageNet":
        #     TEMPLATES = IMAGENET_TEMPLATES_SELECT
        # else:
        #     TEMPLATES = []
        # TEMPLATES += [CUSTOM_TEMPLATES[dataset]]
        
        TEXT_WIDTH = 512
        IMAGE_WIDTH = 768
        OUTPUT_DIM = 512
        
        text_encoder = TextEncoder(clip_model, TEXT_WIDTH, OUTPUT_DIM)
        image_encoder = ImageEncoder(clip_model, IMAGE_WIDTH, OUTPUT_DIM)

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.dropout = nn.Dropout(p=0.2)
                
        # self.beta = nn.Parameter(torch.rand(1))
        self.beta = 0.9
        
        print("="*50)
        print('Set self.clip_model.parameters.reguires_grad = False!')
        for name, param in clip_model.named_parameters():
            param.requires_grad = False
        print('Set pretrained_model.parameters.reguires_grad = False')
        for name, param in pretrained_model.named_parameters():
            if param.requires_grad == True:
                param.requires_grad_(False)
                print(f"{name}, {True} => {param.requires_grad}")
        print("="*50)
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name} will be updated.")
        print("="*50)
        
        
    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts

        # encoder_image
        x = self.image_encoder(image)                                   # 12, batch_size, 768
        x = self.dropout(x)
        image_features, image_Q = self.image_encoder.self_attention(x)  # batch_size, 512
        
        prompts = self.prompt_learner()
        x = self.text_encoder(prompts)
        x = x[:, torch.arange(x.shape[1]), tokenized_prompts.argmax(dim=-1)]
        x = x.type(torch.float32).to(self.device)
        x = self.dropout(x)

        text_features_cross = self.text_encoder.cross_attention(
            x,
            image_Q.type(torch.float32).to(self.device)
        )
        text_features_cross = text_features_cross / text_features_cross.norm(dim=-1, keepdim=True)
        text_features_cross = text_features_cross.permute(0, 2, 1)          # batch_size, 512, n_class

        text_features_self = self.text_encoder.self_attention(x)
        text_features_self = text_features_self / text_features_self.norm(dim=-1, keepdim=True)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        # cross-attention
        logits_cross = logit_scale * torch.einsum('ab,abc->ac', image_features, text_features_cross)
        
        # self-attention
        logits_self = logit_scale * image_features @ text_features_self.t()
        
        logits = self.beta*logits_cross + (1-self.beta)*logits_self
        
        return logits
    

@TRAINER_REGISTRY.register()
class LFAonCoOp(TrainerX):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.LFA.PREC == "fp32" or cfg.TRAINER.LFA.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            
        print(f"Loading Pretrained-model")
        pretrained_model = load_pretrained_model(cfg, 'CoOp')
        
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, pretrained_model)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        self.model = self.model.float()
        
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("layer_aggregation", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.LFA.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.LFA.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            
            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)