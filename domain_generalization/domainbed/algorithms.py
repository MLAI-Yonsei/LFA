# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np
import pickle
import os

# import domainbed.captionizer as captionizer
from domainbed import datasets
from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches

import clip
from clip.model import ResidualAttentionBlock

ALGORITHMS = [
    # some algorithms moved to 'other_algorithms.py'
    "ZSCLIP",
    "LPCLIP",
    "CLIP_QLoRA",
    "CLIP_LFA",
    "CLIP_QLoRA_LFA",
    "DPLCLIP",
]

PRTRAINED_QLoRA_PATH = {
    'PACS': [
        '/output_qlora_dir/PACS/test_env_0/model_ckpt.pkl',
        '/output_qlora_dir/PACS/test_env_1/model_ckpt.pkl',
        '/output_qlora_dir/PACS/test_env_2/model_ckpt.pkl',
        '/output_qlora_dir/PACS/test_env_3/model_ckpt.pkl',
    ],
    'VLCS': [
        '/output_qlora_dir/VLCS/test_env_0/model_ckpt.pkl',
        '/output_qlora_dir/VLCS/test_env_1/model_ckpt.pkl',
        '/output_qlora_dir/VLCS/test_env_2/model_ckpt.pkl',
        '/output_qlora_dir/VLCS/test_env_3/model_ckpt.pkl',
    ],
    'OfficeHome': [
        '/output_qlora_dir/OfficeHome/test_env_0/model_ckpt.pkl',
        '/output_qlora_dir/OfficeHome/test_env_1/model_ckpt.pkl',
        '/output_qlora_dir/OfficeHome/test_env_2/model_ckpt.pkl',
        '/output_qlora_dir/OfficeHome/test_env_3/model_ckpt.pkl', 
    ],
    'TerraIncognita': [
        '/output_qlora_dir/TerraIncognita/test_env_0/model_ckpt.pkl',
        '/output_qlora_dir/TerraIncognita/test_env_1/model_ckpt.pkl',
        '/output_qlora_dir/TerraIncognita/test_env_2/model_ckpt.pkl',
        '/output_qlora_dir/TerraIncognita/test_env_3/model_ckpt.pkl',
    ],
    'DomainNet': [
        '/output_qlora_dir/DomainNet/test_env_0/model_ckpt.pkl',
        '/output_qlora_dir/DomainNet/test_env_1/model_ckpt.pkl',
        '/output_qlora_dir/DomainNet/test_env_2/model_ckpt.pkl',
        '/output_qlora_dir/DomainNet/test_env_3/model_ckpt.pkl',
        '/output_qlora_dir/DomainNet/test_env_4/model_ckpt.pkl',
        '/output_qlora_dir/DomainNet/test_env_5/model_ckpt.pkl',
    ],
}

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class LowRankLinear(nn.Module):
    ### it works as a linear layer, if rank is None.
    def __init__(self, rank, input_dim, output_dim, origin: torch.Tensor = None):
        super(LowRankLinear, self).__init__()
        self.rank = rank
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.origin = nn.Parameter(origin.clone().detach()) or nn.Parameter(torch.randn(input_dim, output_dim))
        self.down = None
        self.up = None
        self.scaling = None
        self.dropout = nn.Dropout(p=0.25)
        if self.rank:
            self.down = nn.Linear(input_dim, rank, bias=False)
            self.up = nn.Linear(rank, output_dim, bias=False) 
            self.scaling = 1
            self.origin.requires_grad_(False)
        else:
            self.origin.requires_grad_(True)

    def forward(self, x):
        if self.rank:
            return x @ self.origin + self.scaling * self.dropout(self.up(self.down(x)))
        else:
            return x @ self.origin.type(x.dtype)


class CLIP_LFA(nn.Module):
    ### Use https://github.com/openai/CLIP for CLIP model
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP_LFA, self).__init__()

        # Set Variables
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.beta = hparams["beta"] if "beta" in hparams else 0.9
        text_width = hparams["text_width"] if "text_width" in hparams else 512
        image_width = hparams["image_width"] if "image_width" in hparams else 768
        output_dim = hparams["output_dim"] if "output_dim" in hparams else 512
        rank = hparams["rank"] if "rank" in hparams else 10
        print("\nrank is", rank, "\n")
        self.scaler = np.sqrt(output_dim)

        # Load CLIP model
        self.clip_model = clip.load(self.hparams["clip_backbone"])[0].float()
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
        print("=" * 50)
        print("Set self.self.clip_model.parameters.reguires_grad = False")
        for name, param in self.clip_model.named_parameters():
            param.requires_grad = False
        print("=" * 50)
        
        # Add Feature Extractor Hooks
        self.visual_feature_buffer = []
        def visual_feature_extractor_hook(module, input, output):
            self.visual_feature_buffer.append(output.detach())
        for module in self.clip_model.visual.transformer.resblocks.modules():
            if isinstance(module, ResidualAttentionBlock):
                module.register_forward_hook(visual_feature_extractor_hook)

        self.textual_feature_buffer = []
        def textual_feature_extractor_hook(module, input, output):
            self.textual_feature_buffer.append(output.detach())
        for module in self.clip_model.transformer.resblocks.modules():
            if isinstance(module, ResidualAttentionBlock):
                module.register_forward_hook(textual_feature_extractor_hook)

        # Set Modules
        self.dropout = nn.Dropout(p=0.2)
        self.visual_self_Q = LowRankLinear(rank, image_width, output_dim, origin=self.clip_model.visual.proj)
        self.visual_self_K = LowRankLinear(rank, image_width, output_dim, origin=self.clip_model.visual.proj)
        self.visual_self_V = LowRankLinear(rank, image_width, output_dim, origin=self.clip_model.visual.proj)
        self.textual_self_Q = LowRankLinear(rank, text_width, output_dim, origin=self.clip_model.text_projection)
        self.textual_self_K = LowRankLinear(rank, text_width, output_dim, origin=self.clip_model.text_projection)
        self.textual_self_V = LowRankLinear(rank, text_width, output_dim, origin=self.clip_model.text_projection)
        self.textual_cross_K = LowRankLinear(rank, text_width, output_dim, origin=self.clip_model.text_projection)
        self.textual_cross_V = LowRankLinear(rank, text_width, output_dim, origin=self.clip_model.text_projection)
        self.softmax_cross = nn.Softmax(dim=2)
        self.softmax_self = nn.Softmax(dim=1)

        # Preload Text Features
        with torch.no_grad():
            classnames = [name.replace("_", " ") for name in hparams["class_names"]]
            tokenized_text = clip.tokenize([f"a photo of a {ppt}" for ppt in classnames]).to(self.device)
            self.text_features = self.encode_text(tokenized_text)

        print("=" * 50)
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name} will be updated.")
        print("=" * 50)

        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams["lr"], momentum=self.hparams["momentum"]
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"# of Learnable Parameters: {n_params}")

        self.max_memory_allocated = torch.cuda.max_memory_allocated()
        print(f"Max Memory Allocated: {self.max_memory_allocated}")
        
    def encode_text(self, tokenized_text):
        ### output: [n_layer, n_cls, d_model]
        self.textual_feature_buffer = []
        self.clip_model.encode_text(tokenized_text)
        text_features = torch.stack(self.textual_feature_buffer)    # n_layer, 77, n_cls, 512
        text_features = text_features.permute(0, 2, 1, 3)           # n_layer, n_cls, 77, 512
        text_features = self.clip_model.ln_final(text_features).type(self.dtype)
        text_features = text_features[
            :,  # 12, n_cls, 512
            torch.arange(text_features.shape[1]),
            tokenized_text.argmax(-1),
        ]  # n_layer, n_cls, d_model
        return text_features
    
    def text_cross_attention(self, text_features, image_Q):
        ### output: [batch_size, n_cls, d_model]
        x = text_features.permute(1, 0, 2)  # n_cls, n_layer, width
        Q = image_Q  # batch_size, d_model
        K = self.textual_cross_K(x).permute(0, 2, 1)    # n_cls, d_model, n_layer
        V = self.textual_cross_V(x)                     # n_cls, n_layer, d_model
        attn_score = (
            torch.einsum("ab,cbd->acd", Q, K) / self.scaler
        )  # batch_size, n_cls, n_layer
        attn_prob = self.softmax_cross(attn_score.float())
        attn_value = torch.einsum(
            "abc,bcd->abd", attn_prob, V
        )  # batch_size, n_cls, d_model
        return attn_value

    def text_self_attention(self, text_features):
        ### output: [n_cls, d_model]
        x = text_features.permute(1, 0, 2)  # n_cls, n_layer, width
        Q = self.textual_self_Q(x)[:, -1, :].squeeze(1) # n_cls, 1, d_model -> batch_size, d_model
        K = self.textual_self_K(x).permute(0, 2, 1)     # n_cls, d_model, n_layer
        V = self.textual_self_V(x)                      # n_cls, n_layer, d_model
        attn_score = torch.einsum("ab,abd->ad", Q, K) / self.scaler  # n_cls, n_layer
        attn_prob = self.softmax_self(attn_score.float())
        attn_value = torch.einsum("ab,abc->ac", attn_prob, V)  # n_cls, d_model
        return attn_value

    def encode_image(self, images, feature_paths=None):
        ### output: [n_layer, batch_size, d_model]
        if feature_paths != None:
            list_image_features = []
            for image, feature_path in zip(images, feature_paths):
                try:
                    with open(os.path.join(self.hparams["feature_store_dir"], 
                                        os.path.splitext(feature_path)[0] + ".pkl"), 'rb') as fr:
                        image_features = pickle.load(fr)
                        list_image_features.append(image_features)
                except:
                    self.visual_feature_buffer = []
                    self.clip_model.encode_image(image.unsqueeze(0))
                    image_features = torch.stack(self.visual_feature_buffer)    # n_layer, 197, batch_size, d_model
                    image_features = image_features.permute(0, 2, 1, 3)
                    image_features = self.clip_model.visual.ln_post(image_features[:,:,0,:])
                    list_image_features.append(image_features)
                    os.makedirs(os.path.join(self.hparams["feature_store_dir"], os.path.dirname(feature_path)), exist_ok=True)
                    with open(os.path.join(self.hparams["feature_store_dir"], 
                                        os.path.splitext(feature_path)[0] + ".pkl"), 'wb') as fw:
                        pickle.dump(image_features, fw)
            image_features = torch.cat(list_image_features, dim=1)
        else:
            self.visual_feature_buffer = []
            self.clip_model.encode_image(images)
            image_features = torch.stack(self.visual_feature_buffer)
            image_features = image_features.permute(0, 2, 1, 3)
            image_features = self.clip_model.visual.ln_post(image_features[:,:,0,:])
        return image_features
    
    def image_self_attention(self, image_features):
        ### output: [batch_size, d_model]
        x = image_features.permute(1, 0, 2)  # batch_size, n_layer, width
        Q = self.visual_self_Q(x)[:, -1, :].squeeze(
            1
        )  # batch_size, 1, d_model -> batch_size, d_model
        K = self.visual_self_K(x).permute(0, 2, 1)  # batch_size, d_model, n_layer
        V = self.visual_self_V(x)                   # batch_size, n_layer, d_model
        attn_score = (
            torch.einsum("ab,abd->ad", Q, K) / self.scaler
        )  # batch_size, n_layer
        attn_prob = self.softmax_self(attn_score.float())
        attn_value = torch.einsum("ab,abc->ac", attn_prob, V)  # batch_size, d_model
        return attn_value, Q
        
    def update(self, minibatches, unlabeled=None):
        # minibatches = [[domain_1], [domain_2], [domain_3]], 
        # create minibatches for all domains except the domain of test_envs given by hparams.  
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_path = [['/'.join(path.split('/')[-4:]) for path in data[2]] for data in minibatches]

        xs = [self.encode_image(x, path) for x, path in zip(all_x, all_path)]
        image_bundle = [self.image_self_attention(x) for x in xs]

        # text features
        text_layer_features = self.text_features.type(torch.float32).to(self.device)
        # text features : cross-attention    # 3*[32, 7, 512]
        text_features_cross = [
            self.text_cross_attention(
                text_layer_features, image_Q.type(torch.float32).to(self.device)
            )
            for (_, image_Q) in image_bundle
        ]
        text_features_cross = [
            text_features_cross[i] / text_features_cross[i].norm(dim=-1, keepdim=True)
            for i in range(3)
        ]
        # text : self-attention     # 7, 512
        text_features_self = self.text_self_attention(text_layer_features)
        text_features_self = text_features_self / text_features_self.norm(
            dim=-1, keepdim=True
        )

        # image features : self-attention    # 3*[32, 512]
        image_features = [
            image_features / image_features.norm(dim=-1, keepdim=True)
            for (image_features, _) in image_bundle
        ]

        logit_scale = self.logit_scale.exp()

        # cross-attention
        logits_cross = [
            logit_scale
            * torch.einsum(
                "ab,abc->ac", image_features[i], text_features_cross[i].permute(0, 2, 1)
            )
            for i in range(3)
        ]
        # self-attention
        logits_self = [
            logit_scale * image_features[i] @ text_features_self.t() for i in range(3)
        ]

        logits = [
            self.beta * logits_cross[i] + (1 - self.beta) * logits_self[i]
            for i in range(3)
        ]

        logits = torch.cat(logits)
        loss = F.cross_entropy(logits, all_y)  # [96, 7] and [96]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.max_memory_allocated < torch.cuda.max_memory_allocated():
            self.max_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"Max Memory Allocated: {self.max_memory_allocated}")

        return {"loss": loss.item()}

    def predict(self, x, path):
        if self.max_memory_allocated < torch.cuda.max_memory_allocated():
            self.max_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"Max Memory Allocated: {self.max_memory_allocated}")

        # encoder_image
        path = ['/'.join(p.split('/')[-4:]) for p in path]
        x = self.encode_image(x, path)
        image_features, image_Q = self.image_self_attention(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # encoder_text
        text_layer_features = self.text_features.type(torch.float32).to(self.device)
        # encoder_text : cross-attention
        text_features_cross = self.text_cross_attention(
            text_layer_features, image_Q.type(torch.float32).to(self.device)
        )
        text_features_cross = text_features_cross / text_features_cross.norm(
            dim=-1, keepdim=True
        )
        # encoder_text : self-attention
        text_features_self = self.text_self_attention(text_layer_features)
        text_features_self = text_features_self / text_features_self.norm(
            dim=-1, keepdim=True
        )

        logit_scale = self.logit_scale.exp()

        # cross-attention
        logits_cross = logit_scale * torch.einsum(
            "ab,abc->ac", image_features, text_features_cross.permute(0, 2, 1)
        )

        # self-attention
        logits_self = logit_scale * image_features @ text_features_self.t()

        logits = self.beta * logits_cross + (1 - self.beta) * logits_self

        return logits


class CLIP_QLoRA(Algorithm):
    ### Use https://huggingface.co/docs/transformers/model_doc/clip for CLIP + QLoRA model
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP_QLoRA, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        def print_trainable_parameters(model):
            """
            Prints the number of trainable parameters in the model.
            """
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", quantization_config=bnb_config)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        # The hyperparameters of LoraConfig reference CLIP-LoRA, https://github.com/MaxZanella/CLIP-LoRA
        config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj","k_proj","out_proj"],
            lora_dropout=0.25,
            bias="none",
        )

        self.clip_model = get_peft_model(self.clip_model, config)
        print_trainable_parameters(self.clip_model)

        # embedding dim for image and text encoder.
        self.EMBEDDING_DIM = 512

        classnames = [name.replace("_", " ") for name in hparams["class_names"]]
        self.prompt = torch.cat([clip.tokenize(f"a photo of a {ppt}") for ppt in classnames]).to(self.device)

        print("=" * 50)
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name} will be updated.")
        print("=" * 50)
        
        self.optimizer = torch.optim.SGD(
            self.clip_model.parameters(), lr=self.hparams["lr"], momentum=self.hparams["momentum"]
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"# of Learnable Parameters: {n_params}")

        self.max_memory_allocated = torch.cuda.max_memory_allocated()
        print(f"Max Memory Allocated: {self.max_memory_allocated}")

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        inputs = {
            "input_ids": self.prompt,
            "attention_mask": torch.ones_like(self.prompt),
            "pixel_values": all_x,
        }
        
        outputs = self.clip_model(**inputs)
        logits = outputs.logits_per_image  # this is the image-text similarity score
        
        loss = F.cross_entropy(logits, all_y)  # [96, 7] and [96]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.max_memory_allocated < torch.cuda.max_memory_allocated():
            self.max_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"Max Memory Allocated: {self.max_memory_allocated}")

        return {"loss": loss.item()}

    def predict(self, x, paths):
        if self.max_memory_allocated < torch.cuda.max_memory_allocated():
            self.max_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"Max Memory Allocated: {self.max_memory_allocated}")
        
        inputs = {
            "input_ids": self.prompt,
            "attention_mask": torch.ones_like(self.prompt),
            "pixel_values": x,
        }
        
        outputs = self.clip_model(**inputs)
        logits = outputs.logits_per_image  # this is the image-text similarity score

        return logits.softmax(dim=-1)


class CLIP_QLoRA_LFA(nn.Module):
    ### Use https://huggingface.co/docs/transformers/model_doc/clip for CLIP + QLoRA model
    ### (This will be deprecated) Use https://github.com/openai/CLIP for CLIP's projection weight and layernorm layer.
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP_QLoRA_LFA, self).__init__()
        
        assert (hparams['feature_store_dir'] == None, "feature store is now disabled, it will be enabled in the future.")

        # Set Variables
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.beta = 0.9
        text_width = hparams["text_width"] if "text_width" in hparams else 512
        image_width = hparams["image_width"] if "image_width" in hparams else 768
        output_dim = hparams["output_dim"] if "output_dim" in hparams else 512
        rank = hparams["rank"] if "rank" in hparams else None
        print("\nrank is", rank, "\n")
        self.scaler = np.sqrt(output_dim)

        # Load CLIP model (will be deprecated)
        self._clip_model = clip.load("ViT-B/16")[0].float()
        self.vision_projection = self._clip_model.visual.proj.detach()
        self.text_projection = self._clip_model.text_projection.detach()
        self.logit_scale = self._clip_model.logit_scale.detach()
        self.dtype = self._clip_model.dtype
        self._clip_model.cuda()
        print("=" * 50)
        print("Set self.self._clip_model.parameters.reguires_grad = False")
        for name, param in self._clip_model.named_parameters():
            param.requires_grad = False
        print("=" * 50)
        
        ### NOTE: CLIP-QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", quantization_config=bnb_config)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj","k_proj","out_proj"],
            lora_dropout=0.25,
            bias="none",
        )
        self.clip_model = get_peft_model(self.clip_model, config)
        
        # NOTE: Load Pretrained CLIP-QLoRA Model
        if hparams["load_pretrained"] == 1:
            load_data = torch.load(PRTRAINED_QLoRA_PATH[hparams["dataset"]][hparams["test_envs"][0]])
            missing_keys, unexpected_keys = self.load_state_dict(load_data['model_dict'], strict=False)
            print(f"Pretrained Model Loaded: {hparams['dataset']}, {[hparams['test_envs'][0]]}")
            
        # Freeze CLIP Model
        self.clip_model.cuda()
        print("=" * 50)
        print("Set self.self.clip_model.parameters.reguires_grad = False")
        for name, param in self.clip_model.named_parameters():
            param.requires_grad = False
        print("=" * 50)

        # Add Feature Extractor Hooks
        self.visual_feature_buffer = []
        self.textual_feature_buffer = []

        def visual_feature_extractor_hook(module, input, output):
            self.visual_feature_buffer.append(output[0].detach())

        for name, module in self.clip_model.base_model.model.vision_model.encoder.named_modules():
            if len(name.split('.')) == 2:   #name.endswith("self_attn"):
                module.register_forward_hook(visual_feature_extractor_hook)

        def textual_feature_extractor_hook(module, input, output):
            self.textual_feature_buffer.append(output[0].detach())

        for name, module in self.clip_model.base_model.model.text_model.encoder.named_modules():
            if len(name.split('.')) == 2:   #name.endswith("self_attn"):
                module.register_forward_hook(textual_feature_extractor_hook)
                
        # Set Modules
        self.dropout = nn.Dropout(p=0.2)
        self.visual_self_Q = LowRankLinear(
            rank, image_width, output_dim, origin=self.vision_projection
        )
        self.visual_self_K = LowRankLinear(
            rank, image_width, output_dim, origin=self.vision_projection
        )
        self.visual_self_V = LowRankLinear(
            rank, image_width, output_dim, origin=self.vision_projection
        )
        self.textual_self_Q = LowRankLinear(
            rank, text_width, output_dim, origin=self.text_projection
        )
        self.textual_self_K = LowRankLinear(
            rank, text_width, output_dim, origin=self.text_projection
        )
        self.textual_self_V = LowRankLinear(
            rank, text_width, output_dim, origin=self.text_projection
        )
        self.textual_cross_K = LowRankLinear(
            rank, text_width, output_dim, origin=self.text_projection
        )
        self.textual_cross_V = LowRankLinear(
            rank, text_width, output_dim, origin=self.text_projection
        )
        self.softmax_cross = nn.Softmax(dim=2)
        self.softmax_self = nn.Softmax(dim=1)

        # Preload Text Features
        with torch.no_grad():
            classnames = [name.replace("_", " ") for name in hparams["class_names"]]
            tokenized_text = clip.tokenize(
                [f"a photo of a {ppt}" for ppt in classnames]
            ).to(self.device)
            self.tokenized_text = tokenized_text
            self.text_features = None
            
        print("=" * 50)
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name} will be updated.")
        print("=" * 50)

        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams["lr"], momentum=self.hparams["momentum"]
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"# of Learnable Parameters: {n_params}")

        self.max_memory_allocated = torch.cuda.max_memory_allocated()
        print(f"Max Memory Allocated: {self.max_memory_allocated}")

    # def encode_text(self, tokenized_text):
    def encode_text(self):
        assert self.textual_feature_buffer is None, "textual_feature_buffer is not None"
        text_features = torch.stack(
            self.textual_feature_buffer
        )  # n_layer, 77, n_cls, 512
        text_features = self._clip_model.ln_final(text_features).type(self.dtype)
        text_features = text_features[
            :,  # 12, n_cls, 512
            torch.arange(text_features.shape[1]),
            self.tokenized_text.argmax(-1),
        ]  # n_layer, n_cls, d_model
        return text_features

    def text_cross_attention(self, text_features, image_Q):
        x = text_features.permute(1, 0, 2)  # n_cls, n_layer, width
        Q = image_Q  # batch_size, d_model
        K = self.textual_cross_K(x).permute(0, 2, 1)  # n_cls, d_model, n_layer
        V = self.textual_cross_V(x)  # n_cls, n_layer, d_model
        attn_score = (
            torch.einsum("ab,cbd->acd", Q, K) / self.scaler
        )  # batch_size, n_cls, n_layer
        attn_prob = self.softmax_cross(attn_score.float())
        attn_value = torch.einsum(
            "abc,bcd->abd", attn_prob, V
        )  # batch_size, n_cls, d_model
        return attn_value

    def text_self_attention(self, text_features):
        x = text_features.permute(1, 0, 2)  # n_cls, n_layer, width
        Q = self.textual_self_Q(x)[:, -1, :].squeeze(
            1
        )  # n_cls, 1, d_model -> batch_size, d_model
        K = self.textual_self_K(x).permute(0, 2, 1)  # n_cls, d_model, n_layer
        V = self.textual_self_V(x)  # n_cls, n_layer, d_model
        attn_score = torch.einsum("ab,abd->ad", Q, K) / self.scaler  # n_cls, n_layer
        attn_prob = self.softmax_self(attn_score.float())
        attn_value = torch.einsum("ab,abc->ac", attn_prob, V)  # n_cls, d_model
        return attn_value

    def encode_image(self):
        assert self.visual_feature_buffer is None, "visual_feature_buffer is not None"
        image_features = torch.stack(self.visual_feature_buffer)
        image_features = self._clip_model.visual.ln_post(image_features[:, :, 0, :])
        return image_features

    def image_self_attention(self, image_features):
        x = image_features.permute(1, 0, 2)  # batch_size, n_layer, width
        Q = self.visual_self_Q(x)[:, -1, :].squeeze(
            1
        )  # batch_size, 1, d_model -> batch_size, d_model
        K = self.visual_self_K(x).permute(0, 2, 1)  # batch_size, d_model, n_layer
        V = self.visual_self_V(x)  # batch_size, n_layer, d_model
        attn_score = (
            torch.einsum("ab,abd->ad", Q, K) / self.scaler
        )  # batch_size, n_layer
        attn_prob = self.softmax_self(attn_score)
        # print(attn_prob.dtype, V.dtype)
        attn_value = torch.einsum("ab,abc->ac", attn_prob, V)  # batch_size, d_model
        return attn_value, Q

    def update(self, minibatches, unlabeled=None):
        # minibatches = [[domain_1], [domain_2], [domain_3]],
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_path = [
            ["/".join(path.split("/")[-4:]) for path in data[2]] for data in minibatches
        ]
        n_domains = len(all_x)  # num of domains -m(test_envs)
        
        self.visual_feature_buffer = []
        self.textual_feature_buffer = []
        
        inputs = {
            "input_ids": self.tokenized_text,
            "attention_mask": torch.ones_like(self.tokenized_text),
            "pixel_values": all_x,
        }
        self.clip_model(**inputs)   # forward pass, hook will save features here.
        
        if self.text_features is None:
            self.text_features = text_features = self.encode_text()
        else:
            text_features = self.text_features
        
        xs = self.encode_image()
        image_features, image_Q = self.image_self_attention(xs)

        # text features
        text_layer_features = text_features.type(torch.float32).to(self.device)
        # text features : cross-attention
        text_features_cross = self.text_cross_attention(
            text_layer_features, image_Q.type(torch.float32).to(self.device)
        )
        text_features_cross = text_features_cross / text_features_cross.norm(dim=-1, keepdim=True)
        # text : self-attention
        text_features_self = self.text_self_attention(text_layer_features)
        text_features_self = text_features_self / text_features_self.norm(
            dim=-1, keepdim=True
        )

        # image features : self-attention
        image_features = image_features.type(torch.float32)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        # cross-attention
        logits_cross = logit_scale*torch.einsum(
            "ab,abc->ac", image_features, text_features_cross.permute(0, 2, 1)
        )
        # self-attention
        logits_self = logit_scale*image_features @ text_features_self.t()

        logits = self.beta * logits_cross + (1 - self.beta) * logits_self

        loss = F.cross_entropy(logits, all_y)  # [96, 7] and [96]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.max_memory_allocated < torch.cuda.max_memory_allocated():
            self.max_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"Max Memory Allocated: {self.max_memory_allocated}")

        return {"loss": loss.item()}

    def predict(self, x, path):
        if self.max_memory_allocated < torch.cuda.max_memory_allocated():
            self.max_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"Max Memory Allocated: {self.max_memory_allocated}")

        self.visual_feature_buffer = []
        self.textual_feature_buffer = []
        inputs = {
            "input_ids": self.tokenized_text,
            "attention_mask": torch.ones_like(self.tokenized_text),
            "pixel_values": x,
        }
        self.clip_model(**inputs)
        
        if self.text_features is None:
            self.text_features = text_features = self.encode_text()
        else:
            text_features = self.text_features
        
        x = self.encode_image()
        image_features, image_Q = self.image_self_attention(x)
        image_features = image_features.type(torch.float32)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_layer_features = text_features.type(torch.float32).to(self.device)
        text_features_cross = self.text_cross_attention(
            text_layer_features, image_Q.type(torch.float32).to(self.device)
        )
        text_features_cross = text_features_cross / text_features_cross.norm(dim=-1, keepdim=True)
        text_features_self = self.text_self_attention(text_layer_features)
        text_features_self = text_features_self / text_features_self.norm(
            dim=-1, keepdim=True
        )

        logit_scale = self.logit_scale.exp()
        logits_cross = logit_scale*torch.einsum(
            "ab,abc->ac", image_features, text_features_cross.permute(0, 2, 1)
        )
        logits_self = logit_scale*image_features @ text_features_self.t()
        logits = self.beta * logits_cross + (1 - self.beta) * logits_self

        return logits    


class ZSCLIP(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ZSCLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_model = clip.load(self.hparams["clip_backbone"])[0].float()

        # embedding dim for image and text encoder.
        self.EMBEDDING_DIM = 512  #

        classnames = [name.replace("_", " ") for name in hparams["class_names"]]
        self.prompt = torch.cat(
            [clip.tokenize(f"a photo of a {ppt}") for ppt in classnames]
        ).to(self.device)

        # NOTE: Zero-shot
        print("=" * 50)
        print("Set self.clip_model.parameters.reguires_grad = False!")
        for name, param in self.clip_model.named_parameters():
            param.requires_grad = False
        print("=" * 50)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"# of Learnable Parameters: {n_params}")

        self.max_memory_allocated = torch.cuda.max_memory_allocated()
        print(f"Max Memory Allocated: {self.max_memory_allocated}")

    def update(self, minibatches, unlabeled=None):
        if self.max_memory_allocated < torch.cuda.max_memory_allocated():
            self.max_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"Max Memory Allocated: {self.max_memory_allocated}")

        return {"loss": 0}
        # return {'loss': loss.item()}

    def predict(self, x, paths):
        if self.max_memory_allocated < torch.cuda.max_memory_allocated():
            self.max_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"Max Memory Allocated: {self.max_memory_allocated}")

        logits_per_image, _ = self.clip_model(x, self.prompt)
        return logits_per_image.softmax(dim=-1)
    
    
class LPCLIP(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(LPCLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        
        assert (hparams['feature_store_dir'] is None, "feature store is now disabled, it will be enabled in the future.")
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        text_width = hparams["text_width"] if "text_width" in hparams else 512
        image_width = hparams["image_width"] if "image_width" in hparams else 768
        output_dim = hparams["output_dim"] if "output_dim" in hparams else 512
        rank = hparams["rank"] if "rank" in hparams else None
        print("\nrank is", rank, "\n")

        self.clip_model = clip.load(self.hparams["clip_backbone"])[0].float()
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

        self.image_projection = LowRankLinear(
            rank, image_width, output_dim, origin=self.clip_model.visual.proj
        )
        self.text_projection = LowRankLinear(
            rank, text_width, output_dim, origin=self.clip_model.text_projection
        )
        self.clip_model.visual.proj = (
            None  # clip_model.encode_image() doesn't do projection.
        )
        self.clip_model.text_projection = None

        # Add Feature Extractor Hooks
        self.visual_feature_buffer = []

        def visual_feature_extractor_hook(module, input, output):
            self.visual_feature_buffer.append(output.detach())

        for module in self.clip_model.visual.transformer.resblocks.modules():
            if isinstance(module, ResidualAttentionBlock):
                module.register_forward_hook(visual_feature_extractor_hook)

        with torch.no_grad():
            classnames = [name.replace("_", " ") for name in hparams["class_names"]]
            tokenized_text = clip.tokenize(
                [f"a photo of a {ppt}" for ppt in classnames]
            ).to(self.device)
            self.text_features = self.encode_text(tokenized_text)

        # NOTE: Linear-Probing
        print("=" * 50)
        print("Set self.clip_model.parameters.reguires_grad = False!")
        for name, param in self.clip_model.named_parameters():
            param.requires_grad = False

        print("=" * 50)
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(f"{name} will be updated.")

        print("=" * 50)

        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"],
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"# of Learnable Parameters: {n_params}")

        self.max_memory_allocated = torch.cuda.max_memory_allocated()
        print(f"Max Memory Allocated: {self.max_memory_allocated}")

    def encode_text(self, text):
        x = self.clip_model.token_embedding(text).type(
            self.dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        return x

    def encode_image(self, images, feature_paths=None):
        self.visual_feature_buffer = []
        image_features = self.clip_model.encode_image(images)
        image_features = self.visual_feature_buffer[self.hparams["activate_layer_lpclip"]]
        image_features = image_features.permute(1, 0, 2)  # batch_size, 197, 768
        image_features = self.clip_model.visual.ln_post(image_features[:, 0, :])
        return image_features

    def update(self, minibatches, unlabeled=None):
        # minibatches = [[domain_1], [domain_2], [domain_3]]
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])
        all_path = [
            ["/".join(path.split("/")[-4:]) for path in data[2]] for data in minibatches
        ]

        image_features = torch.cat(
            [self.image_projection(self.encode_image(x)) for x in all_x]
        )
        text_features = self.text_projection(self.text_features)
        # text_features = self.text_features @ self.text_projection

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        loss = F.cross_entropy(logits_per_image, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.max_memory_allocated < torch.cuda.max_memory_allocated():
            self.max_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"Max Memory Allocated: {self.max_memory_allocated}")

        return {"loss": loss.item()}

    def predict(self, x, paths):
        if self.max_memory_allocated < torch.cuda.max_memory_allocated():
            self.max_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"Max Memory Allocated: {self.max_memory_allocated}")

        # image_features = self.encode_image(x, paths) @ self.image_projection
        # text_features = self.text_features @ self.text_projection
        if self.hparams["feature_store_dir"] != None:
            image_features = self.image_projection(self.encode_image(x, paths))
        else:
            image_features = self.image_projection(self.encode_image(x))
        text_features = self.text_projection(self.text_features)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        return logits_per_image


class DPLCLIP(ZSCLIP):
    def __init__(
        self, input_shape, num_classes, num_domains, hparams, sentence_prompt=False
    ):
        super(DPLCLIP, self).__init__(input_shape, num_classes, num_domains, hparams)

        #  initial prompt.
        prompt_prefix = " ".join(["X"] * hparams["num_domain_tokens"])

        classnames = [
            f"a photo of a {name.replace('_', ' ')}" for name in hparams["class_names"]
        ]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        # prompts:  ['X X X X X X X X dog.', 'X X X X X X X X elephant.' ...]

        #  to get default token_prefix and token_suffix.
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(
            self.device
        )
        # tokenized_prompts[0] = tensor([49406,   343,   343,   343,   343,   343,   343,   343,   343,  1929, 269, 49407, 0, 0, ...])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(
                self.clip_model.dtype
            )

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        #  torch.Size([7, 1, 512])
        #  [-0.0001,  0.0002, -0.0046,  ...,  0.0010,  0.0025,  0.0049]

        self.register_buffer(
            "token_suffix", embedding[:, hparams["num_domain_tokens"] + 1 :, :]
        )  # CLS, EOS
        # torch.Size([7, 68, self.EMBEDDING_DIM]), 68 := 77 - num_domain_tokens_tokens - 2.
        # [ 0.0013,  0.0046, -0.0115,  ...,  0.0112,  0.0147,  0.0040],...,.

        self.network = networks.MLP(
            self.EMBEDDING_DIM,
            self.EMBEDDING_DIM * hparams["num_domain_tokens"],
            hparams,
        ).to(device=self.device, dtype=self.clip_model.dtype)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.network.apply(init_weights)

        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"],
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"# of Learnable Parameters: {n_params}")

        self.max_memory_allocated = torch.cuda.max_memory_allocated()
        print(f"Max Memory Allocated: {self.max_memory_allocated}")

    def update(self, minibatches, unlabeled=None):
        # minibatches = [[domain_1], [domain_2], [domain_3]]
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        #  encode image for each domain.
        image_features = [self.clip_model.encode_image(x) for x in all_x]

        #  extract domain_feature for each domain. [32, self.EMBEDDING_DIM] -> [32, self.EMBEDDING_DIM * num_domain_tokens] -> [self.EMBEDDING_DIM * num_domain_tokens].
        domain_features = [self.network(feature) for feature in image_features]
        image_features = torch.cat(image_features)
        #  reshape [self.batch_size, self.EMBEDDING_DIM.]:  -> [1, self.EMBEDDING_DIM.]
        mean_domain_features = [
            feature.mean(dim=0, keepdim=True) for feature in domain_features
        ]

        #  reshape [1, self.EMBEDDING_DIM.]:  -> [7, self.EMBEDDING_DIM.]
        _mean_domain_features = [
            feature.repeat_interleave(len(self.hparams["class_names"]), dim=0)
            for feature in mean_domain_features
        ]

        #  generate text_feature from domain_feature. text_features.size = [3, 7, 512]
        # text_features = [self._get_text_features(feature) for feature in _mean_domain_features]
        text_features = torch.cat(
            [self._get_text_features(feature) for feature in _mean_domain_features]
        )

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = (
            self.clip_model.logit_scale.exp() * image_features @ text_features.t()
        )
        loss = F.cross_entropy(logits_per_image, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.max_memory_allocated < torch.cuda.max_memory_allocated():
            self.max_memory_allocated = torch.cuda.max_memory_allocated()
            print(f"Max Memory Allocated: {self.max_memory_allocated}")

        return {"loss": loss.item()}

    def _get_text_features(self, domain_feature, coop=False):
        #  reshape domain_feature: [7, 16 * self.EMBEDDING_DIM] -> [7, 16, self.EMBEDDING_DIM]
        domain_feature = domain_feature.reshape(
            -1, self.hparams["num_domain_tokens"], self.EMBEDDING_DIM
        )

        #  reshape domain_feature: [7, 16, self.EMBEDDING_DIM] -> [7, 77, self.EMBEDDING_DIM]
        domain_feature = torch.cat(
            [self.token_prefix, domain_feature, self.token_suffix], dim=1
        )

        #  refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        x = domain_feature + self.clip_model.positional_embedding.type(
            self.clip_model.dtype
        )
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        #  mapping domain_features to text_features.
        text_features = (
            x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)]
            @ self.clip_model.text_projection
        )
        return text_features

    def predict(self, x, paths):
        image_feature = self.clip_model.encode_image(x)

        domain_feature = self.network(image_feature)
        mean_domain_feature = torch.mean(
            domain_feature, dim=0, keepdim=True
        ).repeat_interleave(len(self.hparams["class_names"]), dim=0)
        text_feature = self._get_text_features(mean_domain_feature)

        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return self.clip_model.logit_scale.exp() * image_feature @ text_feature.t()

    def encode_text(self):
        pass

    def encode_image(self):
        pass
