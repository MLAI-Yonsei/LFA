import os
import re

dataset_path = "/datasets/PACS"

if __name__ == '__main__':
    os.makedirs(dataset_path, exist_ok=True)
    domain_list = os.listdir(dataset_path)
    for domain_ in domain_list:
        classes_path = os.path.join(dataset_path, domain_)
        os.makedirs(classes_path, exist_ok=True)
        classes_list = os.listdir(classes_path)
        for class_ in classes_list:
            images_path = os.path.join(classes_path, class_)
            os.makedirs(images_path, exist_ok=True)
            images_list = os.listdir(images_path)
            for file in images_list:
                full_path = os.path.join(images_path, file)
                name, ext = os.path.splitext(file)
                if ext == '.txt' or ext == '.pickle': 
                    os.remove(full_path)
                