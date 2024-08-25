import torch
from transformers import SiglipProcessor, SiglipModel


def cache_siglip(model_str="google/siglip-so400m-patch14-384"):
    model = SiglipModel.from_pretrained(model_str, torch_dtype=torch.float16)
    processor = SiglipProcessor.from_pretrained(model_str)

if __name__ == "__main__":
    cache_siglip()