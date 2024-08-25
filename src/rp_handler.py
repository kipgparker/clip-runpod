import torch
import runpod
from transformers import SiglipProcessor, SiglipModel
from rp_schemas import INPUT_SCHEMA
from runpod.serverless.utils.rp_validator import validate
import time

device = torch.device("cuda:0")
model_str="google/siglip-so400m-patch14-384"
model = SiglipModel.from_pretrained(model_str, torch_dtype=torch.float16, device_map=device)
processor = SiglipProcessor.from_pretrained(model_str)

def process_request(job):
    start_time = time.time()

    validated_input = validate(job["input"], INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    tokens = processor(text=job_input["text"])["input_ids"]

    with torch.no_grad():
        input_ids = tokens.to(device)
        embedding = model.get_text_features(input_ids.unsqueeze(0))[0].tolist()
            
    torch.cuda.synchronize()

    total_time = time.time() - start_time
    return {"total_time": total_time, "embedding": embedding}

def main():
    runpod.serverless.start({
        "handler": process_request,
    })

if __name__ == "__main__":
    main()