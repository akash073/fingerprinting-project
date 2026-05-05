from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "distilgpt2"
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

print(model.dtype)
print(model.device)

prompt = "The capital of France is"

inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    inputs,
    max_new_tokens=50,
    temperature=1,
    top_p=0.9,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,  # Avoids padding warning
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))