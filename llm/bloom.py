from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "bigscience/bloom-560m"
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # float32 for CPU stability
).to(device)

print(model.dtype)
print(model.device)

# BLOOM is a completion model — no chat template, use plain prompts
prompt = "The capital of France is"

inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
outputs = model.generate(
    inputs,
    max_new_tokens=50,
    temperature=1,
    top_p=0.9,
    do_sample=False,
    repetition_penalty=1.3,  # Helps reduce repetition common in BLOOM
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))