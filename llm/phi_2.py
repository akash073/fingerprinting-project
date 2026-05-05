from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cpu"
checkpoint = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.float32,  # Use float32 on CPU (float16 is unstable on CPU)
    trust_remote_code=True,
).to(device)

print(model.dtype)
print(model.device)

prompt = "Instruct: What is the capital of France?\nOutput:"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=100, temperature=1, top_p=0.9, do_sample=False)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))