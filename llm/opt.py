from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "facebook/opt-125m"
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

prompt = "What is the capital of France?"
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))