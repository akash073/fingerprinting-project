from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,  # float32 for CPU stability
).to(device)

messages = [
    {"role": "system", "content": "You are a friendly and helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        # temperature=1,
        # top_p=0.9,
        # do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(response)


################### Old working code for TinyLlama 1.1B Chat v1.0 with 4-bit quantization ###################
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
# import torch

# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# # 4-bit quantization config
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Load model with 4-bit quantization
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     device_map="auto",
# )

# # TinyLlama uses the ChatML format
# messages = [
#     {"role": "system", "content": "You are a friendly and helpful assistant."},
#     {"role": "user", "content": "What is the capital of France?"}
# ]

# # Apply chat template
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )

# inputs = tokenizer([text], return_tensors="pt").to(model.device)

# # Generate response
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=512,
#         temperature=0.7,
#         top_p=0.9,
#         do_sample=True,
#         repetition_penalty=1.1,   # Helps reduce repetition common in small models
#     )

# response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
# print(response)