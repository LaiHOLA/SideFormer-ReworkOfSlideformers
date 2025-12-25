import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "./qwen"

tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="cuda",
    attn_implementation="sdpa",
)

x = tok("Introduce yourself please", return_tensors="pt").to("cuda")
with torch.no_grad():
    y = model.generate(**x, max_new_tokens=20)
print(tok.decode(y[0], skip_special_tokens=True))
print("bf16_supported:", torch.cuda.is_bf16_supported())
