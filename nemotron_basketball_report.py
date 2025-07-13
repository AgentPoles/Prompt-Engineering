# ------------------------------------------------------------------
# 0. One-time installs (quiet output)
# ------------------------------------------------------------------
# !pip -q install --upgrade transformers accelerate bitsandbytes sentencepiece --progress-bar off

# ------------------------------------------------------------------
# 1. Load Nemotron-51B-Instruct with 8-bit quantization
# ------------------------------------------------------------------
import os, json, torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_ID = "nvidia/Llama-3_1-Nemotron-51B-Instruct"

# 8-bit quantization config
bnb_cfg = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
)

# Load tokenizer with proper setup
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=True,
)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# ------------------------------------------------------------------
# 2. Read your local files
# ------------------------------------------------------------------
with open("game_1_data.json", encoding="utf-8") as f:
    game_json = f.read()

with open("prompt_basicreflection.txt", encoding="utf-8") as f:
    tmpl_reflect = f.read()

with open("prompt_noreflection.txt", encoding="utf-8") as f:
    tmpl_noreflect = f.read()

# ------------------------------------------------------------------
# 3. Proper prompt formatting function
# ------------------------------------------------------------------
def create_chat_prompt(template: str, json_data: str) -> str:
    """Create a properly formatted chat prompt for Nemotron"""
    
    # Combine template with JSON data
    full_prompt = f"{template.strip()}\n\n# Game Data\n```json\n{json_data.strip()}\n```"
    
    # Format as chat messages
    messages = [
        {
            "role": "user", 
            "content": full_prompt
        }
    ]
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return formatted_prompt

# ------------------------------------------------------------------
# 4. Improved generation function
# ------------------------------------------------------------------
def generate_report(prompt: str, max_new_tokens: int = 600):
    """Generate basketball report with proper parameters"""
    
    # Tokenize input
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=4096,  # Ensure we don't exceed context length
        padding=True
    ).to(model.device)
    
    # Generate with consistent parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,           # Enable sampling
            temperature=0.7,          # Control randomness
            top_p=0.9,               # Nucleus sampling
            top_k=50,                # Top-k sampling
            repetition_penalty=1.1,   # Prevent repetition
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return result.strip()

# ------------------------------------------------------------------
# 5. Create properly formatted prompts
# ------------------------------------------------------------------
prompt_reflect = create_chat_prompt(tmpl_reflect, game_json)
prompt_noreflect = create_chat_prompt(tmpl_noreflect, game_json)

# ------------------------------------------------------------------
# 6. Generate and display results
# ------------------------------------------------------------------
print("="*100)
print("🟠  WITH REFLECTION")
print("="*100)
try:
    result_reflect = generate_report(prompt_reflect)
    print(result_reflect)
except Exception as e:
    print(f"Error with reflection prompt: {e}")

print("\n" + "="*100)
print("🟢  NO REFLECTION")
print("="*100)
try:
    result_noreflect = generate_report(prompt_noreflect)
    print(result_noreflect)
except Exception as e:
    print(f"Error with no-reflection prompt: {e}")

# ------------------------------------------------------------------
# 7. Optional: Save results to files
# ------------------------------------------------------------------
with open("report_with_reflection.txt", "w", encoding="utf-8") as f:
    f.write(result_reflect)

with open("report_no_reflection.txt", "w", encoding="utf-8") as f:
    f.write(result_noreflect)

print(f"\n✅ Reports saved to files!") 