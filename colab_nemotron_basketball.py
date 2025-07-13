# ============================================================================
# NEMOTRON BASKETBALL REPORT GENERATOR - GOOGLE COLAB VERSION
# ============================================================================

# ------------------------------------------------------------------
# STEP 1: Install Required Packages
# ------------------------------------------------------------------
# !pip -q install --upgrade transformers accelerate bitsandbytes sentencepiece --progress-bar off

# ------------------------------------------------------------------
# STEP 2: Upload Your Files to Colab
# ------------------------------------------------------------------
from google.colab import files
print("Please upload your files:")
print("1. game_1_data.json (or any game data file)")
print("2. prompt_basicreflection.txt")
print("3. prompt_noreflection.txt")
print()

# Uncomment the lines below to upload files
# uploaded = files.upload()

# ------------------------------------------------------------------
# STEP 3: Import Libraries and Setup
# ------------------------------------------------------------------
import os, json, torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# Model configuration
MODEL_ID = "nvidia/Llama-3_1-Nemotron-51B-Instruct"
print(f"Loading model: {MODEL_ID}")

# ------------------------------------------------------------------
# STEP 4: Configure Quantization for Colab
# ------------------------------------------------------------------
bnb_cfg = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
)

# ------------------------------------------------------------------
# STEP 5: Load Tokenizer
# ------------------------------------------------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=True,
)

# Fix tokenizer padding
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded successfully!")

# ------------------------------------------------------------------
# STEP 6: Load Model with Error Handling
# ------------------------------------------------------------------
print("Loading model (this may take a few minutes)...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Try using a smaller model or check your GPU memory")

# ------------------------------------------------------------------
# STEP 7: File Reading Function
# ------------------------------------------------------------------
def read_files():
    """Read the uploaded files"""
    try:
        # Read game data
        with open("game_1_data.json", encoding="utf-8") as f:
            game_json = f.read()
        
        # Read prompts
        with open("prompt_basicreflection.txt", encoding="utf-8") as f:
            tmpl_reflect = f.read()
        
        with open("prompt_noreflection.txt", encoding="utf-8") as f:
            tmpl_noreflect = f.read()
        
        print("All files read successfully!")
        return game_json, tmpl_reflect, tmpl_noreflect
    
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure you've uploaded all required files")
        return None, None, None
    except Exception as e:
        print(f"Error reading files: {e}")
        return None, None, None

# ------------------------------------------------------------------
# STEP 8: Chat Prompt Formatting
# ------------------------------------------------------------------
def create_chat_prompt(template: str, json_data: str) -> str:
    """Create properly formatted chat prompt for Nemotron"""
    
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
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted_prompt
    except Exception as e:
        print(f"Error formatting prompt: {e}")
        return full_prompt  # Fallback to simple format

# ------------------------------------------------------------------
# STEP 9: Generation Function with Error Handling
# ------------------------------------------------------------------
def generate_report(prompt: str, max_new_tokens: int = 600):
    """Generate basketball report with proper error handling"""
    
    try:
        # Tokenize input with length checking
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=True
        ).to(model.device)
        
        print(f"Input tokens: {inputs['input_ids'].shape[1]}")
        
        # Generate with optimized parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return result.strip()
        
    except Exception as e:
        print(f"Generation error: {e}")
        return f"Error generating report: {str(e)}"

# ------------------------------------------------------------------
# STEP 10: Main Execution
# ------------------------------------------------------------------
def main():
    """Main execution function"""
    
    # Read files
    game_json, tmpl_reflect, tmpl_noreflect = read_files()
    
    if not all([game_json, tmpl_reflect, tmpl_noreflect]):
        print("Cannot proceed without all files")
        return
    
    # Create formatted prompts
    print("Creating prompts...")
    prompt_reflect = create_chat_prompt(tmpl_reflect, game_json)
    prompt_noreflect = create_chat_prompt(tmpl_noreflect, game_json)
    
    # Generate reports
    print("\n" + "="*80)
    print("GENERATING REPORT WITH REFLECTION")
    print("="*80)
    
    result_reflect = generate_report(prompt_reflect)
    print(result_reflect)
    
    print("\n" + "="*80)
    print("GENERATING REPORT WITHOUT REFLECTION")
    print("="*80)
    
    result_noreflect = generate_report(prompt_noreflect)
    print(result_noreflect)
    
    # Save results
    print("\nSaving results...")
    with open("report_with_reflection.txt", "w", encoding="utf-8") as f:
        f.write(result_reflect)
    
    with open("report_no_reflection.txt", "w", encoding="utf-8") as f:
        f.write(result_noreflect)
    
    print("Reports saved to files!")
    
    # Download files
    print("\nDownload your reports:")
    files.download("report_with_reflection.txt")
    files.download("report_no_reflection.txt")

# ------------------------------------------------------------------
# STEP 11: Run the Main Function
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
    
# ------------------------------------------------------------------
# ALTERNATIVE: Test with Sample Data (if upload fails)
# ------------------------------------------------------------------
def test_with_sample():
    """Test function with sample data"""
    
    sample_prompt = """
    You are a sports journalist. Write a 350-450 word basketball game report.
    
    Use the following game data:
    - Home team: Hawks (15-35 record)
    - Away team: Opponent
    - Final score: Hawks 105, Opponent 98
    - Date: January 29, 2018
    - Location: Philips Arena, Atlanta
    
    Write a professional sports report about this game.
    """
    
    messages = [{"role": "user", "content": sample_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    result = generate_report(formatted_prompt, max_new_tokens=500)
    print("SAMPLE REPORT:")
    print("="*60)
    print(result)

# Uncomment to test with sample data
# test_with_sample() 