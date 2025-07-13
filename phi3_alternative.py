import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import gc
import os

# Set environment variables for memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def cleanup_memory():
    """Aggressive memory cleanup"""
    print("🧹 Aggressive memory cleanup...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        print(f"🔍 Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3 - torch.cuda.memory_allocated() / 1024**3:.1f}GB")

def load_optimized_model():
    """Load model with maximum memory optimization"""
    cleanup_memory()
    
    model_name = "microsoft/Phi-3-medium-128k-instruct"
    print(f"🚀 Loading {model_name} with maximum memory optimization...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with maximum optimization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "30GB"}
    )
    
    model.config.use_cache = False
    
    print(f"✅ Model loaded with {torch.cuda.memory_allocated() / 1024**3:.1f}GB GPU memory")
    return model, tokenizer

def create_narrative_prompt(game_data):
    """Create a narrative-style prompt that forces text generation"""
    # Extract key information
    home_team = game_data["home_team"]["name"]
    away_team = game_data["away_team"]["name"]
    home_score = game_data["home_team"]["score"]
    away_score = game_data["away_team"]["score"]
    date = game_data["date"]
    venue = game_data["venue"]
    
    # Get top players
    home_players = sorted(game_data["home_team"]["players"].items(), key=lambda x: x[1]["pts"], reverse=True)[:3]
    away_players = sorted(game_data["away_team"]["players"].items(), key=lambda x: x[1]["pts"], reverse=True)[:3]
    
    prompt = f"""Write a professional basketball game report. Do NOT repeat any JSON data. Write in complete sentences and paragraphs.

GAME: {away_team} at {home_team}
FINAL SCORE: {home_team} {home_score}, {away_team} {away_score}
DATE: {date}
VENUE: {venue}

TOP PERFORMERS:
{home_team}: {home_players[0][0]} ({home_players[0][1]['pts']} pts, {home_players[0][1]['reb']} reb, {home_players[0][1]['ast']} ast)
{away_team}: {away_players[0][0]} ({away_players[0][1]['pts']} pts, {away_players[0][1]['reb']} reb, {away_players[0][1]['ast']} ast)

Write a 350-450 word sports journalism report about this game. Start with: "The {home_team} defeated the {away_team}" or "The {away_team} defeated the {home_team}". Write in complete sentences. Do not include any JSON formatting or data structures."""
    
    return prompt

def generate_with_simple_format(model, tokenizer, prompt, max_new_tokens=500):
    """Generate using simple input format"""
    # Simple tokenization without chat template
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=3500
    ).to(model.device)
    
    print(f"📊 Input tokens: {inputs.input_ids.shape[1]}")
    print(f"📊 Memory before generation: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
    
    # Generate with very restrictive parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # Higher penalty
            no_repeat_ngram_size=4,  # Larger ngram size
            use_cache=False,
            bad_words_ids=[],  # We could add JSON-related tokens here
            early_stopping=True
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    try:
        # Load model
        model, tokenizer = load_optimized_model()
        
        # Load JSON data
        with open('data.json', 'r') as f:
            json_data = json.load(f)
        
        # Create narrative prompt
        prompt = create_narrative_prompt(json_data)
        
        print("="*80)
        print("🏀 PHI-3-MEDIUM - NARRATIVE PROMPT APPROACH")
        print("="*80)
        print("\n📝 PROMPT:")
        print(prompt[:500] + "...")
        print("\n" + "="*80)
        
        # Generate report
        report = generate_with_simple_format(model, tokenizer, prompt, max_new_tokens=600)
        
        print("\n📰 GENERATED REPORT:")
        print("-" * 50)
        print(report)
        print("-" * 50)
        
        # Count words
        word_count = len(report.split())
        print(f"📊 Word count: {word_count}")
        
        # Check if it looks like a proper report
        if any(word in report.lower() for word in ['defeated', 'scored', 'points', 'game', 'quarter']):
            print("✅ Output appears to be a proper sports report!")
        else:
            print("⚠️ Output may not be a proper sports report")
            
        if 350 <= word_count <= 450:
            print("✅ Word count is within target range!")
        else:
            print("⚠️ Word count is outside target range (350-450)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_memory()

if __name__ == "__main__":
    main() 