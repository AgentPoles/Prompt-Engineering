import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import gc
import os
from datetime import datetime

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

def load_qwen_72b():
    """Load Qwen 72B with 4-bit quantization for A100 80GB"""
    print("🚀 Loading Qwen 72B with 4-bit quantization...")
    
    # 4-bit quantization config optimized for A100
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model_name = "Qwen/Qwen2.5-72B-Instruct"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print(f"✅ Qwen 72B loaded successfully")
        if torch.cuda.is_available():
            print(f"📊 GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
        
        return model, tokenizer
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

def load_game_data():
    """Load and optimize game data"""
    with open('data.json', 'r') as f:
        data = json.load(f)
    return data

def create_sports_prompt(game_data):
    """Create optimized prompt for sports journalism"""
    # Load the reflection prompt
    with open('prompt_basicreflection.txt', 'r') as f:
        base_prompt = f.read()
    
    # Convert game data to condensed format
    game_info = game_data.get('game_info', {})
    teams = game_data.get('teams', {})
    
    # Extract key information
    condensed_data = {
        'game_info': {
            'date': game_info.get('date'),
            'venue': game_info.get('venue'),
            'final_score': game_info.get('final_score')
        },
        'teams': {}
    }
    
    # Process team data
    for team_name, team_data in teams.items():
        condensed_data['teams'][team_name] = {
            'record': team_data.get('record'),
            'stats': team_data.get('team_stats', {}),
            'quarters': team_data.get('quarters', []),
            'top_players': {}
        }
        
        # Get top 3 players by points
        players = team_data.get('players', {})
        sorted_players = sorted(players.items(), key=lambda x: x[1].get('pts', 0), reverse=True)[:3]
        
        for player_name, player_stats in sorted_players:
            condensed_data['teams'][team_name]['top_players'][player_name] = {
                'pts': player_stats.get('pts', 0),
                'reb': player_stats.get('reb', 0),
                'ast': player_stats.get('ast', 0),
                'stl': player_stats.get('stl', 0),
                'blk': player_stats.get('blk', 0),
                'starter': player_stats.get('starter', False)
            }
    
    # Create the final prompt
    prompt = f"""{base_prompt}

# Game Data (JSON):
{json.dumps(condensed_data, indent=2)}

# Task:
Write a professional NBA game report following ALL instructions above. The report must be between 350-450 words, written as a single paragraph, and based strictly on the provided JSON data."""

    return prompt

def generate_report(model, tokenizer, prompt):
    """Generate sports report with Qwen 72B"""
    print("🏀 Generating sports report with Qwen 72B...")
    
    # Create chat format for Qwen
    messages = [
        {"role": "system", "content": "You are a professional NBA sports journalist. Write accurate, engaging game reports based on provided data."},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4000)
    input_ids = inputs["input_ids"].to(model.device)
    
    print(f"📊 Input tokens: {len(input_ids[0])}")
    print(f"📊 Memory before generation: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
    
    # Generation parameters optimized for Qwen 72B
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False  # Save memory
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return response.strip()

def main():
    """Main execution function"""
    print("=" * 80)
    print("🏀 QWEN 72B SPORTS JOURNALISM GENERATOR - A100 80GB OPTIMIZED")
    print("=" * 80)
    
    # Memory cleanup
    cleanup_memory()
    
    # Load model
    model, tokenizer = load_qwen_72b()
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Load game data
    print("📊 Loading game data...")
    game_data = load_game_data()
    
    # Create prompt
    print("📝 Creating optimized prompt...")
    prompt = create_sports_prompt(game_data)
    
    # Generate report
    report = generate_report(model, tokenizer, prompt)
    
    # Display results
    print("\n" + "=" * 80)
    print("🏀 GENERATED SPORTS REPORT")
    print("=" * 80)
    print(report)
    print("=" * 80)
    
    # Word count
    word_count = len(report.split())
    print(f"📊 Word count: {word_count}")
    print(f"📊 Target range: 350-450 words")
    
    if 350 <= word_count <= 450:
        print("✅ Word count within target range!")
    else:
        print("⚠️ Word count outside target range")
    
    # Final memory cleanup
    cleanup_memory()

if __name__ == "__main__":
    main() 