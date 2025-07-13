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

def load_qwen_32b():
    """Load Qwen 32B with optimized settings for A100 80GB"""
    print("🚀 Loading Qwen 32B with 4-bit quantization...")
    
    # 4-bit quantization config - very conservative for reliability
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model_name = "Qwen/Qwen2.5-32B-Instruct"
    
    try:
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print("🔧 Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2"  # More memory efficient
        )
        
        print(f"✅ Qwen 32B loaded successfully!")
        if torch.cuda.is_available():
            print(f"📊 GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
        
        return model, tokenizer
    except Exception as e:
        print(f"❌ Failed to load model with flash attention, trying without...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print(f"✅ Qwen 32B loaded successfully (without flash attention)!")
            if torch.cuda.is_available():
                print(f"📊 GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
            return model, tokenizer
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            return None, None

def load_game_data():
    """Load and optimize game data"""
    with open('data.json', 'r') as f:
        data = json.load(f)
    return data

def create_sports_prompt(game_data, prompt_type="reflection"):
    """Create optimized prompt for sports journalism with better data processing"""
    
    # Debug: Show what data we're working with
    print(f"🔍 DEBUG: Game data keys: {list(game_data.keys())}")
    
    # Load prompt with fallback
    base_prompt = ""
    if prompt_type == "reflection":
        prompt_files = ["prompt_with_reflection.txt", "prompt_basicreflection.txt"]
    else:
        prompt_files = ["prompt_with_no_reflection.txt", "prompt_noreflection.txt"]
    
    for filename in prompt_files:
        try:
            with open(filename, 'r') as f:
                base_prompt = f.read()
                print(f"📄 Using prompt file: {filename}")
                break
        except FileNotFoundError:
            continue
    
    # Fallback prompt if no file found
    if not base_prompt:
        base_prompt = """You are a professional NBA sports journalist. Write an engaging 350-450 word game report based strictly on the provided game data. Include the final score, key player performances, team statistics, and game flow. Write as a single paragraph."""
    
    # Extract and process game data more comprehensively
    game_info = game_data.get('game_info', {})
    teams = game_data.get('teams', {})
    
    print(f"🔍 DEBUG: Found {len(teams)} teams in data")
    
    # Create comprehensive game summary
    game_summary = {
        'game_info': {
            'date': game_info.get('date', 'Recent game'),
            'venue': game_info.get('venue', 'NBA Arena'),
            'final_score': game_info.get('final_score', 'Score not available')
        },
        'teams': {}
    }
    
    # Process each team thoroughly
    for team_name, team_data in teams.items():
        print(f"🔍 DEBUG: Processing team: {team_name}")
        
        # Get team stats
        team_stats = team_data.get('team_stats', {})
        quarters = team_data.get('quarters', [])
        players = team_data.get('players', {})
        
        print(f"🔍 DEBUG: Team {team_name} has {len(players)} players")
        
        # Get top performers
        top_players = {}
        if players:
            sorted_players = sorted(players.items(), key=lambda x: x[1].get('pts', 0), reverse=True)[:4]
            for player_name, player_stats in sorted_players:
                top_players[player_name] = {
                    'pts': player_stats.get('pts', 0),
                    'reb': player_stats.get('reb', 0),
                    'ast': player_stats.get('ast', 0),
                    'stl': player_stats.get('stl', 0),
                    'blk': player_stats.get('blk', 0),
                    'fgm': player_stats.get('fgm', 0),
                    'fga': player_stats.get('fga', 0),
                    'fg3m': player_stats.get('fg3m', 0),
                    'fg3a': player_stats.get('fg3a', 0),
                    'starter': player_stats.get('starter', False)
                }
        
        game_summary['teams'][team_name] = {
            'record': team_data.get('record', 'Record not available'),
            'team_stats': team_stats,
            'quarters': quarters,
            'top_players': top_players
        }
    
    # Debug: Show processed data
    print(f"🔍 DEBUG: Processed data structure: {json.dumps(game_summary, indent=2)[:500]}...")
    
    # Create clear, direct prompt
    prompt = f"""You are a professional NBA sports journalist. Your task is to write a comprehensive game report based on the provided game data.

REQUIREMENTS:
- Write exactly 350-450 words
- Write as a single continuous paragraph
- Include final score, venue, and date
- Mention key player performances with specific statistics
- Include team records and important team statistics
- Base everything strictly on the provided data

GAME DATA:
{json.dumps(game_summary, indent=2)}

TASK: Write a professional NBA game report following the requirements above. Focus on the actual statistics and game details provided. Generate ONLY the final game report."""

    return prompt

def generate_report(model, tokenizer, prompt, scenario_name):
    """Generate sports report with Qwen 32B"""
    print(f"🏀 Generating sports report with Qwen 32B ({scenario_name})...")
    
    # Create chat format for Qwen
    messages = [
        {"role": "system", "content": "You are a professional NBA sports journalist. Write accurate, detailed game reports based on provided data. Use the actual statistics and information provided in the data."},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize with more generous limits
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=12000)
    input_ids = inputs["input_ids"].to(model.device)
    
    print(f"📊 Input tokens: {len(input_ids[0])}")
    print(f"📊 Memory before generation: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
    
    # Generation parameters optimized for detailed reports
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=800,  # Increased for longer reports
            do_sample=True,
            temperature=0.8,  # Slightly higher for more creativity
            top_p=0.9,
            repetition_penalty=1.15,  # Higher to avoid repetition
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
    print("🏀 QWEN 32B SPORTS JOURNALISM GENERATOR - DEBUG VERSION")
    print("=" * 80)
    
    # Memory cleanup
    cleanup_memory()
    
    # Load model
    model, tokenizer = load_qwen_32b()
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    # Load game data
    print("📊 Loading game data...")
    game_data = load_game_data()
    
    scenarios = [
        ("WITH REFLECTION", "reflection"),
        ("WITHOUT REFLECTION", "no_reflection")
    ]
    
    for scenario_name, scenario_type in scenarios:
        print(f"\n{'='*80}")
        print(f"🏀 SCENARIO: {scenario_name}")
        print(f"{'='*80}")
        
        # Create prompt
        print(f"📝 Creating {scenario_type} prompt...")
        prompt = create_sports_prompt(game_data, scenario_type)
        
        # Generate report
        report = generate_report(model, tokenizer, prompt, scenario_name)
        
        # Display results
        print(f"\n🏀 GENERATED SPORTS REPORT ({scenario_name})")
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
        
        # Memory cleanup between scenarios
        cleanup_memory()

if __name__ == "__main__":
    main() 