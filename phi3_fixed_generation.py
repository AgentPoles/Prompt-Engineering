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
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Use flash attention if available
        low_cpu_mem_usage=True,
        max_memory={0: "30GB"}  # Limit GPU memory usage
    )
    
    # Enable memory efficient attention
    model.config.use_cache = False
    
    print(f"✅ Model loaded with {torch.cuda.memory_allocated() / 1024**3:.1f}GB GPU memory")
    return model, tokenizer

def create_phi3_prompt(system_prompt, user_message):
    """Create properly formatted prompt for Phi-3"""
    # Phi-3 uses a specific chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    return messages

def generate_with_phi3(model, tokenizer, messages, max_new_tokens=500):
    """Generate text with Phi-3 using proper chat template"""
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=4000  # Leave room for generation
    ).to(model.device)
    
    print(f"📊 Input tokens: {inputs.input_ids.shape[1]}")
    print(f"📊 Memory before generation: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
    
    # Generate with optimized parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            use_cache=False  # Disable cache to save memory
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def create_simplified_prompt():
    """Create a simplified, more direct prompt for Phi-3"""
    return """You are a professional sports journalist. Write a basketball game report based on the provided JSON data.

REQUIREMENTS:
- Write 350-450 words
- Use professional sports journalism tone
- Include final score, date, venue
- Describe game flow using actual scores
- Highlight team stats (FG%, 3P%, rebounds, turnovers)
- Feature top 3 players per team with their stats
- Write as one continuous paragraph
- Base everything on the provided data only

Write the report now:"""

def condense_json_data(json_data):
    """Further condense JSON data to focus on essential elements"""
    condensed = {
        "game_info": {
            "date": json_data["date"],
            "venue": json_data["venue"],
            "home_team": json_data["home_team"]["name"],
            "away_team": json_data["away_team"]["name"],
            "home_score": json_data["home_team"]["score"],
            "away_score": json_data["away_team"]["score"],
            "home_record": json_data["home_team"]["record"],
            "away_record": json_data["away_team"]["record"]
        },
        "quarter_scores": json_data["quarter_scores"],
        "team_stats": {
            "home": {
                "fg_pct": json_data["home_team"]["team_stats"]["fg_pct"],
                "fg3_pct": json_data["home_team"]["team_stats"]["fg3_pct"],
                "rebounds": json_data["home_team"]["team_stats"]["rebounds"],
                "turnovers": json_data["home_team"]["team_stats"]["turnovers"],
                "assists": json_data["home_team"]["team_stats"]["assists"]
            },
            "away": {
                "fg_pct": json_data["away_team"]["team_stats"]["fg_pct"],
                "fg3_pct": json_data["away_team"]["team_stats"]["fg3_pct"],
                "rebounds": json_data["away_team"]["team_stats"]["rebounds"],
                "turnovers": json_data["away_team"]["team_stats"]["turnovers"],
                "assists": json_data["away_team"]["team_stats"]["assists"]
            }
        },
        "top_players": {}
    }
    
    # Get top 3 players from each team by points
    for team in ["home_team", "away_team"]:
        team_players = json_data[team]["players"]
        # Sort by points and take top 3
        top_players = sorted(team_players.items(), key=lambda x: x[1]["pts"], reverse=True)[:3]
        
        condensed["top_players"][team] = {}
        for player_name, stats in top_players:
            condensed["top_players"][team][player_name] = {
                "pts": stats["pts"],
                "reb": stats["reb"],
                "ast": stats["ast"],
                "stl": stats["stl"],
                "blk": stats["blk"],
                "fgm": stats["fgm"],
                "fga": stats["fga"],
                "fg3m": stats["fg3m"],
                "fg3a": stats["fg3a"]
            }
    
    return condensed

def main():
    try:
        # Load model
        model, tokenizer = load_optimized_model()
        
        # Load and condense JSON data
        with open('data.json', 'r') as f:
            json_data = json.load(f)
        
        condensed_data = condense_json_data(json_data)
        json_str = json.dumps(condensed_data, indent=2)
        
        # Create simplified prompt
        system_prompt = create_simplified_prompt()
        user_message = f"Here is the basketball game data:\n\n{json_str}"
        
        # Create messages
        messages = create_phi3_prompt(system_prompt, user_message)
        
        print("="*80)
        print("🏀 PHI-3-MEDIUM - SIMPLIFIED PROMPT TEST")
        print("="*80)
        
        # Generate report
        report = generate_with_phi3(model, tokenizer, messages, max_new_tokens=600)
        
        print("\n📰 GENERATED REPORT:")
        print("-" * 50)
        print(report)
        print("-" * 50)
        
        # Count words
        word_count = len(report.split())
        print(f"📊 Word count: {word_count}")
        
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