#!/usr/bin/env python3
"""
Qwen 32B Sports Report Generator - Handles Actual JSON Data Structure
Optimized for limited data scenarios with adaptive prompts
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import warnings
warnings.filterwarnings('ignore')

def aggressive_memory_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)  # GB
    return 0

def load_model_with_fallback():
    """Load model with fallback options"""
    print("🚀 Loading Qwen 32B with 4-bit quantization...")
    
    # 4-bit quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model_id = "Qwen/Qwen2.5-32B-Instruct"
    
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    print("🔧 Loading model with 4-bit quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        )
        print("✅ Qwen 32B loaded successfully (with flash attention)!")
    except Exception as e:
        print("❌ Failed to load model with flash attention, trying without...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        print("✅ Qwen 32B loaded successfully (without flash attention)!")
    
    return model, tokenizer

def analyze_json_structure(data):
    """Analyze and extract available data from JSON"""
    available_data = {
        'has_game_info': False,
        'has_team_records': False,
        'has_player_stats': False,
        'has_team_stats': False,
        'has_quarters': False,
        'game_info': {},
        'teams': {}
    }
    
    print(f"🔍 DEBUG: Full JSON structure keys: {list(data.keys())}")
    
    # Check game info
    if 'game_id' in data or 'date' in data or 'location' in data:
        available_data['has_game_info'] = True
        available_data['game_info'] = {
            'game_id': data.get('game_id', 'Unknown'),
            'date': data.get('date', 'Unknown'),
            'location': data.get('location', 'Unknown'),
            'overtime': data.get('overtime', False)
        }
    
    # Check teams
    if 'teams' in data:
        teams = data['teams']
        print(f"🔍 DEBUG: Teams keys: {list(teams.keys())}")
        
        for team_type in ['home', 'visitor']:
            if team_type in teams:
                team = teams[team_type]
                print(f"🔍 DEBUG: Team '{team_type}' structure: {list(team.keys())}")
                
                # Check for team records
                if 'record' in team:
                    available_data['has_team_records'] = True
                
                # Check for player data
                player_found = False
                for player_key in ['players', 'roster', 'player_stats']:
                    if player_key in team and team[player_key]:
                        player_found = True
                        break
                
                if not player_found and 'statistics' in team:
                    stats = team['statistics']
                    for stat_key in ['players', 'player_stats']:
                        if stat_key in stats and stats[stat_key]:
                            player_found = True
                            break
                
                if player_found:
                    available_data['has_player_stats'] = True
                    print(f"🔍 DEBUG: Team '{team_type}' - player data found")
                else:
                    print(f"🔍 DEBUG: Team '{team_type}' - no players/roster found")
                    print(f"🔍 DEBUG: Available keys: {list(team.keys())}")
                
                # Store team data
                available_data['teams'][team_type] = {
                    'name': team.get('team_name', 'Unknown'),
                    'place': team.get('team_place', 'Unknown'),
                    'code': team.get('team_code', 'Unknown'),
                    'record': team.get('record', 'Unknown')
                }
    
    return available_data

def create_adaptive_prompt(data_analysis, include_reflection=False):
    """Create adaptive prompt based on available data"""
    
    # Base identity and instructions
    identity = """# Identity
You are a seasoned sports journalist. Your role is to craft a vivid and structured report of an NBA game that captures the energy, key moments, and standout performances, engaging an audience of passionate basketball fans with the tone and style of an experienced reporter."""

    # Reflection section (if requested)
    reflection = ""
    if include_reflection:
        reflection = """

# Internal Reflection Process
Before finalizing your report, take time to carefully reflect on your draft:
- Internally review whether your output fully complies with all instructions, layout, style, and factual requirements.
- Double-check every fact, statistic, and player detail strictly against the provided JSON data.
- Verify that your word count is between 350 and 450 words.
- Ensure your narrative is lively yet professional, with no slang, and that you have avoided any editorial phrases not directly supported by data.
- If you discover any issues, fix them internally and review again.

Only when you are fully confident that your report is flawless and meets all expectations should you print the final version.  
Do not print any of your internal reflection or checks — only produce the final polished report."""

    # Adaptive layout based on available data
    layout = """

# Instructions
## Layout
Organize the game report using this sequence:
● Opening Outcome: state the final score, winning team, date, and venue. Introduce both teams by name and include each team's current win-loss record."""

    # Add sections based on available data
    if not data_analysis['has_player_stats'] and not data_analysis['has_team_stats']:
        layout += """
● Game Context: Since detailed statistics are not available, focus on the broader context of the matchup - discuss the significance of the teams' records, the venue, and what this game means for both teams' seasons.
● Team Records Analysis: Elaborate on how each team's current record reflects their season performance and what this matchup represents for their playoff aspirations or rebuilding efforts.
● Competitive Narrative: Describe the general competitive dynamic between teams with these records, the expected style of play, and the strategic implications."""
    else:
        layout += """
● Game Flow Highlights: describe how the game unfolded, noting early leads, comebacks, and crucial quarters based on factual computations from the JSON only. Quantify shifts using explicit scores (halftime margins, point differentials). Avoid subjective phrases like "built momentum" or "seized control" — rely purely on scores.
● Team Performance Report: summarize critical team stats (FG%, 3P%, rebounds, turnovers). Mention other stats only if clearly impactful.
● Player Highlights: spotlight top individual performances (up to 3 players per team), listing points, rebounds, assists, steals, blocks. State double- or triple-doubles factually without subjective language."""

    layout += """
● Closing Note (Optional): mention upcoming opponents only if in the JSON. If absent, omit entirely without saying it's missing."""

    # Directive adapted for available data
    directive = """

## Directive
Base all writing exclusively on the structured JSON data provided. Strictly enforce:
● Word count between 350 and 450 real words. Reports under 350 are incomplete.
● Write as a single continuous block of text (no paragraph breaks) unless explicitly instructed otherwise.
● Maintain a lively yet professional tone appropriate for sports journalism. Exclude slang. Report facts precisely.
● CRITICAL: Only use information that actually exists in the JSON data. Do not invent player statistics, game scores, or team performance data.
● If specific game statistics are not available, focus on team records, game context, and broader season narratives.
● Use only ASCII characters in your output - avoid special Unicode characters.
● Ensure the report flows naturally and chronologically from opening through available data to closing."""

    # Context section
    context = """

# Context
Game data is provided in a JSON file. The available data includes team records and basic game information. Ensure every statement is grounded exclusively in this data."""

    # Final output section
    output_instruction = """

# Final Output
"""
    if include_reflection:
        output_instruction += "Once you are fully satisfied that your report is flawless and meets all requirements, print only the final approved game report."
    else:
        output_instruction += "Print the complete, polished game report following these instructions."

    return identity + reflection + layout + directive + context + output_instruction

def format_game_data_for_prompt(data_analysis):
    """Format available game data for the prompt"""
    prompt_data = "Game Data:\n"
    
    # Game info
    if data_analysis['has_game_info']:
        game_info = data_analysis['game_info']
        prompt_data += f"Date: {game_info['date']}\n"
        prompt_data += f"Location: {game_info['location']}\n"
        prompt_data += f"Game ID: {game_info['game_id']}\n"
        if game_info['overtime']:
            prompt_data += "Overtime: Yes\n"
    
    # Team info
    if data_analysis['teams']:
        prompt_data += "\nTeams:\n"
        for team_type, team_data in data_analysis['teams'].items():
            prompt_data += f"{team_type.capitalize()}: {team_data['name']} ({team_data['record']})\n"
    
    # Data availability note
    prompt_data += f"\nAvailable Data: Basic game information and team records only. No detailed player statistics or game flow data available."
    
    return prompt_data

def generate_report(model, tokenizer, prompt, max_length=600):
    """Generate sports report with better parameters"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000)
    input_length = inputs['input_ids'].shape[1]
    print(f"📊 Input tokens: {input_length}")
    
    print(f"📊 Memory before generation: {get_memory_usage():.1f}GB")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'].to(model.device),
            attention_mask=inputs['attention_mask'].to(model.device),
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part
    prompt_length = len(prompt)
    report = generated_text[prompt_length:].strip()
    
    return report

def count_words(text):
    """Count actual words in text"""
    words = text.split()
    return len([word for word in words if word.strip()])

def main():
    print("=" * 80)
    print("🏀 QWEN 32B - ADAPTIVE DATA STRUCTURE VERSION")
    print("=" * 80)
    
    # Memory cleanup
    print("🧹 Aggressive memory cleanup...")
    aggressive_memory_cleanup()
    print(f"🔍 Available GPU memory: {get_memory_usage():.1f}GB")
    
    # Load model
    model, tokenizer = load_model_with_fallback()
    print(f"📊 GPU memory used: {get_memory_usage():.1f}GB")
    
    # Load and analyze data
    print("📊 Loading and analyzing game data structure...")
    try:
        with open('game_1_data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ game_1_data.json not found!")
        return
    
    # Analyze available data
    data_analysis = analyze_json_structure(data)
    
    # Format data for prompt
    formatted_data = format_game_data_for_prompt(data_analysis)
    
    # Run both scenarios
    scenarios = [
        ("WITH REFLECTION", True),
        ("WITHOUT REFLECTION", False)
    ]
    
    for scenario_name, include_reflection in scenarios:
        print("\n" + "=" * 80)
        print(f"🏀 SCENARIO: {scenario_name}")
        print("=" * 80)
        
        # Create adaptive prompt
        print(f"📝 Creating adaptive prompt for {scenario_name.lower()}...")
        prompt_template = create_adaptive_prompt(data_analysis, include_reflection)
        
        # Combine prompt with data
        full_prompt = prompt_template + "\n\n" + formatted_data + "\n\nReport:\n"
        
        # Generate report
        print(f"🏀 Generating sports report with Qwen 32B ({scenario_name})...")
        report = generate_report(model, tokenizer, full_prompt)
        
        # Display results
        print(f"\n🏀 GENERATED SPORTS REPORT ({scenario_name})")
        print("=" * 80)
        print(report)
        print("=" * 80)
        
        # Word count check
        word_count = count_words(report)
        print(f"📊 Word count: {word_count}")
        print(f"📊 Target range: 350-450 words")
        if 350 <= word_count <= 450:
            print("✅ Word count within target range!")
        else:
            print("⚠️ Word count outside target range")
        
        # Memory cleanup between scenarios
        print("🧹 Aggressive memory cleanup...")
        aggressive_memory_cleanup()
        print(f"🔍 Available GPU memory: {get_memory_usage():.1f}GB")

if __name__ == "__main__":
    main() 