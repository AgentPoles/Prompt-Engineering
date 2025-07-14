#!/usr/bin/env python3
"""
Comprehensive Basketball Game Analysis with Backward Attention
Processes all 19 games with 3 prompt types and analyzes attention patterns
Designed for Colab Enterprise with neat code blocks
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
from datetime import datetime
import time

# ================================================================================
# 📋 BLOCK 1: CONFIGURATION AND SETUP
# ================================================================================

# Configuration
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
BASKETBALL_GAMES_DIR = "basketball_games"
NUM_GAMES = 19
SELECTED_GAMES_FOR_ATTENTION = [0, 5, 10, 15, 18]  # Representative games
MAX_RESPONSE_TOKENS = 600  # Target 350-450 words

# File paths
PROMPT_FILES = {
    'reflection': 'prompt_with_reflection.txt',
    'no_reflection': 'prompt_with_noreflection.txt',
    'dual_identity': 'prompt_with_dual_identity.txt'
}

print("🏀 Basketball Game Analysis Configuration")
print(f"Model: {MODEL_ID}")
print(f"Games to process: {NUM_GAMES}")
print(f"Games for attention analysis: {SELECTED_GAMES_FOR_ATTENTION}")
print(f"Target response length: {MAX_RESPONSE_TOKENS} tokens")

# ================================================================================
# 📦 BLOCK 2: MEMORY MANAGEMENT UTILITIES
# ================================================================================

def clear_memory():
    """Clear GPU and system memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def show_memory_status():
    """Display current memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

print("🧹 Memory management utilities loaded")

# ================================================================================
# 🤖 BLOCK 3: MODEL LOADING
# ================================================================================

print(f"\n🚀 Loading {MODEL_ID}...")
clear_memory()

try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
        ),
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",  # Required for attention analysis
        low_cpu_mem_usage=True
    )
    
    print("✅ Model loaded successfully!")
    show_memory_status()
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# ================================================================================
# 📁 BLOCK 4: PROMPT LOADING
# ================================================================================

print("\n📝 Loading prompt templates...")

prompts = {}
for prompt_type, filename in PROMPT_FILES.items():
    try:
        with open(filename, 'r') as f:
            prompts[prompt_type] = f.read()
        print(f"✅ Loaded {prompt_type} prompt ({len(prompts[prompt_type])} characters)")
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")
        raise

# ================================================================================
# 🏀 BLOCK 5: GAME DATA EXTRACTION UTILITIES
# ================================================================================

def extract_game_info(data):
    """Extract concise game information from JSON data"""
    
    # Get basic game info
    date_info = data.get('date', {})
    location_info = data.get('location', {})
    teams = data.get('teams', {})
    
    # Handle team structure
    team_keys = list(teams.keys())
    if len(team_keys) == 2:
        team1_key, team2_key = team_keys
        team1 = teams[team1_key]
        team2 = teams[team2_key]
    else:
        raise ValueError(f"Unexpected team structure: {team_keys}")
    
    # Build game info string
    game_info = f"""GAME SUMMARY:
Date: {date_info.get('monthname', 'N/A')} {date_info.get('day', 'N/A')}, {date_info.get('year', 'N/A')}
Location: {location_info.get('stadium', 'N/A')}, {location_info.get('place', 'N/A')}"""
    
    # Team information and scores
    team1_score = team1.get('statistics', {}).get('team', {}).get('game', {}).get('pts', 0)
    team2_score = team2.get('statistics', {}).get('team', {}).get('game', {}).get('pts', 0)
    
    game_info += f"""
Teams: {team1.get('team_name', 'Team1')} ({team1.get('record', {}).get('wins', 0)}-{team1.get('record', {}).get('losses', 0)}) vs {team2.get('team_name', 'Team2')} ({team2.get('record', {}).get('wins', 0)}-{team2.get('record', {}).get('losses', 0)})
Final Score: {team1.get('team_name', 'Team1')} {team1_score} - {team2.get('team_name', 'Team2')} {team2_score}
Winner: {team1.get('team_name', 'Team1') if team1_score > team2_score else team2.get('team_name', 'Team2')}"""
    
    # Quarter scores
    periods1 = team1.get('statistics', {}).get('team', {}).get('period', {})
    periods2 = team2.get('statistics', {}).get('team', {}).get('period', {})
    if periods1 and periods2:
        game_info += f"""

QUARTER-BY-QUARTER SCORING:
Q1: {team1.get('team_name', 'Team1')} {periods1.get('1', {}).get('pts', 0)} - {team2.get('team_name', 'Team2')} {periods2.get('1', {}).get('pts', 0)}
Q2: {team1.get('team_name', 'Team1')} {periods1.get('2', {}).get('pts', 0)} - {team2.get('team_name', 'Team2')} {periods2.get('2', {}).get('pts', 0)}
Q3: {team1.get('team_name', 'Team1')} {periods1.get('3', {}).get('pts', 0)} - {team2.get('team_name', 'Team2')} {periods2.get('3', {}).get('pts', 0)}
Q4: {team1.get('team_name', 'Team1')} {periods1.get('4', {}).get('pts', 0)} - {team2.get('team_name', 'Team2')} {periods2.get('4', {}).get('pts', 0)}"""
    
    # Team statistics
    team1_stats = team1.get('statistics', {}).get('team', {}).get('game', {})
    team2_stats = team2.get('statistics', {}).get('team', {}).get('game', {})
    
    if team1_stats and team2_stats:
        fg1_pct = round((team1_stats.get('fgm', 0) / max(team1_stats.get('fga', 1), 1)) * 100, 1)
        fg2_pct = round((team2_stats.get('fgm', 0) / max(team2_stats.get('fga', 1), 1)) * 100, 1)
        
        game_info += f"""

TEAM STATISTICS:
{team1.get('team_name', 'Team1')}: {fg1_pct}% FG ({team1_stats.get('fgm', 0)}/{team1_stats.get('fga', 0)}), {team1_stats.get('reb', 0)} REB, {team1_stats.get('ast', 0)} AST, {team1_stats.get('tov', 0)} TOV
{team2.get('team_name', 'Team2')}: {fg2_pct}% FG ({team2_stats.get('fgm', 0)}/{team2_stats.get('fga', 0)}), {team2_stats.get('reb', 0)} REB, {team2_stats.get('ast', 0)} AST, {team2_stats.get('tov', 0)} TOV"""
    
    # Top players from both teams
    players1 = team1.get('statistics', {}).get('players', {})
    players2 = team2.get('statistics', {}).get('players', {})
    
    if players1:
        top_players1 = sorted(players1.items(), key=lambda x: x[1].get('pts', 0), reverse=True)[:3]
        game_info += f"""

PLAYER PERFORMANCES:
{team1.get('team_name', 'Team1')} Top Scorers:"""
        for name, stats in top_players1:
            pts = stats.get('pts', 0)
            reb = stats.get('reb', 0)
            ast = stats.get('ast', 0)
            game_info += f"\n- {name}: {pts} PTS, {reb} REB, {ast} AST"
    
    if players2:
        top_players2 = sorted(players2.items(), key=lambda x: x[1].get('pts', 0), reverse=True)[:3]
        game_info += f"""

{team2.get('team_name', 'Team2')} Top Scorers:"""
        for name, stats in top_players2:
            pts = stats.get('pts', 0)
            reb = stats.get('reb', 0)
            ast = stats.get('ast', 0)
            game_info += f"\n- {name}: {pts} PTS, {reb} REB, {ast} AST"
    
    return game_info

def create_prompt_with_data(template, game_info):
    """Create complete prompt with game data"""
    # Replace the JSON reference with actual data
    modified_template = template.replace(
        "Game data is provided in a JSON file called data.json. Ensure every statement is grounded exclusively in this data.",
        "Game data is provided below. Ensure every statement is grounded exclusively in this data."
    )
    
    full_prompt = f"{modified_template.strip()}\n\n{game_info}"
    
    # Apply chat template
    try:
        messages = [{"role": "user", "content": full_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        return f"<|user|>\n{full_prompt}\n<|assistant|>\n"

print("🔧 Game data extraction utilities loaded")

# ================================================================================
# 🎯 BLOCK 6: TEXT GENERATION UTILITIES
# ================================================================================

def generate_report(prompt, max_tokens=MAX_RESPONSE_TOKENS):
    """Generate basketball report from prompt"""
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    print(f"    Input tokens: {inputs['input_ids'].shape[1]}")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return result.strip()

def count_words(text):
    """Count actual words in text"""
    return len(text.split())

def process_single_game(game_num, game_data):
    """Process a single game with all three prompt types"""
    
    print(f"\n📊 Processing Game {game_num:02d}...")
    
    # Extract game info
    game_info = extract_game_info(game_data)
    
    # Create results directory
    results_dir = os.path.join(BASKETBALL_GAMES_DIR, f"{game_num:02d}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Process each prompt type
    results = {}
    all_results_text = []
    
    for prompt_type in ['reflection', 'no_reflection', 'dual_identity']:
        print(f"  🎯 Processing {prompt_type}...")
        
        # Create prompt
        prompt = create_prompt_with_data(prompts[prompt_type], game_info)
        
        # Generate report
        start_time = time.time()
        report = generate_report(prompt)
        generation_time = time.time() - start_time
        
        # Count words
        word_count = count_words(report)
        
        # Store results
        results[prompt_type] = {
            'report': report,
            'word_count': word_count,
            'generation_time': generation_time
        }
        
        # Add to combined results
        all_results_text.append(f"{'='*60}")
        all_results_text.append(f"PROMPT TYPE: {prompt_type.upper()}")
        all_results_text.append(f"WORD COUNT: {word_count}")
        all_results_text.append(f"GENERATION TIME: {generation_time:.2f}s")
        all_results_text.append(f"{'='*60}")
        all_results_text.append(report)
        all_results_text.append("")
        
        print(f"    ✅ Generated {word_count} words in {generation_time:.2f}s")
        
        # Clear memory after each generation
        clear_memory()
    
    # Save combined results
    results_file = os.path.join(results_dir, "results.txt")
    with open(results_file, 'w') as f:
        f.write('\n'.join(all_results_text))
    
    print(f"  💾 Results saved to {results_file}")
    
    return results

print("📝 Text generation utilities loaded")

# ================================================================================
# 🎮 BLOCK 7: MAIN PROCESSING LOOP
# ================================================================================

def process_all_games():
    """Process all basketball games with all prompt types"""
    
    print(f"\n🚀 Starting processing of {NUM_GAMES} games...")
    
    total_results = {}
    processing_stats = {
        'total_games': 0,
        'total_reports': 0,
        'avg_word_counts': {'reflection': [], 'no_reflection': [], 'dual_identity': []},
        'avg_generation_times': {'reflection': [], 'no_reflection': [], 'dual_identity': []},
        'errors': []
    }
    
    for game_num in range(NUM_GAMES):
        try:
            # Load game data
            game_file = os.path.join(BASKETBALL_GAMES_DIR, f"{game_num:02d}", "data.json")
            
            if not os.path.exists(game_file):
                print(f"⚠️ Game {game_num:02d} data not found, skipping...")
                continue
            
            with open(game_file, 'r') as f:
                game_data = json.load(f)
            
            # Process the game
            results = process_single_game(game_num, game_data)
            total_results[game_num] = results
            
            # Update stats
            processing_stats['total_games'] += 1
            processing_stats['total_reports'] += 3
            
            for prompt_type in ['reflection', 'no_reflection', 'dual_identity']:
                processing_stats['avg_word_counts'][prompt_type].append(results[prompt_type]['word_count'])
                processing_stats['avg_generation_times'][prompt_type].append(results[prompt_type]['generation_time'])
            
        except Exception as e:
            error_msg = f"Game {game_num:02d}: {str(e)}"
            processing_stats['errors'].append(error_msg)
            print(f"❌ Error processing game {game_num:02d}: {e}")
            continue
    
    # Print summary statistics
    print(f"\n📊 PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Games processed: {processing_stats['total_games']}")
    print(f"Reports generated: {processing_stats['total_reports']}")
    print(f"Errors encountered: {len(processing_stats['errors'])}")
    
    print(f"\n📈 WORD COUNT STATISTICS:")
    for prompt_type in ['reflection', 'no_reflection', 'dual_identity']:
        word_counts = processing_stats['avg_word_counts'][prompt_type]
        if word_counts:
            avg_words = np.mean(word_counts)
            min_words = np.min(word_counts)
            max_words = np.max(word_counts)
            print(f"  {prompt_type}: {avg_words:.1f} avg ({min_words}-{max_words} range)")
    
    print(f"\n⏱️ GENERATION TIME STATISTICS:")
    for prompt_type in ['reflection', 'no_reflection', 'dual_identity']:
        gen_times = processing_stats['avg_generation_times'][prompt_type]
        if gen_times:
            avg_time = np.mean(gen_times)
            print(f"  {prompt_type}: {avg_time:.2f}s avg")
    
    if processing_stats['errors']:
        print(f"\n❌ ERRORS:")
        for error in processing_stats['errors']:
            print(f"  {error}")
    
    return total_results, processing_stats

print("🔄 Main processing loop ready")

# ================================================================================
# 🧠 BLOCK 8: BACKWARD ATTENTION ANALYSIS
# ================================================================================

class BackwardAttentionAnalyzer:
    """Analyze backward attention patterns for basketball reports"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.reflection_keywords = [
            'reflect', 'review', 'check', 'assess', 'evaluate', 
            'analyze', 'verify', 'reconsider', 'double-check', 'confirm'
        ]
    
    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def generate_with_attention_tracking(self, prompt, max_new_tokens=100):
        """Generate text step by step while tracking attention patterns"""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1000,
            padding=False
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        
        # Track generation progress
        generation_data = {
            'input_length': input_ids.shape[1],
            'steps': [],
            'tokens': input_ids[0].tolist(),
            'generated_tokens': [],
        }
        
        # Generate tokens one by one
        current_ids = input_ids.clone()
        
        for step in range(max_new_tokens):
            # Clear memory before each step
            self.clear_memory()
            
            with torch.no_grad():
                outputs = self.model(
                    current_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    use_cache=False
                )
            
            # Get next token
            logits = outputs.logits[0, -1, :]
            next_token_id = torch.multinomial(torch.softmax(logits / 0.7, dim=-1), 1)
            
            # Update sequences
            current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0)], dim=1)
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones(1, 1, device=attention_mask.device)
            ], dim=1)
            
            # Store generation data
            new_token = next_token_id.item()
            generation_data['generated_tokens'].append(new_token)
            generation_data['tokens'].append(new_token)
            
            # Calculate backward attention immediately
            if outputs.attentions:
                backward_metrics = self.calculate_backward_attention(
                    outputs.attentions,
                    generation_data['input_length'],
                    len(generation_data['generated_tokens'])
                )
                del outputs.attentions
            else:
                backward_metrics = {
                    'backward_attention_ratio': 0.0,
                    'attention_to_generated': 0.0,
                    'attention_to_input': 0.0,
                }
            
            # Check if we hit reflection keywords
            new_token_text = self.tokenizer.decode([new_token])
            is_reflection_step = any(keyword in new_token_text.lower() for keyword in self.reflection_keywords)
            
            step_data = {
                'step': step,
                'token': new_token,
                'token_text': new_token_text,
                'is_reflection_step': is_reflection_step,
                'backward_metrics': backward_metrics,
                'total_length': len(generation_data['tokens'])
            }
            
            generation_data['steps'].append(step_data)
            
            # Clear intermediate variables
            del outputs, logits, next_token_id
            
            # Stop if we hit EOS
            if new_token == self.tokenizer.eos_token_id:
                break
        
        return generation_data
    
    def calculate_backward_attention(self, attentions, input_length, generated_length):
        """Calculate metrics for backward attention patterns"""
        
        if not attentions or len(attentions) == 0:
            return {
                'backward_attention_ratio': 0.0,
                'attention_to_generated': 0.0,
                'attention_to_input': 0.0,
            }
        
        try:
            # Get last layer attention (often most meaningful)
            last_layer_attention = attentions[-1][0]  # [num_heads, seq_len, seq_len]
            
            # Average across heads
            avg_attention = torch.mean(last_layer_attention, dim=0)  # [seq_len, seq_len]
            
            # Get attention from current token (last position) to all previous tokens
            current_attention = avg_attention[-1, :].float()  # [seq_len]
            
            # Split attention into different regions
            attention_to_input = current_attention[:input_length].sum().item()
            
            if generated_length > 0:
                attention_to_generated = current_attention[input_length:input_length + generated_length].sum().item()
            else:
                attention_to_generated = 0.0
            
            # Calculate backward attention ratio
            total_attention = attention_to_input + attention_to_generated
            if total_attention > 0:
                backward_attention_ratio = attention_to_generated / total_attention
            else:
                backward_attention_ratio = 0.0
            
            return {
                'backward_attention_ratio': backward_attention_ratio,
                'attention_to_generated': attention_to_generated,
                'attention_to_input': attention_to_input
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating backward attention: {e}")
            return {
                'backward_attention_ratio': 0.0,
                'attention_to_generated': 0.0,
                'attention_to_input': 0.0
            }
    
    def analyze_game_attention(self, game_num, game_data):
        """Analyze attention patterns for a single game across all prompt types"""
        
        print(f"\n🧠 Analyzing attention patterns for Game {game_num:02d}...")
        
        # Extract game info
        game_info = extract_game_info(game_data)
        
        # Analyze each prompt type
        results = {}
        
        for prompt_type in ['reflection', 'no_reflection', 'dual_identity']:
            print(f"  🎯 Analyzing {prompt_type} attention...")
            
            # Create prompt
            prompt = create_prompt_with_data(prompts[prompt_type], game_info)
            
            # Generate with attention tracking
            attention_data = self.generate_with_attention_tracking(prompt, max_new_tokens=50)
            
            # Analyze patterns
            analysis = self.analyze_backward_patterns(attention_data)
            
            results[prompt_type] = {
                'attention_data': attention_data,
                'analysis': analysis
            }
            
            print(f"    ✅ Avg backward ratio: {analysis['avg_backward_ratio']:.4f}")
            
            # Clear memory after each analysis
            self.clear_memory()
        
        return results
    
    def analyze_backward_patterns(self, generation_data):
        """Analyze backward attention patterns in generation data"""
        
        backward_ratios = []
        attention_to_generated = []
        reflection_steps = []
        
        for step_data in generation_data['steps']:
            metrics = step_data['backward_metrics']
            backward_ratios.append(metrics['backward_attention_ratio'])
            attention_to_generated.append(metrics['attention_to_generated'])
            
            if step_data['is_reflection_step']:
                reflection_steps.append({
                    'step': step_data['step'],
                    'token': step_data['token_text'],
                    'backward_ratio': metrics['backward_attention_ratio'],
                    'attention_to_generated': metrics['attention_to_generated']
                })
        
        return {
            'avg_backward_ratio': np.mean(backward_ratios) if backward_ratios else 0.0,
            'max_backward_ratio': np.max(backward_ratios) if backward_ratios else 0.0,
            'avg_attention_to_generated': np.mean(attention_to_generated) if attention_to_generated else 0.0,
            'reflection_spikes': reflection_steps,
            'total_steps': len(generation_data['steps'])
        }

def run_attention_analysis():
    """Run backward attention analysis on selected games"""
    
    print(f"\n🧠 Starting backward attention analysis...")
    print(f"Selected games: {SELECTED_GAMES_FOR_ATTENTION}")
    
    analyzer = BackwardAttentionAnalyzer(model, tokenizer)
    attention_results = {}
    
    for game_num in SELECTED_GAMES_FOR_ATTENTION:
        try:
            # Load game data
            game_file = os.path.join(BASKETBALL_GAMES_DIR, f"{game_num:02d}", "data.json")
            
            if not os.path.exists(game_file):
                print(f"⚠️ Game {game_num:02d} data not found, skipping attention analysis...")
                continue
            
            with open(game_file, 'r') as f:
                game_data = json.load(f)
            
            # Analyze attention patterns
            game_attention = analyzer.analyze_game_attention(game_num, game_data)
            attention_results[game_num] = game_attention
            
        except Exception as e:
            print(f"❌ Error analyzing attention for game {game_num:02d}: {e}")
            continue
    
    # Aggregate results across games
    print(f"\n📊 ATTENTION ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for prompt_type in ['reflection', 'no_reflection', 'dual_identity']:
        ratios = []
        spike_counts = []
        
        for game_num, game_results in attention_results.items():
            if prompt_type in game_results:
                analysis = game_results[prompt_type]['analysis']
                ratios.append(analysis['avg_backward_ratio'])
                spike_counts.append(len(analysis['reflection_spikes']))
        
        if ratios:
            avg_ratio = np.mean(ratios)
            avg_spikes = np.mean(spike_counts)
            print(f"{prompt_type}: {avg_ratio:.4f} avg backward ratio, {avg_spikes:.1f} avg spikes")
    
    # Save attention results
    with open("attention_analysis_results.json", "w") as f:
        # Convert complex objects to JSON-serializable format
        json_results = {}
        for game_num, game_results in attention_results.items():
            json_results[str(game_num)] = {}
            for prompt_type, data in game_results.items():
                json_results[str(game_num)][prompt_type] = {
                    'avg_backward_ratio': data['analysis']['avg_backward_ratio'],
                    'reflection_spikes': data['analysis']['reflection_spikes'],
                    'total_steps': data['analysis']['total_steps']
                }
        json.dump(json_results, f, indent=2)
    
    print(f"💾 Attention analysis results saved to 'attention_analysis_results.json'")
    
    return attention_results

print("🧠 Backward attention analysis utilities loaded")

# ================================================================================
# 🎯 BLOCK 9: EXECUTION CONTROL
# ================================================================================

def main():
    """Main execution function"""
    
    print(f"\n🚀 STARTING COMPREHENSIVE BASKETBALL ANALYSIS")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Phase 1: Process all games with text generation
    print(f"\n📝 PHASE 1: TEXT GENERATION FOR ALL GAMES")
    print(f"{'='*60}")
    
    text_results, processing_stats = process_all_games()
    
    # Phase 2: Attention analysis on selected games
    print(f"\n🧠 PHASE 2: ATTENTION ANALYSIS ON SELECTED GAMES")
    print(f"{'='*60}")
    
    attention_results = run_attention_analysis()
    
    # Final summary
    print(f"\n✅ ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Games processed: {processing_stats['total_games']}")
    print(f"Reports generated: {processing_stats['total_reports']}")
    print(f"Attention analyses: {len(attention_results)}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return text_results, attention_results

print("🎯 Execution control ready")
print("\n" + "="*80)
print("📋 READY TO RUN ANALYSIS")
print("="*80)
print("To start the analysis, run: main()")
print("This will process all games and run attention analysis on selected games")
print("="*80)

if __name__ == "__main__":
    # Uncomment the line below to run automatically
    # main()
    pass 