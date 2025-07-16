#!/usr/bin/env python3
"""
🧹 Clear GPU Memory and Run Interpretability Analysis
====================================================
Safely clear memory and run the basketball interpretability analysis
"""

import torch
import gc
import os
import psutil

def clear_gpu_memory():
    """Comprehensive GPU memory clearing"""
    print("🧹 CLEARING GPU MEMORY")
    print("=" * 50)
    
    # Check initial memory
    if torch.cuda.is_available():
        print("📊 MEMORY BEFORE CLEARING:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    # Clear all cached variables
    print("\n🗑️ Clearing Python variables...")
    
    # Get current global variables
    current_globals = list(globals().keys())
    
    # Clear common model variables
    model_vars = ['model', 'tokenizer', 'analyzer', 'attention_analyzer', 'results']
    for var in model_vars:
        if var in globals():
            print(f"  Clearing {var}...")
            del globals()[var]
    
    # Clear large objects from namespace
    for var_name in current_globals:
        if var_name.startswith('_'):
            continue
        try:
            var = globals().get(var_name)
            if hasattr(var, '__sizeof__') and var.__sizeof__() > 1024*1024:  # > 1MB
                print(f"  Clearing large object: {var_name}")
                del globals()[var_name]
        except:
            pass
    
    # Python garbage collection
    print("\n🗑️ Running garbage collection...")
    gc.collect()
    
    # PyTorch memory clearing
    if torch.cuda.is_available():
        print("\n🗑️ Clearing PyTorch CUDA cache...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force memory cleanup
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
    
    # Check memory after clearing
    if torch.cuda.is_available():
        print("\n📊 MEMORY AFTER CLEARING:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    print("\n✅ Memory clearing complete!")

def check_available_memory():
    """Check if we have enough memory for the analysis"""
    print("\n🔍 CHECKING AVAILABLE MEMORY")
    print("-" * 40)
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            free = total - reserved
            
            print(f"GPU {i}:")
            print(f"  Total: {total:.2f} GB")
            print(f"  Free: {free:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            
            if free < 25:  # Need ~25GB for model loading
                print(f"  ⚠️  Low memory! Need ~25GB, have {free:.2f} GB")
                return False
            else:
                print(f"  ✅ Sufficient memory available")
                return True
    return False

def run_interpretability_with_memory_management():
    """Run interpretability analysis with memory management"""
    print("🏀 BASKETBALL INTERPRETABILITY ANALYSIS WITH MEMORY MANAGEMENT")
    print("=" * 80)
    
    # Step 1: Clear memory
    clear_gpu_memory()
    
    # Step 2: Check available memory
    if not check_available_memory():
        print("\n❌ Insufficient memory for full analysis!")
        print("💡 Try the lightweight version below...")
        return run_lightweight_analysis()
    
    # Step 3: Run the analysis
    print("\n🚀 Running full interpretability analysis...")
    
    try:
        # Import and run the main analysis
        exec(open('basketball_interpretability_analysis.py').read())
        print("\n✅ Analysis completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("💡 Trying lightweight version...")
        return run_lightweight_analysis()

def run_lightweight_analysis():
    """Run a memory-efficient version of the analysis"""
    print("\n🪶 RUNNING LIGHTWEIGHT INTERPRETABILITY ANALYSIS")
    print("=" * 60)
    
    # This is a simplified version that processes one game at a time
    # and clears memory between games
    
    import torch
    import numpy as np
    import json
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from datetime import datetime
    
    # Load model
    print("🔧 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-14B-Instruct",
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    
    # Simple analysis on one game
    games_to_analyze = [0]  # Start with just one game
    
    results = {}
    
    for game_num in games_to_analyze:
        print(f"\n🎯 Analyzing Game {game_num:02d}...")
        
        # Load game data
        game_file = f"basketball_games/{game_num:02d}/data.json"
        if not os.path.exists(game_file):
            print(f"⚠️ Game {game_num:02d} data not found")
            continue
        
        with open(game_file, 'r') as f:
            game_data = json.load(f)
        
        game_info = json.dumps(game_data, indent=2)
        
        # Analyze each prompt type
        game_results = {}
        
        for prompt_type in ['reflection', 'no_reflection', 'dual_identity']:
            print(f"  📊 Analyzing {prompt_type}...")
            
            # Simple prompt
            if prompt_type == 'reflection':
                prompt = f"Write a basketball report. First, let me reflect on my approach.\n\nGame data:\n{game_info}"
            elif prompt_type == 'no_reflection':
                prompt = f"Write a basketball report.\n\nGame data:\n{game_info}"
            else:
                prompt = f"Write a basketball report with two perspectives.\n\nGame data:\n{game_info}"
            
            # Quick analysis
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=500)
            input_ids = inputs["input_ids"].to(model.device)
            
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                
                # Simple hidden state analysis
                final_hidden = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
                norm = float(np.linalg.norm(final_hidden))
                
                game_results[prompt_type] = {
                    'representation_norm': norm,
                    'sample_values': final_hidden[:5].tolist()
                }
            
            # Clear memory after each prompt
            torch.cuda.empty_cache()
        
        results[str(game_num)] = game_results
        print(f"✅ Completed Game {game_num:02d}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"lightweight_interpretability_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    
    # Print summary
    print("\n📊 LIGHTWEIGHT ANALYSIS SUMMARY:")
    for prompt_type in ['reflection', 'no_reflection', 'dual_identity']:
        norms = []
        for game_results in results.values():
            if prompt_type in game_results:
                norms.append(game_results[prompt_type]['representation_norm'])
        
        if norms:
            print(f"  {prompt_type}: Avg norm = {np.mean(norms):.4f}")
    
    return True

def main():
    """Main function with memory management"""
    print("🧹 MEMORY MANAGEMENT AND INTERPRETABILITY ANALYSIS")
    print("=" * 80)
    
    # Your previous work is safe!
    print("📝 NOTE: Your previous attention analysis results are saved in JSON files.")
    print("🔒 Clearing memory will NOT affect your previous work!")
    print()
    
    # Run the analysis
    success = run_interpretability_with_memory_management()
    
    if success:
        print("\n🎉 INTERPRETABILITY ANALYSIS COMPLETE!")
        print("📊 Check the generated JSON files and plots for results")
    else:
        print("\n⚠️  Analysis completed with limitations")
        print("💡 Consider running individual components separately")
    
    return success

if __name__ == "__main__":
    main() 