#!/usr/bin/env python3
"""
🏀 Run Full Interpretability Analysis in Colab
==============================================
Run the comprehensive analysis using code already loaded in your notebook
"""

import torch
import numpy as np
import json
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def convert_to_serializable(obj):
    """Convert numpy/torch types to JSON serializable types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def run_full_analysis_colab():
    """Run the full interpretability analysis in Colab environment"""
    print("🚀 RUNNING FULL INTERPRETABILITY ANALYSIS IN COLAB")
    print("=" * 80)
    
    # Since you already have the code loaded, let's run it directly
    try:
        # Run the main analysis function from your loaded code
        results = main()  # This should be available from your loaded basketball_interpretability_analysis.py
        
        print("\n✅ FULL ANALYSIS COMPLETED!")
        return results
        
    except NameError:
        print("⚠️ Main function not found. Running manual analysis...")
        return run_manual_analysis()

def run_manual_analysis():
    """Run the analysis manually if the main function isn't available"""
    print("\n🔧 RUNNING MANUAL INTERPRETABILITY ANALYSIS")
    print("=" * 60)
    
    # Load the analysis components
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load model and tokenizer
    print("🔧 Loading model and tokenizer...")
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
    
    # Games to analyze
    games_to_analyze = [0, 5, 10, 15, 18]  # Same as attention analysis
    
    print(f"🎯 Analyzing {len(games_to_analyze)} games...")
    
    all_results = {}
    
    for game_num in games_to_analyze:
        print(f"\n📊 Processing Game {game_num:02d}...")
        
        # Load game data
        game_file = f"basketball_games/{game_num:02d}/data.json"
        try:
            with open(game_file, 'r') as f:
                game_data = json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Game {game_num:02d} data not found, skipping...")
            continue
        
        game_info = json.dumps(game_data, indent=2)
        game_results = {}
        
        # Analyze each prompt type
        for prompt_type in ['reflection', 'no_reflection', 'dual_identity']:
            print(f"  🔍 Analyzing {prompt_type}...")
            
            # Create prompts
            if prompt_type == 'reflection':
                prompt = f"""Write a comprehensive basketball report analyzing the game data.

## REFLECTION PROCESS:
Before writing, let me reflect on my approach:
- **Data Review**: What key information do I have?
- **Quality Check**: Is my analysis professional and accurate?
- **Fact Verification**: Are my statements grounded in the data?

## BASKETBALL REPORT:
Now I'll write the report based on my reflection:

Game data:
{game_info}"""
            elif prompt_type == 'no_reflection':
                prompt = f"Write a comprehensive basketball report analyzing the game data.\n\nGame data:\n{game_info}"
            else:  # dual_identity
                prompt = f"""You have two internal voices analyzing this basketball game:

**JOURNALIST**: Provides objective, professional analysis
**FAN**: Adds emotional engagement and excitement

Both voices will collaborate to write a comprehensive report.

Game data:
{game_info}"""
            
            # Apply chat template
            try:
                messages = [{"role": "user", "content": prompt}]
                full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
            
            # Analyze the prompt
            prompt_results = analyze_prompt_comprehensive(model, tokenizer, full_prompt, prompt_type)
            game_results[prompt_type] = prompt_results
            
            # Clear memory
            torch.cuda.empty_cache()
        
        all_results[str(game_num)] = game_results
        print(f"✅ Completed Game {game_num:02d}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"full_interpretability_results_{timestamp}.json"
    
    # Convert all results to JSON serializable format
    serializable_results = convert_to_serializable(all_results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    
    # Create summary and visualizations
    create_comprehensive_summary(all_results)
    
    return all_results

def analyze_prompt_comprehensive(model, tokenizer, prompt, prompt_type):
    """Comprehensive analysis of a single prompt"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
    input_ids = inputs["input_ids"].to(model.device)
    
    results = {}
    
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
        # 1. Hidden State Analysis
        layer_states = []
        for layer_hidden in outputs.hidden_states:
            last_token_state = layer_hidden[0, -1, :].float().cpu().numpy()
            layer_states.append(last_token_state)
        
        final_state = layer_states[-1]
        representation_norm = float(np.linalg.norm(final_state))
        
        # Calculate effective dimensionality
        state_matrix = np.array(layer_states)
        if state_matrix.shape[0] > 1:
            U, s, Vt = np.linalg.svd(state_matrix)
            effective_rank = int(np.sum(s > 0.01 * s[0]))
        else:
            effective_rank = 1
        
        results['hidden_states'] = {
            'representation_norm': representation_norm,
            'effective_rank': effective_rank,
            'num_layers': len(layer_states),
            'final_state_sample': [float(x) for x in final_state[:10]]  # Convert to Python floats
        }
        
        # 2. Information Flow Analysis
        layer_changes = []
        for i in range(1, len(layer_states)):
            similarity = cosine_similarity([layer_states[i-1]], [layer_states[i]])[0][0]
            layer_changes.append(float(1 - similarity))  # Convert to Python float
        
        results['information_flow'] = {
            'avg_layer_change': float(np.mean(layer_changes)),
            'max_layer_change': float(np.max(layer_changes)),
            'layer_changes': layer_changes
        }
        
        # 3. Confidence Analysis (simplified)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        max_prob = torch.max(probs).item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        
        results['confidence'] = {
            'max_confidence': float(max_prob),
            'entropy': float(entropy)
        }
    
    return results

def create_comprehensive_summary(results):
    """Create comprehensive summary and visualizations"""
    print("\n📊 COMPREHENSIVE INTERPRETABILITY SUMMARY")
    print("=" * 80)
    
    prompt_types = ['reflection', 'no_reflection', 'dual_identity']
    
    # 1. Hidden State Analysis
    print("\n🧠 HIDDEN STATE ANALYSIS:")
    for prompt_type in prompt_types:
        norms = []
        ranks = []
        
        for game_results in results.values():
            if prompt_type in game_results:
                hs = game_results[prompt_type]['hidden_states']
                norms.append(hs['representation_norm'])
                ranks.append(hs['effective_rank'])
        
        if norms:
            print(f"  {prompt_type}:")
            print(f"    Avg representation norm: {np.mean(norms):.4f}")
            print(f"    Avg effective rank: {np.mean(ranks):.2f}")
    
    # 2. Information Flow Analysis
    print("\n🌊 INFORMATION FLOW ANALYSIS:")
    for prompt_type in prompt_types:
        flow_changes = []
        
        for game_results in results.values():
            if prompt_type in game_results:
                flow = game_results[prompt_type]['information_flow']
                flow_changes.append(flow['avg_layer_change'])
        
        if flow_changes:
            print(f"  {prompt_type}: Avg layer change = {np.mean(flow_changes):.4f}")
    
    # 3. Confidence Analysis
    print("\n📊 CONFIDENCE ANALYSIS:")
    for prompt_type in prompt_types:
        confidences = []
        entropies = []
        
        for game_results in results.values():
            if prompt_type in game_results:
                conf = game_results[prompt_type]['confidence']
                confidences.append(conf['max_confidence'])
                entropies.append(conf['entropy'])
        
        if confidences:
            print(f"  {prompt_type}:")
            print(f"    Avg confidence: {np.mean(confidences):.4f}")
            print(f"    Avg entropy: {np.mean(entropies):.4f}")
    
    # Create visualization
    create_summary_plot(results)

def create_summary_plot(results):
    """Create summary visualization"""
    print("\n🎨 Creating summary visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('🏀 Basketball Interpretability Analysis Results', fontsize=16, fontweight='bold')
    
    prompt_types = ['reflection', 'no_reflection', 'dual_identity']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    # 1. Hidden State Norms
    ax1 = axes[0]
    norms_data = {pt: [] for pt in prompt_types}
    
    for game_results in results.values():
        for prompt_type in prompt_types:
            if prompt_type in game_results:
                norm = game_results[prompt_type]['hidden_states']['representation_norm']
                norms_data[prompt_type].append(norm)
    
    for i, (prompt_type, norms) in enumerate(norms_data.items()):
        if norms:
            ax1.bar(i, np.mean(norms), color=colors[i], alpha=0.7, 
                   yerr=np.std(norms), capsize=5)
            ax1.text(i, np.mean(norms) + np.std(norms) + 5, 
                    f'{np.mean(norms):.1f}', ha='center', fontweight='bold')
    
    ax1.set_title('Hidden State Representation Norms')
    ax1.set_ylabel('Norm Value')
    ax1.set_xticks(range(len(prompt_types)))
    ax1.set_xticklabels([pt.replace('_', ' ').title() for pt in prompt_types])
    
    # 2. Information Flow
    ax2 = axes[1]
    flow_data = {pt: [] for pt in prompt_types}
    
    for game_results in results.values():
        for prompt_type in prompt_types:
            if prompt_type in game_results:
                flow = game_results[prompt_type]['information_flow']['avg_layer_change']
                flow_data[prompt_type].append(flow)
    
    for i, (prompt_type, flows) in enumerate(flow_data.items()):
        if flows:
            ax2.bar(i, np.mean(flows), color=colors[i], alpha=0.7, 
                   yerr=np.std(flows), capsize=5)
            ax2.text(i, np.mean(flows) + np.std(flows) + 0.001, 
                    f'{np.mean(flows):.3f}', ha='center', fontweight='bold')
    
    ax2.set_title('Average Information Flow Changes')
    ax2.set_ylabel('Layer Change')
    ax2.set_xticks(range(len(prompt_types)))
    ax2.set_xticklabels([pt.replace('_', ' ').title() for pt in prompt_types])
    
    # 3. Confidence Levels
    ax3 = axes[2]
    conf_data = {pt: [] for pt in prompt_types}
    
    for game_results in results.values():
        for prompt_type in prompt_types:
            if prompt_type in game_results:
                conf = game_results[prompt_type]['confidence']['max_confidence']
                conf_data[prompt_type].append(conf)
    
    for i, (prompt_type, confs) in enumerate(conf_data.items()):
        if confs:
            ax3.bar(i, np.mean(confs), color=colors[i], alpha=0.7, 
                   yerr=np.std(confs), capsize=5)
            ax3.text(i, np.mean(confs) + np.std(confs) + 0.01, 
                    f'{np.mean(confs):.3f}', ha='center', fontweight='bold')
    
    ax3.set_title('Average Confidence Levels')
    ax3.set_ylabel('Confidence')
    ax3.set_xticks(range(len(prompt_types)))
    ax3.set_xticklabels([pt.replace('_', ' ').title() for pt in prompt_types])
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"full_interpretability_plots_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Plot saved to {plot_filename}")

def main():
    """Main function for Colab"""
    print("🏀 FULL INTERPRETABILITY ANALYSIS IN COLAB")
    print("=" * 80)
    
    results = run_manual_analysis()
    
    print("\n🎉 COMPLETE INTERPRETABILITY ANALYSIS FINISHED!")
    print("=" * 80)
    print("📊 Generated comprehensive analysis of model internal processing")
    print("🧠 This reveals the mechanistic differences between prompt types")
    print("🎯 Combined with your attention analysis, this provides full picture")
    
    return results

if __name__ == "__main__":
    main() 