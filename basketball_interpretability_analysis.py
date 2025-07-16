#!/usr/bin/env python3
"""
🏀 Basketball Interpretability Analysis
======================================
Comprehensive interpretability analysis for basketball reflection prompts
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
from scipy import stats
from datetime import datetime
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ================================================================================
# 🎯 BLOCK 1: SETUP AND CONFIGURATION
# ================================================================================

print("🏀 BASKETBALL INTERPRETABILITY ANALYSIS")
print("=" * 80)

# Configuration
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
SELECTED_GAMES = [0, 5, 10, 15, 18]  # Same games as attention analysis
MAX_GENERATION_TOKENS = 50  # Keep it manageable
BASKETBALL_GAMES_DIR = "basketball_games"

# Load prompts
def load_prompts():
    """Load the three prompt types"""
    prompts = {}
    
    try:
        with open('prompt_basicreflection.txt', 'r') as f:
            prompts['reflection'] = f.read().strip()
        print("✅ Loaded reflection prompt")
    except FileNotFoundError:
        prompts['reflection'] = """Write a comprehensive basketball report analyzing the game data.

## REFLECTION PROCESS:
Before writing, let me reflect on my approach:
- **Data Review**: What key information do I have?
- **Quality Check**: Is my analysis professional and accurate?
- **Fact Verification**: Are my statements grounded in the data?

## BASKETBALL REPORT:
Now I'll write the report based on my reflection:"""
        print("⚠️ Using default reflection prompt")
    
    try:
        with open('prompt_noreflection.txt', 'r') as f:
            prompts['no_reflection'] = f.read().strip()
        print("✅ Loaded no-reflection prompt")
    except FileNotFoundError:
        prompts['no_reflection'] = """Write a comprehensive basketball report analyzing the game data."""
        print("⚠️ Using default no-reflection prompt")
    
    try:
        with open('prompt_dualidentity.txt', 'r') as f:
            prompts['dual_identity'] = f.read().strip()
        print("✅ Loaded dual identity prompt")
    except FileNotFoundError:
        prompts['dual_identity'] = """You have two internal voices analyzing this basketball game:

**JOURNALIST**: Provides objective, professional analysis
**FAN**: Adds emotional engagement and excitement

Both voices will collaborate to write a comprehensive report."""
        print("⚠️ Using default dual identity prompt")
    
    return prompts

# Load model and tokenizer
def load_model():
    """Load the model and tokenizer for interpretability analysis"""
    print(f"🔧 Loading {MODEL_ID} for interpretability analysis...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"  # Required for interpretability
    )
    
    print("✅ Model loaded successfully!")
    return model, tokenizer

# ================================================================================
# 🎯 BLOCK 2: BASKETBALL-SPECIFIC INTERPRETABILITY ANALYZER
# ================================================================================

class BasketballInterpretabilityAnalyzer:
    """Interpretability analyzer specifically for basketball reflection study"""
    
    def __init__(self, model, tokenizer, prompts):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.results = {}
        
    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def extract_game_info(self, game_data):
        """Extract game information from JSON data"""
        # Simple extraction - adapt based on your data structure
        if isinstance(game_data, dict):
            return json.dumps(game_data, indent=2)
        return str(game_data)
    
    def create_full_prompt(self, prompt_template, game_info):
        """Create complete prompt with game data"""
        full_prompt = f"{prompt_template}\n\nGame data:\n{game_info}"
        
        # Apply chat template if available
        try:
            messages = [{"role": "user", "content": full_prompt}]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            return f"<|user|>\n{full_prompt}\n<|assistant|>\n"
    
    def analyze_single_game(self, game_num):
        """Run comprehensive analysis on a single game"""
        print(f"\n🎯 Analyzing Game {game_num:02d}")
        print("-" * 50)
        
        # Load game data
        game_file = os.path.join(BASKETBALL_GAMES_DIR, f"{game_num:02d}", "data.json")
        if not os.path.exists(game_file):
            print(f"⚠️ Game {game_num:02d} data not found, skipping...")
            return None
        
        with open(game_file, 'r') as f:
            game_data = json.load(f)
        
        game_info = self.extract_game_info(game_data)
        game_results = {}
        
        # Analyze each prompt type
        for prompt_type in ['reflection', 'no_reflection', 'dual_identity']:
            print(f"  📊 Analyzing {prompt_type}...")
            
            # Create full prompt
            full_prompt = self.create_full_prompt(self.prompts[prompt_type], game_info)
            
            # Run all analyses
            prompt_results = {}
            
            # 1. Hidden State Analysis
            prompt_results['hidden_states'] = self.analyze_hidden_states(full_prompt, prompt_type)
            
            # 2. Confidence Analysis
            prompt_results['confidence'] = self.analyze_confidence(full_prompt, prompt_type)
            
            # 3. Concept Activation Analysis
            prompt_results['concepts'] = self.analyze_concept_activation(full_prompt, prompt_type)
            
            # 4. Information Flow Analysis
            prompt_results['information_flow'] = self.analyze_information_flow(full_prompt, prompt_type)
            
            game_results[prompt_type] = prompt_results
            
            # Clear memory after each prompt
            self.clear_memory()
        
        return game_results
    
    def analyze_hidden_states(self, prompt, prompt_type):
        """Analyze hidden state representations"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            
            # Get hidden states from all layers
            layer_states = []
            for layer_hidden in outputs.hidden_states:
                # Use the last token's representation and convert to float32
                last_token_state = layer_hidden[0, -1, :].float().cpu().numpy()
                layer_states.append(last_token_state)
            
            # Calculate metrics
            final_state = layer_states[-1]
            representation_norm = np.linalg.norm(final_state)
            
            # Calculate effective dimensionality
            state_matrix = np.array(layer_states)
            if state_matrix.shape[0] > 1:
                U, s, Vt = np.linalg.svd(state_matrix)
                effective_rank = np.sum(s > 0.01 * s[0])
            else:
                effective_rank = 1
            
            return {
                'representation_norm': float(representation_norm),
                'effective_rank': int(effective_rank),
                'final_state_sample': final_state[:10].tolist(),  # First 10 dims for inspection
                'num_layers': len(layer_states)
            }
    
    def analyze_confidence(self, prompt, prompt_type):
        """Analyze model confidence during generation"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        confidence_scores = []
        entropy_scores = []
        
        current_ids = input_ids.clone()
        
        for step in range(MAX_GENERATION_TOKENS):
            with torch.no_grad():
                outputs = self.model(current_ids)
                logits = outputs.logits[0, -1, :]
                
                # Calculate confidence metrics
                probs = torch.softmax(logits, dim=-1)
                
                # Max probability (confidence)
                max_prob = torch.max(probs).item()
                confidence_scores.append(max_prob)
                
                # Entropy (uncertainty)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
                entropy_scores.append(entropy)
                
                # Get next token
                next_token = torch.multinomial(probs, 1)
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return {
            'avg_confidence': float(np.mean(confidence_scores)),
            'avg_entropy': float(np.mean(entropy_scores)),
            'confidence_std': float(np.std(confidence_scores)),
            'entropy_std': float(np.std(entropy_scores)),
            'num_tokens': len(confidence_scores)
        }
    
    def analyze_concept_activation(self, prompt, prompt_type):
        """Analyze semantic concept activation"""
        # Define concept examples
        concept_examples = {
            'reflection': [
                'Let me think about this carefully',
                'I need to reflect on my approach',
                'Let me review what I said',
                'I should check my reasoning'
            ],
            'analysis': [
                'Analyzing the data shows',
                'Breaking down the problem',
                'Examining the evidence',
                'Systematic investigation reveals'
            ],
            'sports_reporting': [
                'The game was intense',
                'Players performed well',
                'The final score was',
                'Basketball analysis shows'
            ],
            'certainty': [
                'I am confident that',
                'This is definitely true',
                'Without doubt',
                'Clearly the answer is'
            ]
        }
        
                 # Create concept vectors
        concept_vectors = {}
        for concept, examples in concept_examples.items():
            concept_activations = []
            for example in examples:
                inputs = self.tokenizer(example, return_tensors="pt", truncation=True, max_length=100)
                input_ids = inputs["input_ids"].to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=True)
                    activation = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
                    concept_activations.append(activation)
            
            concept_vectors[concept] = np.mean(concept_activations, axis=0)
        
        # Analyze prompt activation
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            prompt_activation = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
        
        # Calculate similarities
        concept_similarities = {}
        for concept, concept_vector in concept_vectors.items():
            similarity = cosine_similarity([prompt_activation], [concept_vector])[0][0]
            concept_similarities[concept] = float(similarity)
        
        return concept_similarities
    
    def analyze_information_flow(self, prompt, prompt_type):
        """Analyze information flow through network layers"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            
            # Get hidden states from all layers
            layer_states = []
            for layer_hidden in outputs.hidden_states:
                last_token_state = layer_hidden[0, -1, :].float().cpu().numpy()
                layer_states.append(last_token_state)
            
            # Calculate layer-to-layer changes
            layer_changes = []
            for i in range(1, len(layer_states)):
                similarity = cosine_similarity([layer_states[i-1]], [layer_states[i]])[0][0]
                layer_changes.append(1 - similarity)  # Convert to "change" measure
            
            # Calculate cumulative norms
            cumulative_norms = [float(np.linalg.norm(state)) for state in layer_states]
            
            return {
                'avg_layer_change': float(np.mean(layer_changes)),
                'max_layer_change': float(np.max(layer_changes)),
                'layer_changes': layer_changes,
                'cumulative_norms': cumulative_norms,
                'final_norm': cumulative_norms[-1]
            }

# ================================================================================
# 🎯 BLOCK 3: ANALYSIS EXECUTION AND VISUALIZATION
# ================================================================================

def run_interpretability_analysis():
    """Run the complete interpretability analysis"""
    print("🚀 STARTING BASKETBALL INTERPRETABILITY ANALYSIS")
    print("=" * 80)
    
    # Load components
    model, tokenizer = load_model()
    prompts = load_prompts()
    
    # Initialize analyzer
    analyzer = BasketballInterpretabilityAnalyzer(model, tokenizer, prompts)
    
    # Run analysis on selected games
    all_results = {}
    
    for game_num in SELECTED_GAMES:
        print(f"\n📊 Processing Game {game_num:02d}...")
        game_results = analyzer.analyze_single_game(game_num)
        
        if game_results:
            all_results[str(game_num)] = game_results
            print(f"✅ Completed Game {game_num:02d}")
        else:
            print(f"❌ Failed to process Game {game_num:02d}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"basketball_interpretability_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    
    # Create summary analysis
    create_summary_analysis(all_results)
    
    return all_results

def create_summary_analysis(results):
    """Create summary analysis and visualizations"""
    print("\n📊 CREATING SUMMARY ANALYSIS")
    print("-" * 50)
    
    # Aggregate results across games
    prompt_types = ['reflection', 'no_reflection', 'dual_identity']
    
    # 1. Hidden State Analysis Summary
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
    
    # 2. Confidence Analysis Summary
    print("\n📊 CONFIDENCE ANALYSIS:")
    for prompt_type in prompt_types:
        confidences = []
        entropies = []
        
        for game_results in results.values():
            if prompt_type in game_results:
                conf = game_results[prompt_type]['confidence']
                confidences.append(conf['avg_confidence'])
                entropies.append(conf['avg_entropy'])
        
        if confidences:
            print(f"  {prompt_type}:")
            print(f"    Avg confidence: {np.mean(confidences):.4f}")
            print(f"    Avg entropy: {np.mean(entropies):.4f}")
    
    # 3. Concept Activation Summary
    print("\n🎯 CONCEPT ACTIVATION ANALYSIS:")
    concepts = ['reflection', 'analysis', 'sports_reporting', 'certainty']
    
    for concept in concepts:
        print(f"  {concept}:")
        for prompt_type in prompt_types:
            activations = []
            
            for game_results in results.values():
                if prompt_type in game_results:
                    concepts_data = game_results[prompt_type]['concepts']
                    if concept in concepts_data:
                        activations.append(concepts_data[concept])
            
            if activations:
                print(f"    {prompt_type}: {np.mean(activations):.4f}")
    
    # 4. Information Flow Summary
    print("\n🌊 INFORMATION FLOW ANALYSIS:")
    for prompt_type in prompt_types:
        flow_changes = []
        
        for game_results in results.values():
            if prompt_type in game_results:
                flow = game_results[prompt_type]['information_flow']
                flow_changes.append(flow['avg_layer_change'])
        
        if flow_changes:
            print(f"  {prompt_type}: Avg layer change = {np.mean(flow_changes):.4f}")
    
    # Create visualization
    create_interpretability_plots(results)

def create_interpretability_plots(results):
    """Create visualization plots for interpretability analysis"""
    print("\n🎨 Creating interpretability visualizations...")
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🏀 Basketball Interpretability Analysis Results', fontsize=16, fontweight='bold')
    
    prompt_types = ['reflection', 'no_reflection', 'dual_identity']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    # 1. Hidden State Norms
    ax1 = axes[0, 0]
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
    
    ax1.set_title('Hidden State Representation Norms')
    ax1.set_ylabel('Norm Value')
    ax1.set_xticks(range(len(prompt_types)))
    ax1.set_xticklabels([pt.replace('_', ' ').title() for pt in prompt_types])
    
    # 2. Confidence Levels
    ax2 = axes[0, 1]
    conf_data = {pt: [] for pt in prompt_types}
    
    for game_results in results.values():
        for prompt_type in prompt_types:
            if prompt_type in game_results:
                conf = game_results[prompt_type]['confidence']['avg_confidence']
                conf_data[prompt_type].append(conf)
    
    for i, (prompt_type, confs) in enumerate(conf_data.items()):
        if confs:
            ax2.bar(i, np.mean(confs), color=colors[i], alpha=0.7, 
                   yerr=np.std(confs), capsize=5)
    
    ax2.set_title('Average Confidence Levels')
    ax2.set_ylabel('Confidence')
    ax2.set_xticks(range(len(prompt_types)))
    ax2.set_xticklabels([pt.replace('_', ' ').title() for pt in prompt_types])
    
    # 3. Concept Activation Heatmap
    ax3 = axes[1, 0]
    concepts = ['reflection', 'analysis', 'sports_reporting', 'certainty']
    
    # Create concept activation matrix
    concept_matrix = np.zeros((len(concepts), len(prompt_types)))
    
    for i, concept in enumerate(concepts):
        for j, prompt_type in enumerate(prompt_types):
            activations = []
            for game_results in results.values():
                if prompt_type in game_results:
                    concepts_data = game_results[prompt_type]['concepts']
                    if concept in concepts_data:
                        activations.append(concepts_data[concept])
            
            if activations:
                concept_matrix[i, j] = np.mean(activations)
    
    sns.heatmap(concept_matrix, 
                xticklabels=[pt.replace('_', ' ').title() for pt in prompt_types],
                yticklabels=concepts,
                annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax3)
    ax3.set_title('Concept Activation Patterns')
    
    # 4. Information Flow
    ax4 = axes[1, 1]
    flow_data = {pt: [] for pt in prompt_types}
    
    for game_results in results.values():
        for prompt_type in prompt_types:
            if prompt_type in game_results:
                flow = game_results[prompt_type]['information_flow']['avg_layer_change']
                flow_data[prompt_type].append(flow)
    
    for i, (prompt_type, flows) in enumerate(flow_data.items()):
        if flows:
            ax4.bar(i, np.mean(flows), color=colors[i], alpha=0.7, 
                   yerr=np.std(flows), capsize=5)
    
    ax4.set_title('Average Information Flow Changes')
    ax4.set_ylabel('Layer Change')
    ax4.set_xticks(range(len(prompt_types)))
    ax4.set_xticklabels([pt.replace('_', ' ').title() for pt in prompt_types])
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"basketball_interpretability_plots_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Plots saved to {plot_filename}")

# ================================================================================
# 🎯 BLOCK 4: MAIN EXECUTION
# ================================================================================

def main():
    """Main execution function"""
    print("🏀 BASKETBALL INTERPRETABILITY ANALYSIS")
    print("=" * 80)
    print("This analysis will reveal what's happening in the model's mind")
    print("when processing different types of basketball report prompts.")
    print()
    
    # Run the analysis
    results = run_interpretability_analysis()
    
    print("\n✅ INTERPRETABILITY ANALYSIS COMPLETE!")
    print("=" * 80)
    print("📊 Generated comprehensive analysis of model internal processing")
    print("🎯 Results show how reflection prompts actually affect the model")
    print("🧠 This provides mechanistic insights beyond attention patterns")
    
    return results

if __name__ == "__main__":
    main() 