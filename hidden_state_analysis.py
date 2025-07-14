#!/usr/bin/env python3
"""
Hidden State Analysis for Reflection
Analyzes whether reflection changes the model's internal representations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import json
import seaborn as sns

class HiddenStateAnalyzer:
    def __init__(self, model_id="Qwen/Qwen2.5-14B-Instruct"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load model for hidden state analysis"""
        print(f"Loading {self.model_id} for hidden state analysis...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,  # KEY: Enable hidden state output
        )
        print("✅ Model loaded for hidden state analysis!")
    
    def get_hidden_states(self, prompt, max_new_tokens=100):
        """Generate text and capture hidden states"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True,
                output_hidden_states=True,  # Get hidden states
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Extract hidden states from the last layer
        hidden_states = []
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            for step_hidden in outputs.hidden_states:
                # Get last layer hidden states
                last_layer_hidden = step_hidden[-1][0]  # [seq_len, hidden_dim]
                hidden_states.append(last_layer_hidden)
        
        return {
            'input_ids': input_ids[0],
            'generated_tokens': outputs.sequences[0],
            'hidden_states': hidden_states,
            'input_length': input_ids.shape[1],
            'generated_text': self.tokenizer.decode(outputs.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
        }
    
    def analyze_hidden_state_patterns(self, hidden_data, label):
        """Analyze patterns in hidden states"""
        
        if not hidden_data['hidden_states']:
            print(f"No hidden states available for {label}")
            return None
        
        # Stack hidden states for analysis
        all_hidden = torch.stack(hidden_data['hidden_states'])  # [num_steps, seq_len, hidden_dim]
        
        # Focus on the last token's hidden state at each step
        last_token_hidden = all_hidden[:, -1, :]  # [num_steps, hidden_dim]
        
        # Convert to numpy for analysis
        hidden_np = last_token_hidden.cpu().numpy()
        
        # Analyze patterns
        analysis = {
            'label': label,
            'hidden_states_shape': hidden_np.shape,
            'mean_activation': np.mean(hidden_np, axis=0),
            'std_activation': np.std(hidden_np, axis=0),
            'activation_magnitude': np.linalg.norm(hidden_np, axis=1),
            'mean_magnitude': np.mean(np.linalg.norm(hidden_np, axis=1)),
            'hidden_states_raw': hidden_np  # For further analysis
        }
        
        return analysis
    
    def compare_hidden_representations(self, reflection_prompt, no_reflection_prompt, runs=3):
        """Compare hidden state representations between reflection and no-reflection"""
        
        print("🔍 Analyzing hidden state patterns...")
        
        reflection_analyses = []
        no_reflection_analyses = []
        
        for i in range(runs):
            print(f"Run {i+1}/{runs}")
            
            # With reflection
            refl_hidden = self.get_hidden_states(reflection_prompt)
            refl_analysis = self.analyze_hidden_state_patterns(refl_hidden, f"reflection_run_{i+1}")
            if refl_analysis:
                reflection_analyses.append(refl_analysis)
            
            # Without reflection
            no_refl_hidden = self.get_hidden_states(no_reflection_prompt)
            no_refl_analysis = self.analyze_hidden_state_patterns(no_refl_hidden, f"no_reflection_run_{i+1}")
            if no_refl_analysis:
                no_reflection_analyses.append(no_refl_analysis)
        
        # Compare representations
        comparison_results = self.compare_representations(reflection_analyses, no_reflection_analyses)
        
        return comparison_results
    
    def compare_representations(self, refl_analyses, no_refl_analyses):
        """Compare the internal representations"""
        
        if not refl_analyses or not no_refl_analyses:
            print("Insufficient data for comparison")
            return None
        
        # Calculate average representations
        refl_magnitudes = [np.mean(analysis['activation_magnitude']) for analysis in refl_analyses]
        no_refl_magnitudes = [np.mean(analysis['activation_magnitude']) for analysis in no_refl_analyses]
        
        # Compare activation patterns
        refl_mean_activations = [analysis['mean_activation'] for analysis in refl_analyses]
        no_refl_mean_activations = [analysis['mean_activation'] for analysis in no_refl_analyses]
        
        # Calculate cosine similarity between average activation patterns
        refl_avg_activation = np.mean(refl_mean_activations, axis=0)
        no_refl_avg_activation = np.mean(no_refl_mean_activations, axis=0)
        
        cosine_sim = cosine_similarity([refl_avg_activation], [no_refl_avg_activation])[0, 0]
        
        # Calculate activation magnitude differences
        magnitude_diff = np.mean(refl_magnitudes) - np.mean(no_refl_magnitudes)
        
        results = {
            'cosine_similarity': cosine_sim,
            'magnitude_difference': magnitude_diff,
            'reflection_avg_magnitude': np.mean(refl_magnitudes),
            'no_reflection_avg_magnitude': np.mean(no_refl_magnitudes),
            'reflection_activations': refl_avg_activation,
            'no_reflection_activations': no_refl_avg_activation,
            'similarity_threshold': 0.98  # High similarity suggests similar processing
        }
        
        return results
    
    def visualize_hidden_state_comparison(self, results):
        """Create visualizations of hidden state patterns"""
        
        if not results:
            print("No results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Activation magnitude comparison
        axes[0, 0].bar(['With Reflection', 'Without Reflection'], 
                      [results['reflection_avg_magnitude'], results['no_reflection_avg_magnitude']],
                      color=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Average Activation Magnitude')
        axes[0, 0].set_ylabel('Magnitude')
        
        # 2. Cosine similarity
        axes[0, 1].bar(['Cosine Similarity'], [results['cosine_similarity']], 
                      color='green' if results['cosine_similarity'] > results['similarity_threshold'] else 'red')
        axes[0, 1].set_title('Representation Similarity')
        axes[0, 1].set_ylabel('Cosine Similarity')
        axes[0, 1].axhline(y=results['similarity_threshold'], color='red', linestyle='--', alpha=0.5)
        
        # 3. PCA of activations
        combined_activations = np.stack([results['reflection_activations'], results['no_reflection_activations']])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined_activations)
        
        axes[1, 0].scatter(pca_result[0, 0], pca_result[0, 1], label='With Reflection', color='blue', s=100)
        axes[1, 0].scatter(pca_result[1, 0], pca_result[1, 1], label='Without Reflection', color='red', s=100)
        axes[1, 0].set_title('PCA of Average Activations')
        axes[1, 0].set_xlabel('PC1')
        axes[1, 0].set_ylabel('PC2')
        axes[1, 0].legend()
        
        # 4. Activation difference heatmap
        activation_diff = results['reflection_activations'] - results['no_reflection_activations']
        # Show first 100 dimensions for visualization
        diff_sample = activation_diff[:100].reshape(10, 10)
        
        sns.heatmap(diff_sample, ax=axes[1, 1], cmap='RdBu_r', center=0, cbar=True)
        axes[1, 1].set_title('Activation Differences (Sample)')
        
        plt.tight_layout()
        plt.savefig('hidden_state_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_hidden_state_analysis(self, results):
        """Print detailed hidden state analysis"""
        
        if not results:
            print("No results to analyze")
            return
        
        print("\n" + "="*80)
        print("🧠 HIDDEN STATE ANALYSIS RESULTS")
        print("="*80)
        
        print(f"Cosine Similarity: {results['cosine_similarity']:.4f}")
        print(f"Activation Magnitude Difference: {results['magnitude_difference']:+.4f}")
        print(f"Reflection Avg Magnitude: {results['reflection_avg_magnitude']:.4f}")
        print(f"No Reflection Avg Magnitude: {results['no_reflection_avg_magnitude']:.4f}")
        
        # Interpretation
        print(f"\n🎯 INTERPRETATION:")
        
        if results['cosine_similarity'] > results['similarity_threshold']:
            print("❌ Very similar internal representations - reflection may not change processing")
        elif results['cosine_similarity'] > 0.90:
            print("🤔 Mostly similar representations with some differences")
        else:
            print("✅ Significantly different internal representations - reflection changes processing")
        
        if abs(results['magnitude_difference']) > 0.1:
            direction = "higher" if results['magnitude_difference'] > 0 else "lower"
            print(f"✅ Reflection produces {direction} activation magnitudes")
        else:
            print("❌ No significant difference in activation magnitudes")

def main():
    print("=" * 80)
    print("🧠 HIDDEN STATE ANALYSIS FOR REFLECTION")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = HiddenStateAnalyzer()
    
    # Create test prompts
    reflection_prompt = """# Identity
You are a seasoned sports journalist.

# Task
Write a basketball report, but FIRST reflect on your approach:

## REFLECTION PROCESS:
- **Data Review**: What information do I have?
- **Quality Check**: Is my writing professional?
- **Fact Verification**: Are my statements accurate?

## FINAL REPORT:
Write the report here.

# Game Data
Hawks vs Timberwolves, Hawks won 101-96

Generate response:"""

    no_reflection_prompt = """# Identity
You are a seasoned sports journalist.

# Task
Write a basketball report based on the game data.

# Game Data
Hawks vs Timberwolves, Hawks won 101-96

Generate response:"""
    
    # Compare hidden state patterns
    results = analyzer.compare_hidden_representations(reflection_prompt, no_reflection_prompt, runs=3)
    
    # Print analysis
    analyzer.print_hidden_state_analysis(results)
    
    # Create visualizations
    analyzer.visualize_hidden_state_comparison(results)
    
    # Save results
    if results:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'cosine_similarity': results['cosine_similarity'],
            'magnitude_difference': results['magnitude_difference'],
            'reflection_avg_magnitude': results['reflection_avg_magnitude'],
            'no_reflection_avg_magnitude': results['no_reflection_avg_magnitude'],
            'similarity_threshold': results['similarity_threshold']
        }
        
        with open("hidden_state_analysis_results.json", "w") as f:
            json.dump(json_results, f, indent=2)
        
        print("💾 Hidden state analysis results saved!")

if __name__ == "__main__":
    main() 