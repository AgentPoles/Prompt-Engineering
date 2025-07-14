#!/usr/bin/env python3
"""
Token Probability Analysis for Reflection
Analyzes whether reflection changes the model's confidence in predictions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
from scipy import stats

class ProbabilityAnalyzer:
    def __init__(self, model_id="Qwen/Qwen2.5-14B-Instruct"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load model for probability analysis"""
        print(f"Loading {self.model_id} for probability analysis...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("✅ Model loaded for probability analysis!")
    
    def get_generation_probabilities(self, prompt, max_new_tokens=200):
        """Generate text and capture token probabilities"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True,
                output_scores=True,  # KEY: Get token probabilities
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Extract probabilities
        logits = outputs.scores  # List of logits for each generated token
        generated_tokens = outputs.sequences[0][input_ids.shape[1]:]  # Only generated part
        
        # Convert logits to probabilities
        probabilities = []
        entropies = []
        top_k_probs = []
        
        for i, logit in enumerate(logits):
            probs = torch.softmax(logit[0], dim=-1)  # Convert to probabilities
            
            # Probability of the chosen token
            chosen_token_prob = probs[generated_tokens[i]].item()
            probabilities.append(chosen_token_prob)
            
            # Entropy (measure of uncertainty)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            entropies.append(entropy)
            
            # Top-k probabilities (distribution shape)
            top_k_prob, _ = torch.topk(probs, k=10)
            top_k_probs.append(top_k_prob.cpu().numpy())
        
        return {
            'generated_tokens': generated_tokens,
            'probabilities': probabilities,
            'entropies': entropies,
            'top_k_probs': top_k_probs,
            'generated_text': self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        }
    
    def analyze_confidence_patterns(self, prob_data, label):
        """Analyze confidence patterns in generation"""
        
        probs = prob_data['probabilities']
        entropies = prob_data['entropies']
        
        analysis = {
            'label': label,
            'mean_probability': np.mean(probs),
            'std_probability': np.std(probs),
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'low_confidence_tokens': sum(1 for p in probs if p < 0.1),
            'high_confidence_tokens': sum(1 for p in probs if p > 0.8),
            'confidence_distribution': np.histogram(probs, bins=10, range=(0, 1))[0].tolist(),
            'entropy_distribution': np.histogram(entropies, bins=10)[0].tolist()
        }
        
        return analysis
    
    def compare_reflection_confidence(self, reflection_prompt, no_reflection_prompt, runs=3):
        """Compare confidence patterns between reflection and no-reflection"""
        
        print("🔍 Analyzing confidence patterns...")
        
        reflection_analyses = []
        no_reflection_analyses = []
        
        for i in range(runs):
            print(f"Run {i+1}/{runs}")
            
            # With reflection
            refl_probs = self.get_generation_probabilities(reflection_prompt)
            refl_analysis = self.analyze_confidence_patterns(refl_probs, f"reflection_run_{i+1}")
            reflection_analyses.append(refl_analysis)
            
            # Without reflection
            no_refl_probs = self.get_generation_probabilities(no_reflection_prompt)
            no_refl_analysis = self.analyze_confidence_patterns(no_refl_probs, f"no_reflection_run_{i+1}")
            no_reflection_analyses.append(no_refl_analysis)
        
        # Aggregate results
        aggregated_results = self.aggregate_confidence_results(reflection_analyses, no_reflection_analyses)
        
        return aggregated_results
    
    def aggregate_confidence_results(self, refl_analyses, no_refl_analyses):
        """Aggregate results across multiple runs"""
        
        metrics = ['mean_probability', 'mean_entropy', 'low_confidence_tokens', 'high_confidence_tokens']
        
        results = {}
        
        for metric in metrics:
            refl_values = [analysis[metric] for analysis in refl_analyses]
            no_refl_values = [analysis[metric] for analysis in no_refl_analyses]
            
            # Statistical comparison
            t_stat, p_value = stats.ttest_ind(refl_values, no_refl_values)
            
            results[metric] = {
                'reflection_mean': np.mean(refl_values),
                'reflection_std': np.std(refl_values),
                'no_reflection_mean': np.mean(no_refl_values),
                'no_reflection_std': np.std(no_refl_values),
                'difference': np.mean(refl_values) - np.mean(no_refl_values),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return results
    
    def visualize_confidence_comparison(self, results):
        """Create visualizations of confidence patterns"""
        
        metrics = ['mean_probability', 'mean_entropy', 'low_confidence_tokens', 'high_confidence_tokens']
        titles = ['Average Token Probability', 'Average Entropy', 'Low Confidence Tokens', 'High Confidence Tokens']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            data = results[metric]
            
            refl_mean = data['reflection_mean']
            refl_std = data['reflection_std']
            no_refl_mean = data['no_reflection_mean']
            no_refl_std = data['no_reflection_std']
            
            # Bar plot with error bars
            x = ['With Reflection', 'Without Reflection']
            y = [refl_mean, no_refl_mean]
            yerr = [refl_std, no_refl_std]
            
            bars = axes[i].bar(x, y, yerr=yerr, capsize=10, 
                             color=['lightblue', 'lightcoral'], alpha=0.7)
            
            axes[i].set_title(title)
            axes[i].set_ylabel('Value')
            
            # Add significance indicator
            if data['significant']:
                axes[i].text(0.5, max(y) * 1.1, f"p < 0.05", ha='center', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('confidence_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_confidence_analysis(self, results):
        """Print detailed confidence analysis"""
        
        print("\n" + "="*80)
        print("📊 CONFIDENCE ANALYSIS RESULTS")
        print("="*80)
        
        for metric, data in results.items():
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  With Reflection: {data['reflection_mean']:.4f} ± {data['reflection_std']:.4f}")
            print(f"  Without Reflection: {data['no_reflection_mean']:.4f} ± {data['no_reflection_std']:.4f}")
            print(f"  Difference: {data['difference']:+.4f}")
            print(f"  Statistical significance: {'✅ YES' if data['significant'] else '❌ NO'} (p={data['p_value']:.4f})")
        
        # Interpretation
        print(f"\n🎯 INTERPRETATION:")
        
        prob_diff = results['mean_probability']['difference']
        entropy_diff = results['mean_entropy']['difference']
        
        if prob_diff < -0.05 and entropy_diff > 0.1:
            print("✅ Reflection increases thoughtfulness - lower confidence, higher entropy")
        elif prob_diff > 0.05 and entropy_diff < -0.1:
            print("🤔 Reflection increases confidence - higher probability, lower entropy")
        else:
            print("❌ No clear confidence pattern difference detected")

def main():
    print("=" * 80)
    print("🎯 TOKEN PROBABILITY ANALYSIS FOR REFLECTION")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = ProbabilityAnalyzer()
    
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
    
    # Compare confidence patterns
    results = analyzer.compare_reflection_confidence(reflection_prompt, no_reflection_prompt, runs=3)
    
    # Print analysis
    analyzer.print_confidence_analysis(results)
    
    # Create visualizations
    analyzer.visualize_confidence_comparison(results)
    
    # Save results
    with open("probability_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("💾 Probability analysis results saved!")

if __name__ == "__main__":
    main() 