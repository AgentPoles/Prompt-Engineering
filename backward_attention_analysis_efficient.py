#!/usr/bin/env python3
"""
Memory-Efficient Backward Attention Analysis for Reflection Behavior
Optimized version that tracks backward attention without storing all patterns
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import gc

class EfficientBackwardAttentionAnalyzer:
    def __init__(self, model_id="Qwen/Qwen2.5-14B-Instruct"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.load_model()
        
        # Keywords that indicate reflection checkpoints
        self.reflection_keywords = [
            'reflect', 'review', 'check', 'assess', 'evaluate', 
            'analyze', 'verify', 'reconsider', 'double-check', 'confirm'
        ]
    
    def load_model(self):
        """Load model with attention output enabled"""
        print(f"Loading {self.model_id} for backward attention analysis...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # More aggressive quantization for memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,  # Use 4-bit instead of 8-bit
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ),
            device_map="auto",
            torch_dtype=torch.float16,  # Use float16 instead of bfloat16
            attn_implementation="eager",
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded for backward attention tracking!")
    
    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def analyze_single_generation(self, prompt, max_new_tokens=50):
        """Generate text and analyze backward attention patterns efficiently"""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1000,  # Reduced max length
            padding=False
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        input_length = input_ids.shape[1]
        
        # Track metrics without storing full attention patterns
        metrics = {
            'steps': [],
            'backward_ratios': [],
            'attention_to_generated': [],
            'reflection_points': [],
            'input_length': input_length
        }
        
        # Generate with efficient attention tracking
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                output_attentions=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True  # Use cache for efficiency
            )
        
        # Extract generated tokens
        generated_tokens = outputs.sequences[0][input_length:]
        
        # Process attention patterns efficiently
        if hasattr(outputs, 'attentions') and outputs.attentions:
            for step, step_attention in enumerate(outputs.attentions):
                if step_attention is None or len(step_attention) == 0:
                    continue
                
                try:
                    # Calculate backward attention for this step
                    backward_metrics = self.calculate_backward_attention_efficient(
                        step_attention, input_length, step + 1
                    )
                    
                    # Check if this step involves reflection
                    if step < len(generated_tokens):
                        token_text = self.tokenizer.decode([generated_tokens[step]])
                        is_reflection_step = any(
                            keyword in token_text.lower() 
                            for keyword in self.reflection_keywords
                        )
                        
                        if is_reflection_step:
                            metrics['reflection_points'].append({
                                'step': step,
                                'token': token_text,
                                'backward_ratio': backward_metrics['backward_ratio']
                            })
                    
                    # Store metrics
                    metrics['steps'].append(step)
                    metrics['backward_ratios'].append(backward_metrics['backward_ratio'])
                    metrics['attention_to_generated'].append(backward_metrics['attention_to_generated'])
                    
                    # Clear memory after processing each step
                    del step_attention
                    self.clear_memory()
                    
                except Exception as e:
                    print(f"⚠️ Error processing step {step}: {e}")
                    continue
        
        return metrics
    
    def calculate_backward_attention_efficient(self, step_attention, input_length, generated_length):
        """Calculate backward attention metrics efficiently"""
        
        try:
            # Get last layer attention (most meaningful)
            last_layer = step_attention[-1][0]  # [num_heads, seq_len, seq_len]
            
            # Average across heads and convert to float32
            avg_attention = torch.mean(last_layer, dim=0).float()
            
            # Get attention from current position to previous tokens
            current_pos = avg_attention.shape[0] - 1
            current_attention = avg_attention[current_pos, :]
            
            # Calculate attention to different regions
            attention_to_input = current_attention[:input_length].sum().item()
            
            if generated_length > 0:
                gen_start = input_length
                gen_end = min(input_length + generated_length, current_attention.shape[0])
                attention_to_generated = current_attention[gen_start:gen_end].sum().item()
            else:
                attention_to_generated = 0.0
            
            # Calculate backward attention ratio
            total_attention = attention_to_input + attention_to_generated
            backward_ratio = attention_to_generated / total_attention if total_attention > 0 else 0.0
            
            return {
                'backward_ratio': backward_ratio,
                'attention_to_generated': attention_to_generated,
                'attention_to_input': attention_to_input
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating backward attention: {e}")
            return {
                'backward_ratio': 0.0,
                'attention_to_generated': 0.0,
                'attention_to_input': 0.0
            }
    
    def compare_reflection_vs_normal(self, reflection_prompt, normal_prompt):
        """Compare backward attention patterns between conditions"""
        
        print("🔍 Analyzing backward attention patterns...")
        
        # Clear memory before starting
        self.clear_memory()
        
        # Generate with reflection prompt
        print("  📝 Generating with reflection prompt...")
        reflection_metrics = self.analyze_single_generation(reflection_prompt)
        
        # Clear memory between generations
        self.clear_memory()
        
        # Generate with normal prompt
        print("  📝 Generating with normal prompt...")
        normal_metrics = self.analyze_single_generation(normal_prompt)
        
        # Clear memory after generation
        self.clear_memory()
        
        # Calculate comparison metrics
        comparison = self.calculate_comparison_metrics(reflection_metrics, normal_metrics)
        
        return comparison
    
    def calculate_comparison_metrics(self, reflection_metrics, normal_metrics):
        """Calculate comparison metrics between reflection and normal conditions"""
        
        # Calculate averages
        refl_avg_backward = np.mean(reflection_metrics['backward_ratios']) if reflection_metrics['backward_ratios'] else 0.0
        norm_avg_backward = np.mean(normal_metrics['backward_ratios']) if normal_metrics['backward_ratios'] else 0.0
        
        refl_avg_generated = np.mean(reflection_metrics['attention_to_generated']) if reflection_metrics['attention_to_generated'] else 0.0
        norm_avg_generated = np.mean(normal_metrics['attention_to_generated']) if normal_metrics['attention_to_generated'] else 0.0
        
        # Calculate differences
        backward_diff = refl_avg_backward - norm_avg_backward
        generated_diff = refl_avg_generated - norm_avg_generated
        
        return {
            'reflection_metrics': {
                'avg_backward_ratio': refl_avg_backward,
                'avg_attention_to_generated': refl_avg_generated,
                'reflection_points': reflection_metrics['reflection_points'],
                'total_steps': len(reflection_metrics['steps'])
            },
            'normal_metrics': {
                'avg_backward_ratio': norm_avg_backward,
                'avg_attention_to_generated': norm_avg_generated,
                'reflection_points': normal_metrics['reflection_points'],
                'total_steps': len(normal_metrics['steps'])
            },
            'differences': {
                'backward_ratio_diff': backward_diff,
                'attention_to_generated_diff': generated_diff
            },
            'raw_data': {
                'reflection_backward_ratios': reflection_metrics['backward_ratios'],
                'normal_backward_ratios': normal_metrics['backward_ratios']
            }
        }
    
    def visualize_comparison(self, comparison):
        """Create visualizations of the comparison"""
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Average backward attention comparison
            conditions = ['With Reflection', 'Without Reflection']
            backward_values = [
                comparison['reflection_metrics']['avg_backward_ratio'],
                comparison['normal_metrics']['avg_backward_ratio']
            ]
            
            ax1.bar(conditions, backward_values, color=['orange', 'blue'], alpha=0.7)
            ax1.set_ylabel('Average Backward Attention Ratio')
            ax1.set_title('Backward Attention Comparison')
            ax1.set_ylim(0, max(backward_values) * 1.2 if max(backward_values) > 0 else 1)
            
            # Plot 2: Attention to generated content comparison
            generated_values = [
                comparison['reflection_metrics']['avg_attention_to_generated'],
                comparison['normal_metrics']['avg_attention_to_generated']
            ]
            
            ax2.bar(conditions, generated_values, color=['orange', 'blue'], alpha=0.7)
            ax2.set_ylabel('Average Attention to Generated Content')
            ax2.set_title('Attention to Generated Content')
            ax2.set_ylim(0, max(generated_values) * 1.2 if max(generated_values) > 0 else 1)
            
            # Plot 3: Backward attention over time (reflection)
            refl_ratios = comparison['raw_data']['reflection_backward_ratios']
            if refl_ratios:
                ax3.plot(range(len(refl_ratios)), refl_ratios, 'o-', color='orange', alpha=0.7)
                ax3.set_xlabel('Generation Step')
                ax3.set_ylabel('Backward Attention Ratio')
                ax3.set_title('Backward Attention Over Time (With Reflection)')
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Backward attention over time (normal)
            norm_ratios = comparison['raw_data']['normal_backward_ratios']
            if norm_ratios:
                ax4.plot(range(len(norm_ratios)), norm_ratios, 'o-', color='blue', alpha=0.7)
                ax4.set_xlabel('Generation Step')
                ax4.set_ylabel('Backward Attention Ratio')
                ax4.set_title('Backward Attention Over Time (Without Reflection)')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Error creating visualization: {e}")
    
    def print_analysis_summary(self, comparison):
        """Print a summary of the analysis"""
        
        print("\n" + "="*60)
        print("📊 EFFICIENT BACKWARD ATTENTION ANALYSIS RESULTS")
        print("="*60)
        
        refl = comparison['reflection_metrics']
        norm = comparison['normal_metrics']
        diff = comparison['differences']
        
        print(f"🔄 Average Backward Attention Ratio:")
        print(f"  With Reflection:    {refl['avg_backward_ratio']:.4f}")
        print(f"  Without Reflection: {norm['avg_backward_ratio']:.4f}")
        print(f"  Difference:         {diff['backward_ratio_diff']:.4f}")
        
        print(f"\n🎯 Attention to Generated Content:")
        print(f"  With Reflection:    {refl['avg_attention_to_generated']:.4f}")
        print(f"  Without Reflection: {norm['avg_attention_to_generated']:.4f}")
        print(f"  Difference:         {diff['attention_to_generated_diff']:.4f}")
        
        print(f"\n⚡ Reflection Points Found:")
        print(f"  With Reflection:    {len(refl['reflection_points'])}")
        print(f"  Without Reflection: {len(norm['reflection_points'])}")
        
        if refl['reflection_points']:
            print(f"\n🔍 Reflection Point Details:")
            for point in refl['reflection_points'][:3]:  # Show first 3
                print(f"  Step {point['step']}: '{point['token']}' -> {point['backward_ratio']:.4f}")
        
        # Interpretation
        print(f"\n💡 INTERPRETATION:")
        if diff['backward_ratio_diff'] > 0.05:
            print("  ✅ Strong evidence of increased backward attention during reflection!")
        elif diff['backward_ratio_diff'] > 0.02:
            print("  ⚠️ Moderate evidence of increased backward attention during reflection.")
        elif diff['backward_ratio_diff'] > 0.01:
            print("  🔍 Weak evidence of increased backward attention during reflection.")
        else:
            print("  ❌ No clear evidence of increased backward attention during reflection.")
        
        print(f"\n📈 Statistical Summary:")
        print(f"  Total steps analyzed: {refl['total_steps']} vs {norm['total_steps']}")
        print(f"  Memory usage: Optimized for efficiency")

def main():
    print("=" * 80)
    print("🔄 EFFICIENT BACKWARD ATTENTION ANALYSIS FOR REFLECTION")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = EfficientBackwardAttentionAnalyzer()
    
    # Create test prompts (shorter to save memory)
    reflection_prompt = """Math problem: A train travels 120 miles in 2 hours, stops 30 minutes, then travels 180 miles in 3 hours. What's the average speed?

Let me solve step by step:
1. Total distance = 120 + 180 = 300 miles
2. Total time = 2 + 0.5 + 3 = 5.5 hours
3. Let me REFLECT and CHECK my calculation
4. Average speed = 300/5.5 = 54.55 mph

Answer:"""

    normal_prompt = """Math problem: A train travels 120 miles in 2 hours, stops 30 minutes, then travels 180 miles in 3 hours. What's the average speed?

Solution:"""
    
    # Run analysis
    comparison = analyzer.compare_reflection_vs_normal(reflection_prompt, normal_prompt)
    
    # Print summary
    analyzer.print_analysis_summary(comparison)
    
    # Create visualizations
    analyzer.visualize_comparison(comparison)
    
    # Save results
    with open("efficient_backward_attention_results.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print("\n💾 Results saved to 'efficient_backward_attention_results.json'")

if __name__ == "__main__":
    main() 