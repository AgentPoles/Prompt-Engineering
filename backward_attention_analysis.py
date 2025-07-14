#!/usr/bin/env python3
"""
Backward Attention Analysis for Reflection Behavior
Tracks whether the model pays more attention to its own previous output
when encountering reflection prompts during generation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import re
import gc

class BackwardAttentionAnalyzer:
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
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,  # Use 8-bit for better accuracy
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
            ),
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded for backward attention tracking!")
    
    def clear_memory(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def generate_with_attention_tracking(self, prompt, max_new_tokens=50):
        """Generate text step by step while tracking attention patterns"""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1000,  # Reduced for memory efficiency
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
            'attention_patterns': []
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
            
            # Store attention patterns (only the metrics we need)
            if outputs.attentions:
                # Calculate backward attention immediately, don't store full patterns
                backward_metrics = self.calculate_backward_attention(
                    outputs.attentions,
                    generation_data['input_length'],
                    len(generation_data['generated_tokens'])
                )
                # Clear the attention data immediately
                del outputs.attentions
            else:
                backward_metrics = {
                    'backward_attention_ratio': 0.0,
                    'attention_to_generated': 0.0,
                    'attention_to_input': 0.0,
                    'attention_spread': 0.0
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
                'attention_spread': 0.0
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
            
            # Calculate attention spread (how distributed vs focused)
            attention_entropy = -torch.sum(
                current_attention * torch.log(current_attention + 1e-10)
            ).item()
            
            return {
                'backward_attention_ratio': backward_attention_ratio,
                'attention_to_generated': attention_to_generated,
                'attention_to_input': attention_to_input,
                'attention_spread': attention_entropy
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating backward attention: {e}")
            return {
                'backward_attention_ratio': 0.0,
                'attention_to_generated': 0.0,
                'attention_to_input': 0.0,
                'attention_spread': 0.0
            }
    
    def analyze_reflection_vs_normal(self, reflection_prompt, normal_prompt):
        """Compare backward attention patterns between reflection and normal generation"""
        
        print("🔍 Analyzing backward attention patterns...")
        
        # Clear memory before starting
        self.clear_memory()
        
        # Generate with both prompts
        print("  📝 Generating with reflection prompt...")
        reflection_data = self.generate_with_attention_tracking(reflection_prompt)
        
        # Clear memory between generations
        self.clear_memory()
        
        print("  📝 Generating with normal prompt...")
        normal_data = self.generate_with_attention_tracking(normal_prompt)
        
        # Clear memory after generation
        self.clear_memory()
        
        # Analyze patterns
        reflection_analysis = self.analyze_backward_patterns(reflection_data)
        normal_analysis = self.analyze_backward_patterns(normal_data)
        
        # Compare results
        comparison = {
            'reflection_analysis': reflection_analysis,
            'normal_analysis': normal_analysis,
            'reflection_raw_data': reflection_data,  # Add raw data for visualization
            'normal_raw_data': normal_data,  # Add raw data for visualization
            'differences': {
                'avg_backward_ratio_diff': (
                    reflection_analysis['avg_backward_ratio'] - 
                    normal_analysis['avg_backward_ratio']
                ),
                'reflection_spikes': reflection_analysis['reflection_spikes'],
                'normal_spikes': normal_analysis['reflection_spikes']
            }
        }
        
        return comparison
    
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
            'backward_ratio_trend': backward_ratios,
            'total_steps': len(generation_data['steps'])
        }
    
    def visualize_backward_attention(self, generation_data, title="Backward Attention Over Time"):
        """Visualize backward attention patterns over generation steps"""
        
        steps = []
        backward_ratios = []
        attention_to_generated = []
        reflection_points = []
        
        for step_data in generation_data['steps']:
            steps.append(step_data['step'])
            backward_ratios.append(step_data['backward_metrics']['backward_attention_ratio'])
            attention_to_generated.append(step_data['backward_metrics']['attention_to_generated'])
            
            if step_data['is_reflection_step']:
                reflection_points.append(step_data['step'])
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Backward attention ratio
        ax1.plot(steps, backward_ratios, 'b-', linewidth=2, label='Backward Attention Ratio')
        ax1.axhline(y=np.mean(backward_ratios), color='r', linestyle='--', alpha=0.7, label='Average')
        
        # Mark reflection points
        for point in reflection_points:
            ax1.axvline(x=point, color='orange', linestyle=':', alpha=0.8, label='Reflection Point' if point == reflection_points[0] else "")
        
        ax1.set_xlabel('Generation Step')
        ax1.set_ylabel('Backward Attention Ratio')
        ax1.set_title('Backward Attention Ratio Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Attention to generated tokens
        ax2.plot(steps, attention_to_generated, 'g-', linewidth=2, label='Attention to Generated')
        ax2.axhline(y=np.mean(attention_to_generated), color='r', linestyle='--', alpha=0.7, label='Average')
        
        # Mark reflection points
        for point in reflection_points:
            ax2.axvline(x=point, color='orange', linestyle=':', alpha=0.8, label='Reflection Point' if point == reflection_points[0] else "")
        
        ax2.set_xlabel('Generation Step')
        ax2.set_ylabel('Attention to Generated Tokens')
        ax2.set_title('Attention to Generated Content Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def print_analysis_summary(self, comparison):
        """Print a summary of the backward attention analysis"""
        
        print("\n" + "="*60)
        print("📊 BACKWARD ATTENTION ANALYSIS RESULTS")
        print("="*60)
        
        refl = comparison['reflection_analysis']
        norm = comparison['normal_analysis']
        diff = comparison['differences']
        
        print(f"🔄 Average Backward Attention Ratio:")
        print(f"  With Reflection:    {refl['avg_backward_ratio']:.4f}")
        print(f"  Without Reflection: {norm['avg_backward_ratio']:.4f}")
        print(f"  Difference:         {diff['avg_backward_ratio_diff']:.4f}")
        
        print(f"\n🎯 Attention to Generated Content:")
        print(f"  With Reflection:    {refl['avg_attention_to_generated']:.4f}")
        print(f"  Without Reflection: {norm['avg_attention_to_generated']:.4f}")
        
        print(f"\n⚡ Reflection Spikes:")
        print(f"  Reflection condition: {len(diff['reflection_spikes'])} spikes")
        print(f"  Normal condition:     {len(diff['normal_spikes'])} spikes")
        
        if diff['reflection_spikes']:
            print(f"\n🔍 Reflection Spike Details:")
            for spike in diff['reflection_spikes'][:3]:  # Show first 3
                print(f"  Step {spike['step']}: '{spike['token']}' -> {spike['backward_ratio']:.4f}")
        
        # Interpretation
        print(f"\n💡 INTERPRETATION:")
        if diff['avg_backward_ratio_diff'] > 0.05:
            print("  ✅ Strong evidence of backward attention during reflection!")
            print("     The model shows significantly more attention to previously generated content")
            print("     when reflection prompts are present, suggesting genuine review behavior.")
        elif diff['avg_backward_ratio_diff'] > 0.02:
            print("  ⚠️ Moderate evidence of backward attention during reflection.")
            print("     The model shows some increased attention to previous content during reflection.")
        elif diff['avg_backward_ratio_diff'] > 0.01:
            print("  🔍 Weak evidence of backward attention during reflection.")
            print("     Small increase in backward attention, may indicate limited reflection.")
        else:
            print("  ❌ No clear evidence of increased backward attention during reflection.")
            print("     The model doesn't appear to significantly review previous content")
            print("     when encountering reflection prompts.")
        
        print(f"\n📈 Analysis Configuration:")
        print(f"  - 8-bit quantization used for accuracy")
        print(f"  - Memory cleared between generation steps")
        print(f"  - Attention patterns processed immediately, not stored")
        print(f"  - Step-by-step generation preserved for accurate analysis")

def main():
    print("=" * 80)
    print("🔄 BACKWARD ATTENTION ANALYSIS FOR REFLECTION")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = BackwardAttentionAnalyzer()
    
    # Create test prompts (shorter for memory efficiency)
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
    comparison = analyzer.analyze_reflection_vs_normal(reflection_prompt, normal_prompt)
    
    # Print summary
    analyzer.print_analysis_summary(comparison)
    
    # Visualize results
    analyzer.visualize_backward_attention(
        comparison['reflection_raw_data'], 
        "Backward Attention - WITH Reflection"
    )
    
    analyzer.visualize_backward_attention(
        comparison['normal_raw_data'], 
        "Backward Attention - WITHOUT Reflection"
    )
    
    # Save results
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    json_results = convert_numpy(comparison)
    
    with open("backward_attention_results.json", "w") as f:
        json.dump(json_results, f, indent=2)
    
    print("\n💾 Backward attention analysis results saved to 'backward_attention_results.json'")

if __name__ == "__main__":
    main() 