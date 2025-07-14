#!/usr/bin/env python3
"""
Attention Analysis for Reflection Behavior
Visualizes where the model focuses when reflecting vs not reflecting
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json

class AttentionAnalyzer:
    def __init__(self, model_id="Qwen/Qwen2.5-14B-Instruct"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load model with attention output enabled"""
        print(f"Loading {self.model_id} for attention analysis...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # KEY: Force eager attention for output_attentions
        )
        print("✅ Model loaded with attention tracking!")
    
    def get_attention_patterns(self, prompt, max_new_tokens=100):
        """Generate text and capture attention patterns"""
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1500,
            padding=True
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                output_attentions=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )
        
        # Extract attention weights - handle case where they might be None
        attentions = getattr(outputs, 'attentions', None)
        generated_tokens = outputs.sequences[0]
        
        if attentions is None:
            print("⚠️ Warning: No attention weights captured")
        
        return {
            'input_ids': input_ids[0],
            'generated_tokens': generated_tokens,
            'attentions': attentions,
            'input_length': input_ids.shape[1]
        }
    
    def analyze_attention_to_reflection_words(self, attention_data, reflection_keywords):
        """Analyze how much attention is paid to reflection-related words"""
        
        input_tokens = attention_data['input_ids']
        input_length = attention_data['input_length']
        
        # Convert tokens to text to find reflection words
        try:
            input_text = self.tokenizer.decode(input_tokens, skip_special_tokens=True)
            token_strings = [self.tokenizer.decode([token]) for token in input_tokens]
        except Exception as e:
            print(f"⚠️ Error decoding tokens: {e}")
            return {
                'reflection_positions': [],
                'attention_to_reflection': [],
                'total_reflection_attention': 0.0
            }
        
        # Find reflection word positions
        reflection_positions = []
        found_keywords = []
        for i, token_str in enumerate(token_strings):
            for keyword in reflection_keywords:
                if keyword.lower() in token_str.lower():
                    reflection_positions.append(i)
                    found_keywords.append(f"{keyword}@{i}")
                    break
        
        print(f"Found reflection keywords at positions: {reflection_positions}")
        print(f"Keywords found: {found_keywords}")
        
        # Debug: Show some tokens around reflection positions
        if reflection_positions:
            print("Debug: Tokens around reflection keywords:")
            for pos in reflection_positions[:3]:  # Show first 3
                start = max(0, pos-2)
                end = min(len(token_strings), pos+3)
                context = [token_strings[j] for j in range(start, end)]
                print(f"  Position {pos}: {' '.join(context)}")
        else:
            print("Debug: No reflection keywords found. Sample tokens:")
            sample_tokens = token_strings[:20] if len(token_strings) > 20 else token_strings
            print(f"  First tokens: {' '.join(sample_tokens)}")
            print(f"  Looking for keywords: {reflection_keywords}")
        
        # Analyze attention to these positions across all generation steps
        attention_to_reflection = []
        
        if attention_data['attentions'] is not None and len(attention_data['attentions']) > 0:
            for i, step_attention in enumerate(attention_data['attentions']):
                try:
                    if step_attention is not None and len(step_attention) > 0:
                        # step_attention is tuple of layer attentions
                        # Each layer attention: [batch_size, num_heads, seq_len, seq_len]
                        layer_attention = step_attention[0]  # Use first (and only) batch
                        
                        if i == 0:  # Debug info for first step only
                            print(f"Debug: Step {i} attention shape: {layer_attention.shape}")
                            print(f"Debug: Input length: {input_length}")
                            print(f"Debug: Attention dtype: {layer_attention.dtype}")
                        
                        if layer_attention is not None:
                            # Convert to float32 to avoid precision issues
                            layer_attention = layer_attention.float()
                            
                            # Average across all layers and heads
                            # layer_attention shape: [num_heads, seq_len, seq_len]
                            avg_attention = torch.mean(layer_attention, dim=0)  # Average across heads
                            
                            # Get attention from last position to input positions
                            if avg_attention.shape[0] > 0:
                                last_token_attention = avg_attention[-1, :input_length]  # Last token's attention to input
                                
                                # Sum attention to reflection positions
                                reflection_attention = 0.0
                                for pos in reflection_positions:
                                    if pos < len(last_token_attention):
                                        try:
                                            # Ensure we get a scalar value
                                            att_val = last_token_attention[pos]
                                            if att_val.numel() == 1:
                                                reflection_attention += att_val.item()
                                            else:
                                                # If it's not a scalar, take the mean
                                                reflection_attention += att_val.mean().item()
                                        except Exception as e:
                                            print(f"⚠️ Error extracting attention at position {pos}: {e}")
                                            continue
                                attention_to_reflection.append(reflection_attention)
                        
                except Exception as e:
                    print(f"⚠️ Error processing attention step {i}: {e}")
                    continue
        else:
            print("⚠️ No attention data available for analysis")
        
        return {
            'reflection_positions': reflection_positions,
            'attention_to_reflection': attention_to_reflection,
            'total_reflection_attention': sum(attention_to_reflection) if attention_to_reflection else 0.0
        }
    
    def compare_attention_patterns(self, reflection_prompt, no_reflection_prompt):
        """Compare attention patterns between reflection and no-reflection"""
        
        reflection_keywords = ['reflect', 'review', 'check', 'assess', 'evaluate', 'analyze', 'verify']
        
        print("🔍 Analyzing attention patterns...")
        
        # Get attention for both conditions
        refl_attention = self.get_attention_patterns(reflection_prompt)
        no_refl_attention = self.get_attention_patterns(no_reflection_prompt)
        
        # Analyze attention to reflection words
        refl_analysis = self.analyze_attention_to_reflection_words(refl_attention, reflection_keywords)
        no_refl_analysis = self.analyze_attention_to_reflection_words(no_refl_attention, reflection_keywords)
        
        print(f"📊 Attention to reflection words:")
        print(f"  With reflection: {refl_analysis['total_reflection_attention']:.4f}")
        print(f"  Without reflection: {no_refl_analysis['total_reflection_attention']:.4f}")
        
        return {
            'reflection_attention': refl_attention,
            'no_reflection_attention': no_refl_attention,
            'reflection_analysis': refl_analysis,
            'no_reflection_analysis': no_refl_analysis
        }
    
    def visualize_attention_heatmap(self, attention_data, title="Attention Heatmap"):
        """Create attention heatmap visualization"""
        
        if not attention_data['attentions'] or len(attention_data['attentions']) == 0:
            print(f"No attention data available for visualization: {title}")
            return
        
        # Take first few generation steps for visualization
        valid_steps = []
        for i, step_attention in enumerate(attention_data['attentions']):
            if step_attention is not None and len(step_attention) > 0:
                valid_steps.append(i)
                if len(valid_steps) >= 5:  # Limit to 5 steps
                    break
        
        if not valid_steps:
            print(f"No valid attention steps found for visualization: {title}")
            return
        
        steps_to_show = len(valid_steps)
        fig, axes = plt.subplots(1, steps_to_show, figsize=(15, 4))
        if steps_to_show == 1:
            axes = [axes]
        
        for plot_idx, step_idx in enumerate(valid_steps):
            try:
                step_attention = attention_data['attentions'][step_idx][0]  # [num_heads, seq_len, seq_len]
                
                if step_attention is not None:
                    # Average across heads
                    avg_attention = torch.mean(step_attention, dim=0)  # [seq_len, seq_len]
                    
                    # Focus on attention to input tokens
                    if avg_attention.shape[0] > 0:
                        input_attention = avg_attention[-1, :attention_data['input_length']]  # Last token's attention to input
                        
                        # Convert to float32 for visualization (fix BFloat16 issue)
                        input_attention_viz = input_attention.float().cpu().numpy().reshape(1, -1)
                        
                        # Create heatmap
                        sns.heatmap(
                            input_attention_viz,
                            ax=axes[plot_idx],
                            cmap='Blues',
                            cbar=True,
                            xticklabels=False,
                            yticklabels=False
                        )
                        axes[plot_idx].set_title(f"Step {step_idx+1}")
                    else:
                        axes[plot_idx].text(0.5, 0.5, "No data", ha='center', va='center')
                        axes[plot_idx].set_title(f"Step {step_idx+1} (No data)")
                        
            except Exception as e:
                print(f"⚠️ Error visualizing step {step_idx}: {e}")
                axes[plot_idx].text(0.5, 0.5, "Error", ha='center', va='center')
                axes[plot_idx].set_title(f"Step {step_idx+1} (Error)")
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

def main():
    print("=" * 80)
    print("🧠 ATTENTION ANALYSIS FOR REFLECTION")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = AttentionAnalyzer()
    
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
    
    # Compare attention patterns
    results = analyzer.compare_attention_patterns(reflection_prompt, no_reflection_prompt)
    
    # Visualize attention heatmaps
    analyzer.visualize_attention_heatmap(
        results['reflection_attention'], 
        "Attention with Reflection"
    )
    
    analyzer.visualize_attention_heatmap(
        results['no_reflection_attention'], 
        "Attention without Reflection"
    )
    
    # Save results
    with open("attention_analysis_results.json", "w") as f:
        # Convert tensors to lists for JSON serialization
        json_results = {
            'reflection_attention_sum': results['reflection_analysis']['total_reflection_attention'],
            'no_reflection_attention_sum': results['no_reflection_analysis']['total_reflection_attention'],
            'reflection_positions': results['reflection_analysis']['reflection_positions'],
            'attention_difference': results['reflection_analysis']['total_reflection_attention'] - results['no_reflection_analysis']['total_reflection_attention']
        }
        json.dump(json_results, f, indent=2)
    
    print("💾 Attention analysis results saved!")

if __name__ == "__main__":
    main() 