#!/usr/bin/env python3
"""
Simplified Attention Analysis - Just the core functionality
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def simple_attention_analysis():
    print("=" * 80)
    print("🧠 SIMPLIFIED ATTENTION ANALYSIS")
    print("=" * 80)
    
    # Load model
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    print(f"Loading {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    print("✅ Model loaded!")
    
    # Test prompts
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

    reflection_keywords = ['reflect', 'review', 'check', 'assess', 'evaluate', 'analyze', 'verify']
    
    def analyze_prompt(prompt, label):
        print(f"\n{'='*60}")
        print(f"🔍 ANALYZING: {label}")
        print('='*60)
        
        # Tokenize
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1000,
            padding=True
        )
        
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        print(f"📝 Input tokens: {input_ids.shape[1]}")
        
        # Find reflection keywords in input
        token_strings = [tokenizer.decode([token]) for token in input_ids[0]]
        reflection_positions = []
        found_keywords = []
        
        for i, token_str in enumerate(token_strings):
            for keyword in reflection_keywords:
                if keyword.lower() in token_str.lower():
                    reflection_positions.append(i)
                    found_keywords.append(f"{keyword}@{i}")
                    break
        
        print(f"🎯 Found reflection keywords: {found_keywords}")
        print(f"📍 Positions: {reflection_positions}")
        
        # Generate with attention
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,  # Keep it short
                    do_sample=False,    # Greedy for consistency
                    output_attentions=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Analyze attention
            attentions = getattr(outputs, 'attentions', None)
            
            if attentions is not None and len(attentions) > 0:
                print(f"✅ Captured attention for {len(attentions)} generation steps")
                
                total_reflection_attention = 0.0
                valid_steps = 0
                
                for step_i, step_attention in enumerate(attentions):
                    if step_attention is not None and len(step_attention) > 0:
                        try:
                            # Get first layer attention: [num_heads, seq_len, seq_len]
                            layer_attention = step_attention[0].float()
                            
                            # Average across heads
                            avg_attention = torch.mean(layer_attention, dim=0)
                            
                            # Get last token's attention to input
                            if avg_attention.shape[0] > 0:
                                last_token_att = avg_attention[-1, :input_ids.shape[1]]
                                
                                # Sum attention to reflection positions
                                step_reflection_att = 0.0
                                for pos in reflection_positions:
                                    if pos < len(last_token_att):
                                        att_val = last_token_att[pos]
                                        if att_val.numel() == 1:
                                            step_reflection_att += att_val.item()
                                        else:
                                            step_reflection_att += att_val.mean().item()
                                
                                total_reflection_attention += step_reflection_att
                                valid_steps += 1
                                
                                if step_i < 3:  # Show first 3 steps
                                    print(f"  Step {step_i}: reflection attention = {step_reflection_att:.4f}")
                        
                        except Exception as e:
                            print(f"⚠️ Error in step {step_i}: {e}")
                            continue
                
                avg_reflection_attention = total_reflection_attention / max(valid_steps, 1)
                print(f"📊 Average reflection attention: {avg_reflection_attention:.4f}")
                return avg_reflection_attention
                
            else:
                print("❌ No attention data captured")
                return 0.0
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return 0.0
    
    # Analyze both prompts
    refl_attention = analyze_prompt(reflection_prompt, "WITH REFLECTION")
    no_refl_attention = analyze_prompt(no_reflection_prompt, "WITHOUT REFLECTION")
    
    # Compare results
    print(f"\n{'='*80}")
    print("📊 COMPARISON RESULTS")
    print('='*80)
    print(f"With Reflection:    {refl_attention:.4f}")
    print(f"Without Reflection: {no_refl_attention:.4f}")
    print(f"Difference:         {refl_attention - no_refl_attention:+.4f}")
    
    if refl_attention > no_refl_attention * 1.5:
        print("✅ Reflection prompt shows significantly higher attention to reflection keywords!")
    elif refl_attention > no_refl_attention * 1.1:
        print("🤔 Reflection prompt shows slightly higher attention to reflection keywords")
    else:
        print("❌ No significant difference in attention patterns")

if __name__ == "__main__":
    simple_attention_analysis() 