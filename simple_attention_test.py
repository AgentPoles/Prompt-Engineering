#!/usr/bin/env python3
"""
Simple Attention Test - Basic version to test if attention capture works
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def test_attention_capture():
    print("=" * 60)
    print("🧠 SIMPLE ATTENTION CAPTURE TEST")
    print("=" * 60)
    
    # Load model with eager attention
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
        attn_implementation="eager",  # Force eager attention
    )
    print("✅ Model loaded!")
    
    # Simple test prompt
    test_prompt = """Write a short basketball report. First reflect on your approach."""
    
    inputs = tokenizer(
        test_prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=500,
        padding=True
    )
    
    print(f"📝 Input tokens: {inputs['input_ids'].shape[1]}")
    
    # Test 1: Try to get attention during forward pass
    print("\n🔍 Test 1: Forward pass with attention")
    try:
        with torch.no_grad():
            outputs = model(
                inputs['input_ids'].to(model.device),
                attention_mask=inputs['attention_mask'].to(model.device),
                output_attentions=True
            )
        
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            print(f"✅ Forward pass attention capture successful!")
            print(f"   Number of layers: {len(outputs.attentions)}")
            print(f"   Attention shape: {outputs.attentions[0].shape}")
        else:
            print("❌ Forward pass attention capture failed")
            
    except Exception as e:
        print(f"❌ Forward pass error: {e}")
    
    # Test 2: Try generation with fewer tokens
    print("\n🔍 Test 2: Generation with attention (limited)")
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'].to(model.device),
                attention_mask=inputs['attention_mask'].to(model.device),
                max_new_tokens=20,  # Very limited generation
                do_sample=False,    # Greedy decoding
                output_attentions=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            print(f"✅ Generation attention capture successful!")
            print(f"   Number of generation steps: {len(outputs.attentions)}")
            if len(outputs.attentions) > 0:
                print(f"   First step attention layers: {len(outputs.attentions[0])}")
        else:
            print("❌ Generation attention capture failed")
            print(f"   Available attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
            
    except Exception as e:
        print(f"❌ Generation error: {e}")
    
    # Test 3: Try different attention implementation
    print("\n🔍 Test 3: Check model configuration")
    try:
        print(f"   Model config attention: {getattr(model.config, 'attn_implementation', 'Not specified')}")
        print(f"   Model has attention: {hasattr(model, 'get_attention_mask')}")
        
        # Check if we can access the model's attention modules
        attention_modules = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                attention_modules.append(name)
        
        print(f"   Found attention modules: {len(attention_modules)}")
        if attention_modules:
            print(f"   Example modules: {attention_modules[:3]}")
            
    except Exception as e:
        print(f"❌ Configuration check error: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS:")
    print("=" * 60)
    
    # Try alternative approach
    print("Alternative approach: Use model internals directly")
    
    try:
        # Get a simple forward pass and hook into attention
        def attention_hook(module, input, output):
            print(f"   Attention hook triggered on {module.__class__.__name__}")
            return output
        
        # Register hooks on attention modules
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() and 'self_attn' in name:
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
                if len(hooks) >= 3:  # Limit to first 3 layers
                    break
        
        if hooks:
            print(f"📍 Registered {len(hooks)} attention hooks")
            
            # Simple forward pass
            with torch.no_grad():
                outputs = model(
                    inputs['input_ids'].to(model.device)[:, :50],  # Limit input length
                    attention_mask=inputs['attention_mask'].to(model.device)[:, :50]
                )
            
            # Clean up hooks
            for hook in hooks:
                hook.remove()
            
            print("✅ Hook-based attention monitoring successful!")
        else:
            print("❌ No attention modules found for hooking")
            
    except Exception as e:
        print(f"❌ Hook-based approach error: {e}")

if __name__ == "__main__":
    test_attention_capture() 