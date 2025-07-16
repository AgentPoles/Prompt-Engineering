#!/usr/bin/env python3
"""
Test script to verify the BFloat16 fix for interpretability analysis
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def test_bfloat16_fix():
    """Test that we can properly convert BFloat16 tensors to numpy"""
    print("🧪 Testing BFloat16 -> numpy conversion fix...")
    
    # Create a simple test case
    test_tensor = torch.randn(10, 20).to(torch.bfloat16)
    print(f"Original tensor dtype: {test_tensor.dtype}")
    
    # Test the conversion
    try:
        # This should fail
        numpy_array_direct = test_tensor.cpu().numpy()
        print("❌ Direct conversion worked unexpectedly")
    except TypeError as e:
        print(f"✅ Direct conversion failed as expected: {e}")
    
    # Test the fix
    try:
        numpy_array_fixed = test_tensor.float().cpu().numpy()
        print(f"✅ Fixed conversion worked! Shape: {numpy_array_fixed.shape}")
        print(f"✅ Result dtype: {numpy_array_fixed.dtype}")
    except Exception as e:
        print(f"❌ Fixed conversion failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test that the model loads correctly"""
    print("\n🔧 Testing model loading...")
    
    try:
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
        
        print("✅ Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None, None

def test_hidden_states_extraction(model, tokenizer):
    """Test that we can extract hidden states without BFloat16 errors"""
    print("\n🧠 Testing hidden states extraction...")
    
    if model is None or tokenizer is None:
        print("❌ Cannot test - model not loaded")
        return False
    
    try:
        # Test prompt
        prompt = "Write a basketball report about the game."
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=100)
        input_ids = inputs["input_ids"].to(model.device)
        
        # Get hidden states
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            
            # Test the extraction with the fix
            layer_states = []
            for layer_hidden in outputs.hidden_states:
                # Use the fixed conversion
                last_token_state = layer_hidden[0, -1, :].float().cpu().numpy()
                layer_states.append(last_token_state)
            
            print(f"✅ Extracted {len(layer_states)} layer states")
            print(f"✅ Final state shape: {layer_states[-1].shape}")
            print(f"✅ Final state dtype: {layer_states[-1].dtype}")
            
            # Test some basic operations
            representation_norm = np.linalg.norm(layer_states[-1])
            print(f"✅ Representation norm: {representation_norm:.4f}")
            
            return True
            
    except Exception as e:
        print(f"❌ Hidden states extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 TESTING INTERPRETABILITY FIXES")
    print("=" * 60)
    
    # Test 1: BFloat16 conversion
    test1_passed = test_bfloat16_fix()
    
    # Test 2: Model loading
    model, tokenizer = test_model_loading()
    
    # Test 3: Hidden states extraction
    test3_passed = test_hidden_states_extraction(model, tokenizer)
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("-" * 30)
    print(f"BFloat16 fix: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Model loading: {'✅ PASSED' if model is not None else '❌ FAILED'}")
    print(f"Hidden states: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    if test1_passed and model is not None and test3_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The interpretability analysis should now work correctly!")
        return True
    else:
        print("\n❌ Some tests failed - check the output above")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 Ready to run the full interpretability analysis!")
        print("💡 Use: exec(open('basketball_interpretability_analysis.py').read())")
    else:
        print("\n🔧 Please check the errors above before running the full analysis") 