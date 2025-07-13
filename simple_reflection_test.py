#!/usr/bin/env python3
"""
Simple Reflection Test - Does the model actually reflect?
"""

import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load model (using your working 14B setup)
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
print(f"Loading {MODEL_ID} for reflection testing...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print("✅ Model loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading Qwen: {e}")
    print("Please ensure the model is available or modify MODEL_ID")
    exit(1)

# ============================================================================
# REFLECTION TESTS
# ============================================================================

def create_visible_reflection_prompt():
    """Create a prompt that forces the model to show its reflection process"""
    
    return """# Identity
You are a seasoned sports journalist writing about basketball games.

# Task
Write a basketball game report, but FIRST show your internal reflection process.

# Instructions
1. **SHOW YOUR REFLECTION PROCESS** - Don't hide it, show it explicitly:
   
   ## REFLECTION PROCESS:
   - **Data Review**: What data do I have? What's missing?
   - **Structure Check**: Am I following the required format?
   - **Word Count Planning**: How will I reach 350-450 words?
   - **Fact Verification**: Are all my statements supported by data?
   - **Quality Assessment**: Is this professional sports journalism?
   - **Revision Notes**: What do I need to change?

2. **THEN WRITE THE FINAL REPORT** after reflection:
   
   ## FINAL REPORT:
   [Your polished report here]

# Game Data
Date: January 29, 2018
Location: Philips Arena, Atlanta
Teams: Hawks (15-35) vs Timberwolves (32-21)
Final Score: Hawks 101 - Timberwolves 96
Winner: Hawks

Generate your response:"""

def create_no_reflection_prompt():
    """Create a standard prompt without reflection"""
    
    return """# Identity
You are a seasoned sports journalist writing about basketball games.

# Task
Write a basketball game report (350-450 words) based on the data provided.

# Game Data
Date: January 29, 2018
Location: Philips Arena, Atlanta
Teams: Hawks (15-35) vs Timberwolves (32-21)
Final Score: Hawks 101 - Timberwolves 96
Winner: Hawks

Generate your report:"""

def generate_and_analyze(prompt, label):
    """Generate report and analyze the output"""
    
    print(f"\n{'='*60}")
    print(f"🔍 TESTING: {label}")
    print('='*60)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
    input_length = inputs['input_ids'].shape[1]
    
    print(f"📊 Input tokens: {input_length}")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs.to(model.device),
            max_new_tokens=600,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    result = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    # Analyze
    analysis = analyze_output(result, label)
    
    print(f"📝 Generated Output:")
    print("-" * 60)
    print(result)
    print("-" * 60)
    
    return result, analysis

def analyze_output(text, label):
    """Analyze the output to see if reflection occurred"""
    
    import re
    
    # Check for reflection indicators
    reflection_sections = {
        'reflection_process': bool(re.search(r'REFLECTION PROCESS', text, re.IGNORECASE)),
        'data_review': bool(re.search(r'data review|data.*?missing|what data', text, re.IGNORECASE)),
        'structure_check': bool(re.search(r'structure.*?check|format.*?check', text, re.IGNORECASE)),
        'word_count_planning': bool(re.search(r'word count|350.*?450|length.*?plan', text, re.IGNORECASE)),
        'fact_verification': bool(re.search(r'fact.*?check|verify|supported.*?data', text, re.IGNORECASE)),
        'quality_assessment': bool(re.search(r'quality.*?assess|professional.*?check', text, re.IGNORECASE)),
        'revision_notes': bool(re.search(r'revision|change|modify|improve', text, re.IGNORECASE)),
        'final_report_section': bool(re.search(r'FINAL REPORT', text, re.IGNORECASE))
    }
    
    # Count reflection-related words
    reflection_words = len(re.findall(r'\b(?:reflect|review|check|assess|consider|evaluate|analyze|verify|revision)\b', text.lower()))
    
    # General analysis
    analysis = {
        'label': label,
        'word_count': len(text.split()),
        'reflection_sections': reflection_sections,
        'reflection_words': reflection_words,
        'has_explicit_reflection': any(reflection_sections.values()),
        'reflection_percentage': sum(reflection_sections.values()) / len(reflection_sections) * 100
    }
    
    return analysis

def compare_results(refl_analysis, no_refl_analysis):
    """Compare the two approaches"""
    
    print(f"\n{'='*80}")
    print("📊 COMPARISON ANALYSIS")
    print('='*80)
    
    print(f"📋 REFLECTION SECTIONS FOUND:")
    print(f"{'Section':<25} {'With Reflection':<20} {'Without Reflection':<20}")
    print("-" * 65)
    
    for section in refl_analysis['reflection_sections']:
        refl_found = "✅ YES" if refl_analysis['reflection_sections'][section] else "❌ NO"
        no_refl_found = "✅ YES" if no_refl_analysis['reflection_sections'][section] else "❌ NO"
        print(f"{section.replace('_', ' ').title():<25} {refl_found:<20} {no_refl_found:<20}")
    
    print(f"\n📊 QUANTITATIVE COMPARISON:")
    print(f"{'Metric':<25} {'With Reflection':<20} {'Without Reflection':<20}")
    print("-" * 65)
    print(f"{'Word Count':<25} {refl_analysis['word_count']:<20} {no_refl_analysis['word_count']:<20}")
    print(f"{'Reflection Words':<25} {refl_analysis['reflection_words']:<20} {no_refl_analysis['reflection_words']:<20}")
    print(f"{'Reflection %':<25} {refl_analysis['reflection_percentage']:.1f}%{'':<15} {no_refl_analysis['reflection_percentage']:.1f}%")
    
    # Determine if reflection actually worked
    if refl_analysis['has_explicit_reflection'] and not no_refl_analysis['has_explicit_reflection']:
        print(f"\n✅ CONCLUSION: Reflection prompt IS working - model shows explicit reflection process")
    elif refl_analysis['reflection_words'] > no_refl_analysis['reflection_words'] * 1.5:
        print(f"\n🤔 CONCLUSION: Reflection prompt may be working - more reflection language used")
    else:
        print(f"\n❌ CONCLUSION: Reflection prompt NOT working effectively - no clear difference")

def main():
    print("=" * 80)
    print("🔍 SIMPLE REFLECTION TEST")
    print("=" * 80)
    
    # Test 1: With explicit reflection
    refl_prompt = create_visible_reflection_prompt()
    refl_result, refl_analysis = generate_and_analyze(refl_prompt, "WITH REFLECTION")
    
    # Test 2: Without reflection
    no_refl_prompt = create_no_reflection_prompt()
    no_refl_result, no_refl_analysis = generate_and_analyze(no_refl_prompt, "WITHOUT REFLECTION")
    
    # Compare results
    compare_results(refl_analysis, no_refl_analysis)
    
    # Save results
    with open("reflection_test_results.txt", "w") as f:
        f.write("REFLECTION TEST RESULTS\n")
        f.write("=" * 40 + "\n\n")
        f.write("WITH REFLECTION:\n")
        f.write("-" * 20 + "\n")
        f.write(refl_result)
        f.write("\n\n" + "=" * 40 + "\n\n")
        f.write("WITHOUT REFLECTION:\n")
        f.write("-" * 20 + "\n")
        f.write(no_refl_result)
    
    print(f"\n💾 Results saved to reflection_test_results.txt")

if __name__ == "__main__":
    main() 