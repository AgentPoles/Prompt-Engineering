#!/usr/bin/env python3
"""
Basketball Report Reflection Analyzer
Visualizes what the model does with and without reflection
"""

import os, json, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================================================
# MODEL LOADING
# ============================================================================

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
print(f"Loading {MODEL_ID} for reflection analysis...")

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
    MODEL_ID = "microsoft/Phi-3-medium-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

# ============================================================================
# ENHANCED REFLECTION PROMPT (Makes reflection visible)
# ============================================================================

def create_visible_reflection_prompt(base_prompt, game_info):
    """Create a reflection prompt that shows the internal process"""
    
    # Modify the reflection section to be visible
    visible_reflection_prompt = base_prompt.replace(
        "Do not print any of your internal reflection or checks — only produce the final polished report.",
        """SHOW your internal reflection process by organizing your response as follows:

## REFLECTION PROCESS (Show this section):
1. **Data Analysis**: What specific data is available and what's missing?
2. **Compliance Check**: Does my draft meet all requirements (350-450 words, proper structure, factual accuracy)?
3. **Quality Assessment**: Are there any editorial phrases not supported by data?
4. **Revision Notes**: What changes did I make after reflection?

## FINAL REPORT (After reflection):
[Your polished report here]

This way we can see if reflection actually occurred."""
    )
    
    full_prompt = f"{visible_reflection_prompt.strip()}\n\n{game_info}"
    
    try:
        messages = [{"role": "user", "content": full_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        return f"<|user|>\n{full_prompt}\n<|assistant|>\n"

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_text_patterns(text, label):
    """Analyze text patterns to understand model behavior"""
    
    analysis = {
        'label': label,
        'word_count': len(text.split()),
        'sentence_count': len(re.findall(r'[.!?]+', text)),
        'avg_sentence_length': len(text.split()) / max(len(re.findall(r'[.!?]+', text)), 1),
        'question_marks': text.count('?'),
        'exclamation_marks': text.count('!'),
        'numbers': len(re.findall(r'\d+', text)),
        'percentages': len(re.findall(r'\d+%', text)),
        'sports_terms': len(re.findall(r'\b(?:points?|rebounds?|assists?|steals?|blocks?|shots?|field goals?|turnovers?)\b', text.lower())),
        'reflection_indicators': len(re.findall(r'\b(?:reflect|review|check|assess|consider|evaluate|analyze)\b', text.lower())),
        'revision_indicators': len(re.findall(r'\b(?:revise|edit|change|modify|improve|correct)\b', text.lower())),
        'certainty_words': len(re.findall(r'\b(?:clearly|obviously|definitely|certainly|undoubtedly)\b', text.lower())),
        'hedge_words': len(re.findall(r'\b(?:perhaps|maybe|likely|appears?|seems?|might|could)\b', text.lower())),
        'first_person': len(re.findall(r'\b(?:I|my|me|myself)\b', text.lower())),
        'unique_words': len(set(text.lower().split())),
        'repetition_score': 1 - (len(set(text.lower().split())) / max(len(text.split()), 1))
    }
    
    return analysis

def compare_generation_behavior(prompt1, prompt2, game_info, runs=3):
    """Compare generation behavior across multiple runs"""
    
    results = {
        'with_reflection': [],
        'without_reflection': []
    }
    
    print("🔍 Analyzing generation patterns...")
    
    for i in range(runs):
        print(f"Run {i+1}/{runs}")
        
        # With reflection
        refl_prompt = create_visible_reflection_prompt(prompt1, game_info)
        refl_report = generate_report_with_stats(refl_prompt, f"reflection_run_{i+1}")
        results['with_reflection'].append(refl_report)
        
        # Without reflection
        no_refl_prompt = create_standard_prompt(prompt2, game_info)
        no_refl_report = generate_report_with_stats(no_refl_prompt, f"no_reflection_run_{i+1}")
        results['without_reflection'].append(no_refl_report)
    
    return results

def generate_report_with_stats(prompt, label):
    """Generate report and collect statistics"""
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    
    input_length = inputs['input_ids'].shape[1]
    
    # Generate with more detailed tracking
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Decode result
    result = tokenizer.decode(outputs.sequences[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    # Analyze the output
    analysis = analyze_text_patterns(result.strip(), label)
    analysis['input_tokens'] = input_length
    analysis['output_tokens'] = len(outputs.sequences[0]) - input_length
    analysis['text'] = result.strip()
    
    return analysis

def create_standard_prompt(template, game_info):
    """Create standard prompt without reflection"""
    
    full_prompt = f"{template.strip()}\n\n{game_info}"
    
    try:
        messages = [{"role": "user", "content": full_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        return f"<|user|>\n{full_prompt}\n<|assistant|>\n"

def visualize_comparison(results):
    """Create visualizations comparing reflection vs no reflection"""
    
    # Extract metrics for comparison
    metrics = ['word_count', 'sentence_count', 'avg_sentence_length', 'sports_terms', 
               'reflection_indicators', 'revision_indicators', 'certainty_words', 
               'hedge_words', 'first_person', 'repetition_score']
    
    with_refl = results['with_reflection']
    without_refl = results['without_reflection']
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics[:6]):
        if i < len(axes):
            refl_values = [r[metric] for r in with_refl]
            no_refl_values = [r[metric] for r in without_refl]
            
            axes[i].bar(['With Reflection', 'Without Reflection'], 
                       [sum(refl_values)/len(refl_values), sum(no_refl_values)/len(no_refl_values)],
                       color=['lightblue', 'lightcoral'])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Average Count')
    
    plt.tight_layout()
    plt.savefig('reflection_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed report
    print("\n" + "="*80)
    print("📊 REFLECTION ANALYSIS REPORT")
    print("="*80)
    
    for metric in metrics:
        refl_avg = sum(r[metric] for r in with_refl) / len(with_refl)
        no_refl_avg = sum(r[metric] for r in without_refl) / len(without_refl)
        diff = refl_avg - no_refl_avg
        
        print(f"{metric.replace('_', ' ').title():<20}: "
              f"Reflection: {refl_avg:.2f}, No Reflection: {no_refl_avg:.2f}, "
              f"Difference: {diff:+.2f}")

def extract_reflection_sections(text):
    """Extract and analyze reflection sections if they exist"""
    
    reflection_patterns = [
        r'## REFLECTION PROCESS.*?## FINAL REPORT',
        r'REFLECTION:.*?REPORT:',
        r'Data Analysis:.*?(?=## FINAL REPORT|$)',
        r'Compliance Check:.*?(?=## FINAL REPORT|$)',
        r'Quality Assessment:.*?(?=## FINAL REPORT|$)',
        r'Revision Notes:.*?(?=## FINAL REPORT|$)'
    ]
    
    reflections = {}
    for pattern in reflection_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            reflections[pattern] = matches
    
    return reflections

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("🔍 BASKETBALL REPORT REFLECTION ANALYZER")
    print("=" * 80)
    
    # Load game data
    with open("data.json", "r") as f:
        game_data = json.load(f)
    
    # Extract game info (using your existing function)
    # ... (include the extract_game_info_fixed function from previous script)
    
    # Load prompts
    with open("prompt_with_reflection.txt", "r") as f:
        prompt_reflect = f.read()
    
    with open("prompt_with_no_reflection.txt", "r") as f:
        prompt_noreflect = f.read()
    
    # Simplified game info for testing
    game_info = """GAME SUMMARY:
Date: January 29, 2018
Location: Philips Arena, Atlanta
Teams: Hawks (15-35) vs Timberwolves (32-21)
Final Score: Hawks 101 - Timberwolves 96
Winner: Hawks"""
    
    # Run comparison analysis
    print("🔍 Running reflection analysis...")
    results = compare_generation_behavior(prompt_reflect, prompt_noreflect, game_info, runs=3)
    
    # Show individual results
    print("\n" + "="*80)
    print("📝 SAMPLE OUTPUTS WITH REFLECTION:")
    print("="*80)
    print(results['with_reflection'][0]['text'])
    
    print("\n" + "="*80)
    print("📝 SAMPLE OUTPUTS WITHOUT REFLECTION:")
    print("="*80)
    print(results['without_reflection'][0]['text'])
    
    # Analyze reflection sections
    print("\n" + "="*80)
    print("🔍 REFLECTION SECTION ANALYSIS:")
    print("="*80)
    
    for result in results['with_reflection']:
        reflections = extract_reflection_sections(result['text'])
        if reflections:
            print("✅ Reflection sections found:")
            for pattern, matches in reflections.items():
                print(f"- Pattern: {pattern[:50]}...")
                for match in matches[:1]:  # Show first match
                    print(f"  Content: {match[:200]}...")
        else:
            print("❌ No explicit reflection sections found")
    
    # Create visualizations
    visualize_comparison(results)
    
    # Save detailed analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"reflection_analysis_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Analysis complete! Results saved to reflection_analysis_{timestamp}.json")

if __name__ == "__main__":
    main() 