#!/usr/bin/env python3
"""
🧠 Model Interpretability Strategies for Reflection Analysis
===========================================================
Beyond backward attention: Multiple approaches to understand what's happening 
in the model's "mind" when processing different prompt types.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
from scipy import stats
from datetime import datetime

# ================================================================================
# 🎯 STRATEGY 1: HIDDEN STATE ANALYSIS
# ================================================================================

class HiddenStateAnalyzer:
    """Analyze internal representations (hidden states) across different prompts"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_states = {}
        
    def extract_hidden_states(self, prompt, prompt_type, max_tokens=50):
        """Extract hidden states during generation"""
        print(f"🔍 Extracting hidden states for {prompt_type}...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        hidden_states_sequence = []
        current_ids = input_ids.clone()
        
        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(current_ids, output_hidden_states=True)
                
                # Store hidden states from all layers
                step_hidden_states = []
                for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
                    # Get the last token's hidden state
                    last_token_hidden = layer_hidden[0, -1, :].cpu().numpy()
                    step_hidden_states.append(last_token_hidden)
                
                hidden_states_sequence.append(step_hidden_states)
                
                # Get next token
                logits = outputs.logits[0, -1, :]
                next_token = torch.multinomial(torch.softmax(logits / 0.7, dim=-1), 1)
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        self.hidden_states[prompt_type] = hidden_states_sequence
        return hidden_states_sequence
    
    def analyze_representational_geometry(self):
        """Compare the geometric structure of representations"""
        print("📐 Analyzing representational geometry...")
        
        results = {}
        
        for prompt_type, states_sequence in self.hidden_states.items():
            if not states_sequence:
                continue
                
            # Focus on the last layer (most semantic)
            last_layer_states = [step[-1] for step in states_sequence]
            
            if len(last_layer_states) > 1:
                # Calculate pairwise similarities
                similarities = []
                for i in range(len(last_layer_states)-1):
                    sim = cosine_similarity([last_layer_states[i]], [last_layer_states[i+1]])[0][0]
                    similarities.append(sim)
                
                # Calculate dimensionality (effective rank)
                state_matrix = np.array(last_layer_states)
                U, s, Vt = np.linalg.svd(state_matrix)
                effective_rank = np.sum(s > 0.01 * s[0])  # Dimensions with >1% of max singular value
                
                results[prompt_type] = {
                    'avg_similarity': np.mean(similarities),
                    'similarity_std': np.std(similarities),
                    'effective_rank': effective_rank,
                    'representation_norm': np.mean([np.linalg.norm(state) for state in last_layer_states])
                }
        
        return results

# ================================================================================
# 🎯 STRATEGY 2: GRADIENT-BASED SALIENCY ANALYSIS
# ================================================================================

class SaliencyAnalyzer:
    """Analyze which input tokens are most important for the model's decisions"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def compute_gradient_saliency(self, prompt, target_text, prompt_type):
        """Compute gradient-based saliency maps"""
        print(f"🎯 Computing saliency for {prompt_type}...")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
        input_ids = inputs["input_ids"].to(self.model.device)
        input_ids.requires_grad_(True)
        
        # Get embeddings
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(inputs_embeds=embeddings, output_hidden_states=True)
        
        # Calculate loss with respect to target generation
        target_ids = self.tokenizer(target_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        target_ids = target_ids.to(self.model.device)
        
        if target_ids.shape[1] > 0:
            # Use the probability of the first target token as the objective
            target_logits = outputs.logits[0, -1, target_ids[0, 0]]
            target_logits.backward()
            
            # Compute saliency scores
            saliency_scores = torch.norm(embeddings.grad, dim=-1).squeeze().cpu().numpy()
            
            # Get token strings
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            
            return {
                'tokens': tokens,
                'saliency_scores': saliency_scores,
                'total_saliency': np.sum(saliency_scores)
            }
        
        return None

# ================================================================================
# 🎯 STRATEGY 3: PROBING CLASSIFIERS
# ================================================================================

class ProbingAnalyzer:
    """Train probes to understand what information is encoded in hidden states"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.probes = {}
        
    def extract_probe_data(self, prompts_and_labels):
        """Extract hidden states and create training data for probes"""
        print("🔬 Extracting data for probing analysis...")
        
        X = []  # Hidden states
        y = []  # Labels (prompt types)
        
        for prompt, label in prompts_and_labels:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
            input_ids = inputs["input_ids"].to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
                
                # Use the last layer, last token representation
                last_hidden = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
                X.append(last_hidden)
                y.append(label)
        
        return np.array(X), np.array(y)
    
    def train_reflection_probe(self, X, y):
        """Train a probe to classify reflection vs non-reflection prompts"""
        print("🎯 Training reflection detection probe...")
        
        # Simple logistic regression probe
        probe = LogisticRegression(random_state=42, max_iter=1000)
        probe.fit(X, y)
        
        # Evaluate
        accuracy = probe.score(X, y)
        
        # Get feature importance (weights)
        feature_importance = np.abs(probe.coef_[0])
        
        self.probes['reflection_classifier'] = {
            'model': probe,
            'accuracy': accuracy,
            'feature_importance': feature_importance
        }
        
        return accuracy, feature_importance

# ================================================================================
# 🎯 STRATEGY 4: INFORMATION FLOW ANALYSIS
# ================================================================================

class InformationFlowAnalyzer:
    """Analyze how information flows through the network layers"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def analyze_layer_wise_changes(self, prompt, prompt_type):
        """Track how representations change across layers"""
        print(f"🌊 Analyzing information flow for {prompt_type}...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            
            # Get hidden states from all layers
            layer_states = []
            for layer_hidden in outputs.hidden_states:
                # Use the last token's representation
                last_token_state = layer_hidden[0, -1, :].cpu().numpy()
                layer_states.append(last_token_state)
            
            # Calculate layer-to-layer changes
            layer_changes = []
            for i in range(1, len(layer_states)):
                # Cosine similarity between consecutive layers
                similarity = cosine_similarity([layer_states[i-1]], [layer_states[i]])[0][0]
                layer_changes.append(1 - similarity)  # Convert to "change" measure
            
            # Calculate cumulative information
            cumulative_norms = [np.linalg.norm(state) for state in layer_states]
            
            return {
                'layer_changes': layer_changes,
                'cumulative_norms': cumulative_norms,
                'final_representation': layer_states[-1]
            }

# ================================================================================
# 🎯 STRATEGY 5: CONFIDENCE AND UNCERTAINTY ANALYSIS
# ================================================================================

class ConfidenceAnalyzer:
    """Analyze model confidence and uncertainty patterns"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def analyze_prediction_confidence(self, prompt, prompt_type, max_tokens=50):
        """Analyze how confident the model is in its predictions"""
        print(f"📊 Analyzing prediction confidence for {prompt_type}...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        confidence_scores = []
        entropy_scores = []
        top_k_probs = []
        
        current_ids = input_ids.clone()
        
        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(current_ids)
                logits = outputs.logits[0, -1, :]
                
                # Calculate confidence metrics
                probs = torch.softmax(logits, dim=-1)
                
                # Max probability (confidence)
                max_prob = torch.max(probs).item()
                confidence_scores.append(max_prob)
                
                # Entropy (uncertainty)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
                entropy_scores.append(entropy)
                
                # Top-k probability distribution
                top_k_probs.append(torch.topk(probs, k=5).values.cpu().numpy())
                
                # Get next token
                next_token = torch.multinomial(probs, 1)
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return {
            'confidence_scores': confidence_scores,
            'entropy_scores': entropy_scores,
            'avg_confidence': np.mean(confidence_scores),
            'avg_entropy': np.mean(entropy_scores),
            'confidence_std': np.std(confidence_scores)
        }

# ================================================================================
# 🎯 STRATEGY 6: SEMANTIC CONCEPT ACTIVATION
# ================================================================================

class ConceptActivationAnalyzer:
    """Analyze which semantic concepts are activated by different prompts"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.concept_vectors = {}
        
    def create_concept_vectors(self, concept_examples):
        """Create vectors representing different concepts"""
        print("🧠 Creating concept vectors...")
        
        for concept, examples in concept_examples.items():
            concept_activations = []
            
            for example in examples:
                inputs = self.tokenizer(example, return_tensors="pt", truncation=True, max_length=500)
                input_ids = inputs["input_ids"].to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids, output_hidden_states=True)
                    # Use last layer, last token
                    activation = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
                    concept_activations.append(activation)
            
            # Average across examples to get concept vector
            self.concept_vectors[concept] = np.mean(concept_activations, axis=0)
        
        return self.concept_vectors
    
    def measure_concept_activation(self, prompt, prompt_type):
        """Measure how much each concept is activated by the prompt"""
        print(f"🎯 Measuring concept activation for {prompt_type}...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
        input_ids = inputs["input_ids"].to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            prompt_activation = outputs.hidden_states[-1][0, -1, :].cpu().numpy()
        
        concept_similarities = {}
        for concept, concept_vector in self.concept_vectors.items():
            similarity = cosine_similarity([prompt_activation], [concept_vector])[0][0]
            concept_similarities[concept] = similarity
        
        return concept_similarities

# ================================================================================
# 🎯 COMPREHENSIVE ANALYSIS PIPELINE
# ================================================================================

class ComprehensiveInterpretabilityAnalyzer:
    """Combine all interpretability strategies for a complete analysis"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize all analyzers
        self.hidden_state_analyzer = HiddenStateAnalyzer(model, tokenizer)
        self.saliency_analyzer = SaliencyAnalyzer(model, tokenizer)
        self.probing_analyzer = ProbingAnalyzer(model, tokenizer)
        self.info_flow_analyzer = InformationFlowAnalyzer(model, tokenizer)
        self.confidence_analyzer = ConfidenceAnalyzer(model, tokenizer)
        self.concept_analyzer = ConceptActivationAnalyzer(model, tokenizer)
        
    def run_comprehensive_analysis(self, prompts_dict):
        """Run all analyses on the given prompts"""
        print("🚀 RUNNING COMPREHENSIVE INTERPRETABILITY ANALYSIS")
        print("=" * 80)
        
        results = {}
        
        # 1. Hidden State Analysis
        print("\n1️⃣ HIDDEN STATE ANALYSIS")
        print("-" * 40)
        for prompt_type, prompt in prompts_dict.items():
            self.hidden_state_analyzer.extract_hidden_states(prompt, prompt_type)
        
        geometry_results = self.hidden_state_analyzer.analyze_representational_geometry()
        results['hidden_states'] = geometry_results
        
        # 2. Information Flow Analysis
        print("\n2️⃣ INFORMATION FLOW ANALYSIS")
        print("-" * 40)
        flow_results = {}
        for prompt_type, prompt in prompts_dict.items():
            flow_results[prompt_type] = self.info_flow_analyzer.analyze_layer_wise_changes(prompt, prompt_type)
        results['information_flow'] = flow_results
        
        # 3. Confidence Analysis
        print("\n3️⃣ CONFIDENCE ANALYSIS")
        print("-" * 40)
        confidence_results = {}
        for prompt_type, prompt in prompts_dict.items():
            confidence_results[prompt_type] = self.confidence_analyzer.analyze_prediction_confidence(prompt, prompt_type)
        results['confidence'] = confidence_results
        
        # 4. Concept Activation Analysis
        print("\n4️⃣ CONCEPT ACTIVATION ANALYSIS")
        print("-" * 40)
        
        # Define concepts related to reflection and analysis
        concept_examples = {
            'reflection': [
                'Let me think about this carefully',
                'I need to reflect on my approach',
                'Let me review what I said',
                'I should check my reasoning'
            ],
            'analysis': [
                'Analyzing the data shows',
                'Breaking down the problem',
                'Examining the evidence',
                'Systematic investigation reveals'
            ],
            'certainty': [
                'I am confident that',
                'This is definitely true',
                'Without doubt',
                'Clearly the answer is'
            ],
            'uncertainty': [
                'I am not sure about',
                'This might be',
                'Perhaps it could be',
                'It is possible that'
            ]
        }
        
        self.concept_analyzer.create_concept_vectors(concept_examples)
        
        concept_results = {}
        for prompt_type, prompt in prompts_dict.items():
            concept_results[prompt_type] = self.concept_analyzer.measure_concept_activation(prompt, prompt_type)
        results['concepts'] = concept_results
        
        return results
    
    def create_interpretability_report(self, results):
        """Create a comprehensive interpretability report"""
        print("\n📊 INTERPRETABILITY ANALYSIS REPORT")
        print("=" * 80)
        
        # Hidden States Analysis
        print("\n🧠 HIDDEN STATE ANALYSIS:")
        for prompt_type, metrics in results['hidden_states'].items():
            print(f"  {prompt_type}:")
            print(f"    Average similarity: {metrics['avg_similarity']:.4f}")
            print(f"    Effective rank: {metrics['effective_rank']}")
            print(f"    Representation norm: {metrics['representation_norm']:.4f}")
        
        # Information Flow Analysis
        print("\n🌊 INFORMATION FLOW ANALYSIS:")
        for prompt_type, flow_data in results['information_flow'].items():
            avg_change = np.mean(flow_data['layer_changes'])
            print(f"  {prompt_type}: Avg layer change = {avg_change:.4f}")
        
        # Confidence Analysis
        print("\n📊 CONFIDENCE ANALYSIS:")
        for prompt_type, conf_data in results['confidence'].items():
            print(f"  {prompt_type}:")
            print(f"    Average confidence: {conf_data['avg_confidence']:.4f}")
            print(f"    Average entropy: {conf_data['avg_entropy']:.4f}")
        
        # Concept Activation Analysis
        print("\n🎯 CONCEPT ACTIVATION ANALYSIS:")
        for prompt_type, concepts in results['concepts'].items():
            print(f"  {prompt_type}:")
            for concept, activation in concepts.items():
                print(f"    {concept}: {activation:.4f}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interpretability_analysis_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            return obj
        
        with open(filename, 'w') as f:
            json.dump(convert_for_json(results), f, indent=2)
        
        print(f"\n💾 Results saved to {filename}")
        return filename

# ================================================================================
# 🎯 MAIN EXECUTION EXAMPLE
# ================================================================================

def main():
    """Example of running comprehensive interpretability analysis"""
    print("🧠 COMPREHENSIVE MODEL INTERPRETABILITY ANALYSIS")
    print("=" * 80)
    
    # This is a template - you would replace with your actual model and tokenizer
    print("⚠️  This is a template script - replace with your actual model!")
    print("🔧 To use this, you need to:")
    print("   1. Load your model and tokenizer")
    print("   2. Define your prompts")
    print("   3. Run the analysis")
    
    # Example prompts (replace with your actual prompts)
    example_prompts = {
        'reflection': """Write a basketball report, but FIRST reflect on your approach:
## REFLECTION PROCESS:
- **Data Review**: What information do I have?
- **Quality Check**: Is my writing professional?
Generate report:""",
        
        'no_reflection': """Write a basketball report based on the game data.
Generate report:""",
        
        'dual_identity': """You have two internal voices:
JOURNALIST: Writes professionally
FAN: Adds excitement
Generate report:"""
    }
    
    print("\n📋 ANALYSIS STRATEGIES AVAILABLE:")
    print("1️⃣ Hidden State Analysis - Representational geometry")
    print("2️⃣ Gradient Saliency - Which tokens matter most")
    print("3️⃣ Probing Classifiers - What info is encoded")
    print("4️⃣ Information Flow - How data moves through layers")
    print("5️⃣ Confidence Analysis - Model uncertainty patterns")
    print("6️⃣ Concept Activation - Which concepts are triggered")
    
    print("\n🎯 TO USE THIS ANALYSIS:")
    print("analyzer = ComprehensiveInterpretabilityAnalyzer(model, tokenizer)")
    print("results = analyzer.run_comprehensive_analysis(prompts_dict)")
    print("analyzer.create_interpretability_report(results)")

if __name__ == "__main__":
    main() 