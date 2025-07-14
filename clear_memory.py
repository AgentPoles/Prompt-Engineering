# ===================================================================
# 🧹 MEMORY CLEARING CODE BLOCK FOR JUPYTER/COLAB
# ===================================================================
# Run this cell before running the backward attention analysis
# to clear GPU memory and free up space

import gc
import torch
import psutil
import os

def clear_all_memory():
    """Comprehensive memory clearing function"""
    
    print("🧹 Clearing memory...")
    
    # 1. Clear GPU memory
    if torch.cuda.is_available():
        print("  🔥 Clearing GPU memory...")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Show GPU memory before and after
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"      Memory reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    
    # 2. Delete common variables that might be in memory
    variables_to_delete = [
        'model', 'tokenizer', 'analyzer', 'outputs', 'inputs', 
        'attention_data', 'results', 'comparison', 'refl_attention',
        'no_refl_attention', 'reflection_analysis', 'normal_analysis'
    ]
    
    print("  🗑️ Deleting variables...")
    deleted_count = 0
    for var_name in variables_to_delete:
        if var_name in globals():
            del globals()[var_name]
            deleted_count += 1
            print(f"    Deleted: {var_name}")
    
    if deleted_count == 0:
        print("    No variables to delete")
    
    # 3. Force garbage collection
    print("  ♻️ Running garbage collection...")
    collected = gc.collect()
    print(f"    Collected {collected} objects")
    
    # 4. Clear GPU cache again after garbage collection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 5. Show memory status
    print("  📊 Memory status after clearing:")
    
    # System memory
    memory = psutil.virtual_memory()
    print(f"    System RAM: {memory.used / 1024**3:.2f} GB used / {memory.total / 1024**3:.2f} GB total")
    
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"    GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    print("✅ Memory clearing complete!")

# Run the memory clearing
clear_all_memory()

# Optional: Set environment variables for better memory management
print("\n🔧 Setting memory optimization flags...")
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings

print("🚀 Ready to run backward attention analysis!")
print("You can now run the backward_attention_analysis.py script") 