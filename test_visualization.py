#!/usr/bin/env python3
"""
Test script to verify the visualization works with the correct data structure
"""

import json
import numpy as np

# Create sample data that matches your actual JSON structure
sample_data = {
    "0": {
        "reflection": {
            "avg_backward_ratio": 0.3711,
            "reflection_spikes": [],
            "total_steps": 50
        },
        "no_reflection": {
            "avg_backward_ratio": 0.3820,
            "reflection_spikes": [],
            "total_steps": 50
        },
        "dual_identity": {
            "avg_backward_ratio": 0.2951,
            "reflection_spikes": [],
            "total_steps": 50
        }
    },
    "5": {
        "reflection": {
            "avg_backward_ratio": 0.3659,
            "reflection_spikes": [],
            "total_steps": 50
        },
        "no_reflection": {
            "avg_backward_ratio": 0.3659,
            "reflection_spikes": [],
            "total_steps": 50
        },
        "dual_identity": {
            "avg_backward_ratio": 0.3165,
            "reflection_spikes": [],
            "total_steps": 50
        }
    },
    "10": {
        "reflection": {
            "avg_backward_ratio": 0.3602,
            "reflection_spikes": [],
            "total_steps": 50
        },
        "no_reflection": {
            "avg_backward_ratio": 0.3706,
            "reflection_spikes": [],
            "total_steps": 50
        },
        "dual_identity": {
            "avg_backward_ratio": 0.2573,
            "reflection_spikes": [],
            "total_steps": 50
        }
    },
    "15": {
        "reflection": {
            "avg_backward_ratio": 0.3700,
            "reflection_spikes": [],
            "total_steps": 50
        },
        "no_reflection": {
            "avg_backward_ratio": 0.3705,
            "reflection_spikes": [],
            "total_steps": 50
        },
        "dual_identity": {
            "avg_backward_ratio": 0.2811,
            "reflection_spikes": [],
            "total_steps": 50
        }
    },
    "18": {
        "reflection": {
            "avg_backward_ratio": 0.4116,
            "reflection_spikes": [],
            "total_steps": 50
        },
        "no_reflection": {
            "avg_backward_ratio": 0.3640,
            "reflection_spikes": [],
            "total_steps": 50
        },
        "dual_identity": {
            "avg_backward_ratio": 0.2943,
            "reflection_spikes": [],
            "total_steps": 50
        }
    }
}

# Save sample data as JSON
with open("attention_analysis_results.json", "w") as f:
    json.dump(sample_data, f, indent=2)

print("✅ Created sample attention_analysis_results.json with correct structure")

# Test the visualization
print("🧪 Testing visualization script...")

try:
    from visualize_backward_attention import main
    main()
    print("✅ Visualization script works!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n🎯 If successful, you should see 5 publication-ready plots!")
print("📊 The plots show your key finding: reflection doesn't increase backward attention") 