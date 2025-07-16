#!/usr/bin/env python3
"""
🏀 Basketball Games Results Zipper
==================================
Zips the basketball_games folder for download from Google Colab
"""

import os
import zipfile
from datetime import datetime
import json

def zip_basketball_games():
    """Create a zip file of all basketball games analysis results"""
    
    # Configuration
    source_folder = "basketball_games"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"basketball_analysis_results_{timestamp}.zip"
    
    print("🏀 BASKETBALL GAMES RESULTS ZIPPER")
    print("=" * 50)
    print(f"📁 Source folder: {source_folder}")
    print(f"📦 Output zip: {zip_filename}")
    print()
    
    # Check if source folder exists
    if not os.path.exists(source_folder):
        print(f"❌ Error: {source_folder} folder not found!")
        return None
    
    # Create zip file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        total_files = 0
        total_size = 0
        
        # Walk through all files in the basketball_games folder
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, '.')
                
                # Add file to zip
                zipf.write(file_path, arc_path)
                
                # Track stats
                total_files += 1
                total_size += os.path.getsize(file_path)
                
                # Show progress
                if total_files % 10 == 0:
                    print(f"📄 Added {total_files} files...")
    
    # Final stats
    zip_size = os.path.getsize(zip_filename)
    compression_ratio = (1 - zip_size / total_size) * 100 if total_size > 0 else 0
    
    print(f"\n✅ ZIP CREATION COMPLETE!")
    print(f"📊 Files added: {total_files}")
    print(f"📏 Original size: {total_size / 1024 / 1024:.2f} MB")
    print(f"📦 Zip size: {zip_size / 1024 / 1024:.2f} MB")
    print(f"🗜️ Compression: {compression_ratio:.1f}%")
    print(f"📁 Output file: {zip_filename}")
    
    return zip_filename

def create_download_summary():
    """Create a summary of what's included in the download"""
    
    summary = {
        "analysis_info": {
            "timestamp": datetime.now().isoformat(),
            "total_games": 19,
            "prompt_types": ["reflection", "no_reflection", "dual_identity"],
            "reports_generated": 57,
            "attention_analysis_games": [0, 5, 10, 15, 18]
        },
        "folder_structure": {
            "basketball_games/": "Main results folder",
            "basketball_games/00-18/": "Individual game folders",
            "basketball_games/XX/results.txt": "Generated reports for each game",
            "basketball_games/XX/game_data.json": "Original game data"
        },
        "key_findings": {
            "reflection_backward_attention": 0.3758,
            "no_reflection_backward_attention": 0.3706,
            "dual_identity_backward_attention": 0.2889,
            "interpretation": "Reflection prompts don't increase backward attention"
        }
    }
    
    with open("download_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("📝 Created download_summary.json")
    return "download_summary.json"

def main():
    """Main function to zip everything"""
    
    # Create download summary
    summary_file = create_download_summary()
    
    # Zip the basketball games folder
    zip_file = zip_basketball_games()
    
    if zip_file:
        print(f"\n🎯 TO DOWNLOAD IN COLAB:")
        print(f"from google.colab import files")
        print(f"files.download('{zip_file}')")
        print(f"files.download('{summary_file}')")
        
        print(f"\n💡 OR USE THESE COMMANDS:")
        print(f"!ls -la {zip_file}")
        print(f"!ls -la {summary_file}")
        
        return zip_file, summary_file
    
    return None, None

if __name__ == "__main__":
    zip_file, summary_file = main() 