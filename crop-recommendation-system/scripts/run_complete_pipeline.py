"""
Complete ML Pipeline Runner
Run this script to execute the entire crop recommendation system pipeline
"""

import subprocess
import sys
import os

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 characters
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        return False
    except FileNotFoundError:
        print(f"❌ SCRIPT NOT FOUND: {script_name}")
        return False

def main():
    """Run the complete ML pipeline"""
    print("🌾 CROP RECOMMENDATION SYSTEM - COMPLETE PIPELINE")
    print("="*60)
    
    # Define pipeline steps
    pipeline_steps = [
        ("data_analysis.py", "Data Loading and Analysis"),
        ("feature_engineering.py", "Feature Engineering"),
        ("ml_models.py", "ML Model Training"),
        ("ensemble_models.py", "Ensemble Model Training"),
        ("model_evaluation.py", "Model Evaluation"),
        ("model_interpretability.py", "Model Interpretability Analysis")
    ]
    
    # Track success/failure
    results = {}
    
    # Run each step
    for script, description in pipeline_steps:
        success = run_script(script, description)
        results[script] = success
        
        if not success:
            print(f"\n⚠️  Pipeline stopped at {script}")
            print("Please fix the error and run again.")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    
    for script, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{script:<30} {status}")
    
    if all(results.values()):
        print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("You can now run the Streamlit app:")
        print("streamlit run streamlit_app.py")
    else:
        print(f"\n⚠️  PIPELINE INCOMPLETE")
        print("Please fix the errors and run again.")

if __name__ == "__main__":
    main()
