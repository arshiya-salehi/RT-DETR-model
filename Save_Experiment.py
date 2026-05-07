import os
import shutil
import datetime
import json
import re

def extract_hyperparams(filepath):
    """Reads Train.py and extracts ALL_CAPS configuration variables."""
    hyperparams = {}
    if not os.path.exists(filepath):
        return hyperparams
        
    with open(filepath, 'r') as f:
        for line in f:
            # Match lines like "BATCH_SIZE = 16 # comment"
            match = re.match(r'^([A-Z_0-9]+)\s*=\s*(.*)', line.strip())
            if match:
                key, val = match.groups()
                # remove inline comments
                val = val.split('#')[0].strip()
                hyperparams[key] = val
    return hyperparams

def main():
    # Create timestamped experiment directory
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Optional: allow user to pass a custom name
    import sys
    custom_name = ""
    if len(sys.argv) > 1:
        custom_name = "_" + sys.argv[1].replace(" ", "_")
        
    exp_dir = f"experiments/run_{timestamp}{custom_name}"
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"Archiving current results to '{exp_dir}' ...")

    # 1. Copy validation results (the mAP scores and bounding box images)
    if os.path.exists("output/val_results"):
        shutil.copytree("output/val_results", f"{exp_dir}/val_results")
        print("✅ Copied output/val_results/")
        
    # 1.5. Copy test results
    if os.path.exists("output/test_results"):
        shutil.copytree("output/test_results", f"{exp_dir}/test_results")
        print("✅ Copied output/test_results/")
    
    # 2. Copy training history
    if os.path.exists("output/training_history.json"):
        shutil.copy("output/training_history.json", exp_dir)
        print("✅ Copied output/training_history.json")
    
    # 3. Copy logs
    if os.path.exists("logs/train.log"):
        shutil.copy("logs/train.log", exp_dir)
        print("✅ Copied logs/train.log")
        
    # 4. Extract and save hyperparameters from Train.py
    hparams = extract_hyperparams("Train.py")
    if hparams:
        with open(f"{exp_dir}/hyperparameters.json", "w") as f:
            json.dump(hparams, f, indent=4)
        print("✅ Extracted hyperparameters from Train.py")

    # Add a .gitignore inside experiments/ just in case model weights end up here
    with open("experiments/.gitignore", "w") as f:
        f.write("*.pth\n")

    print(f"\n🎉 Archive complete! Your experiment is saved in: {exp_dir}")
    print("\nYou can now safely commit these logs to GitHub:")
    print(f"  git add {exp_dir}")
    print(f"  git commit -m \"Archive training run {timestamp}\"")
    print("  git push")

if __name__ == "__main__":
    main()
