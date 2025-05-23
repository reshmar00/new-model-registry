import os
import argparse
import subprocess
import re

def get_next_model_version(model_name, output_dir="model_outputs"):
    """Determine the next model version based on existing files."""
    version_pattern = re.compile(rf"{re.escape(model_name)}_v(\d+)\.(\d+)\.pkl$")
    versions = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for f in os.listdir(output_dir):
        match = version_pattern.match(f)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            versions.append((major, minor))

    if not versions:
        return f"{model_name}_v0.0.pkl"

    versions.sort()
    last_major, last_minor = versions[-1]
    next_version = f"{model_name}_v{last_major}.{last_minor + 1}.pkl"
    return next_version

def run_dvc_pipeline():
    """Run the DVC pipeline to update the outputs."""
    print("🚀 Running DVC pipeline automation...")
    subprocess.run(["dvc", "repro"], check=True)
    subprocess.run(["dvc", "push"], check=True)
    print("✅ DVC pipeline execution completed!")

def main():
    parser = argparse.ArgumentParser(description="Run the DVC pipeline and auto-version output.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file (e.g., models/model.py)")
    parser.add_argument("--data", type=str, required=True, help="Path to the data file (e.g., data/diabetes.csv)")
    args = parser.parse_args()

    model_name = os.path.splitext(os.path.basename(args.model))[0]
    print(f"Using model file: {args.model}")
    print(f"Using data file: {args.data}")

    # Set environment variables for DVC or other scripts
    output_model_name = get_next_model_version(model_name)
    os.environ["MODEL_NAME"] = model_name
    os.environ["OUTPUT_MODEL"] = output_model_name
    print(f"📦 Output model will be: {output_model_name}")

    # Run the pipeline
    run_dvc_pipeline()

if __name__ == "__main__":
    main()