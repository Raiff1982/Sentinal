import os
import subprocess
import shutil

def retrain_and_deploy():
    # Retrain both LLM and cocoon models
    print("Retraining fine-tuned LLM...")
    subprocess.run([os.path.join(".venv", "Scripts", "python.exe"), "webui/train_llm.py", "t"])
    print("Retraining Codette cocoon model...")
    subprocess.run([os.path.join(".venv", "Scripts", "python.exe"), "webui/train_codette_cocoon.py"])
    # Deploy models (move to webui/ if needed)
    for model_dir in ["finetuned_llm", "codette_cocoon_model"]:
        src = os.path.join(model_dir)
        dst = os.path.join("webui", model_dir)
        if os.path.isdir(src):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"Deployed {model_dir} to webui/")
    print("Retraining and deployment complete.")

if __name__ == "__main__":
    retrain_and_deploy()
