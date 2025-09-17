
from transformers import pipeline



import os


class AIBase:
    def __init__(self, model_names=None, llm_names=None):
        # Use fine-tuned model for sentiment and Codette cocoon model for chat if available
        finetuned_path = os.path.join(os.path.dirname(__file__), "..", "webui", "finetuned_llm")
        cocoon_path = os.path.join(os.path.dirname(__file__), "..", "webui", "codette_cocoon_model")
        self.model_names = model_names or [finetuned_path if os.path.isdir(finetuned_path) else "distilbert-base-uncased", "roberta-base"]
        self.llm_names = llm_names or ([cocoon_path] if os.path.isdir(cocoon_path) else ["distilgpt2", "gpt2"])
        self.classifiers = [pipeline("sentiment-analysis", model=m) for m in self.model_names]
        self.llms = [pipeline("text-generation", model=l, max_length=100) for l in self.llm_names]

    def analyze(self, text):
        # Ensemble: collect results from all classifiers
        results = [clf(text)[0] for clf in self.classifiers]
        # Majority vote or average score
        labels = [r['label'] for r in results]
        scores = [r['score'] for r in results]
        majority_label = max(set(labels), key=labels.count)
        avg_score = sum(scores) / len(scores)
        return [{"label": majority_label, "score": avg_score, "details": results}]

    def chat(self, prompt):
        # Use first LLM for now, can be extended to ensemble
        responses = [llm(prompt)[0]["generated_text"] for llm in self.llms]
        # Return all responses for comparison
        cleaned = []
        for i, resp in enumerate(responses):
            cleaned.append(resp[len(prompt):].strip() if resp.startswith(prompt) else resp.strip())
        return cleaned[0] if cleaned else ""
