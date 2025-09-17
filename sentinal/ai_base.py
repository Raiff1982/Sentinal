
from transformers import pipeline

class AIBase:
    def __init__(self, model_name="distilbert-base-uncased", llm_name="distilgpt2"):
        self.model_name = model_name
        self.llm_name = llm_name
        self.classifier = pipeline("sentiment-analysis", model=model_name)
        self.llm = pipeline("text-generation", model=llm_name, max_length=100)

    def analyze(self, text):
        return self.classifier(text)

    def chat(self, prompt):
        response = self.llm(prompt)[0]["generated_text"]
        # Remove prompt from response for cleaner output
        return response[len(prompt):].strip() if response.startswith(prompt) else response.strip()
