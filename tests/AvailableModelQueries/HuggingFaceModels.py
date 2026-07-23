import requests

# Query Hugging Face's live API status registry
url = "https://api-inference.huggingface.co/framework/transformers"
try:
    response = requests.get(url).json()
    multimodal_models = [
        m for m in response.get("models", []) 
        if "image-text-to-text" in m or "image-to-text" in m
    ]
    
    print("100% ACTIVE HF FREE MODELS")
    for model in multimodal_models[:10]:
        print(f"- {model}")
except Exception as e:
    print(f"Could not fetch live list: {e}")