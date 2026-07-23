import os
from dotenv import load_dotenv
from google import genai

load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY", ""))

print("All available models:\n")
for model in client.models.list():
    print(f"Name: {model.name}")
    print(f"Supported actions: {model.supported_actions}")
    print("---")