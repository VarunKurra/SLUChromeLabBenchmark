# Models: 
# moonshotai/Kimi-K2.6 
# Qwen/Qwen3-VL-235B-A22B-Instruct
# Qwen/Qwen3.6-27B 
# Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled 
# DavidAU/Qwen3.5-27B-Claude-4.6-OS-INSTRUCT 

# PROVIDER DIFFICULTIES
# microsoft/Phi-3.5-vision-instruct

import os
import base64
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

HF_TOKEN = os.getenv("HF_TOKEN", "")

model_id = "moonshotai/Kimi-K2.6"

print(f"Analyzing cloud architecture for: {model_id}...")

model_id_lower = model_id.lower()

if "qwen3-vl-235b" in model_id_lower or "glm-4.6v" in model_id_lower:
    provider_host = "novita"
elif "kimi" in model_id_lower:
    provider_host = "fireworks-ai"
else:
    provider_host = "featherless-ai"

print(f"-> Routing Traffic Via Provider: {provider_host}")

client = InferenceClient(
    provider=provider_host,
    api_key=HF_TOKEN
)

extra_body_params = {}
if any(x in model_id_lower for x in ["qwen", "reasoning", "phi"]):
    print("-> Specialized model family detected. Adjusting parameters for direct output.")
    extra_body_params = {
        "chat_template_kwargs": {"enable_thinking": False}
    }
else:
    print("-> Standard vision stream detected. Processing rapid frame response context.")

local_image_path = "/Users/varun/Downloads/Test_Graph.webp"
user_prompt = "Tell me about this graph. Keep in mind I am blind."

print("Encoding local webp graph into base64 text payload...")
try:
    if not os.path.exists(local_image_path):
        raise FileNotFoundError(f"File missing at {local_image_path}")
        
    with open(local_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
    print(f"Sending visual request straight to the cloud clusters...")
    
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/webp;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=4000,
        extra_body=extra_body_params if extra_body_params else None
    )
    
    print(f"\n--- Success! [{model_id}] Accessibility Breakdown ---")
    print(response.choices[0].message.content)

except Exception as e:
    print(f"\n❌ Execution Error: {e}")