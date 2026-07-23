# Models:
# nex-agi/nex-n2-pro:free

# moonshotai/kimi-k2.6:free - Full Version Obtained through NVIDIA
# nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free - Full Version Obtained through NVIDIA

import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "")
)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_base64 = encode_image("/Users/varun/Downloads/Test_Graph.webp")

image_data_url = f"data:image/webp;base64,{image_base64}"

models = [
    "nex-agi/nex-n2-pro:free"
]

for model in models:
    print("\n" + "="*80)
    print("MODEL:", model)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Tell me about this graph. Keep in mind I am blind."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ]
    )
    print(response.choices[0].message.content)