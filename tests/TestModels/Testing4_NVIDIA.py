# Models:
# microsoft/phi-4-multimodal-instruct
# nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
# nvidia/nemotron-nano-12b-v2-vl
# mistralai/mistral-large-3-675b-instruct-2512
# mistralai/ministral-14b-instruct-2512
# meta/llama-4-maverick-17b-128e-instruct

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY", "")
)

image_url = "https://www.image2url.com/r2/default/images/1780602309583-a0c99451-72f0-4294-babb-7bc69cb09002.webp"

detailed_prompt = "Tell me about this graph. Keep in mind I am blind."

start_time = time.perf_counter()
response = client.chat.completions.create(
    model="meta/llama-4-maverick-17b-128e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": detailed_prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ],
    temperature=0.1,
    max_tokens=3000
)
elapsed_seconds = time.perf_counter() - start_time

print(f"Response time: {elapsed_seconds:.2f} seconds")
print(response.choices[0].message.content)