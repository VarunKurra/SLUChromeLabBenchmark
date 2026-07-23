# Models
# holo3-1-35b-a3b

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

client = OpenAI(
    base_url="https://api.hcompany.ai/v1/",
    api_key=os.getenv("HCOMPANY_API_KEY", "")
)

image_url = "https://www.image2url.com/r2/default/images/1780602309583-a0c99451-72f0-4294-babb-7bc69cb09002.webp"

chat_completion = client.chat.completions.create(
    model="holo3-1-35b-a3b",
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
                        "url": image_url
                    }
                }
            ]
        }
    ]
)

print(chat_completion.choices[0].message.content)