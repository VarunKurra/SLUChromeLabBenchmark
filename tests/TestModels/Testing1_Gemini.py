# Models:
# gemini-3.5-flash
# gemini-3.1-flash-lite
# gemini-2.5-flash
# gemma-4-31b-it
# gemma-4-26b-a4b-it

import os
os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GLOG_minloglevel"] = "3"
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env")))

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY", ""))
image = Image.open("/users/varun/Downloads/Test_Graph.webp")

response = client.models.generate_content(
    model="models/gemma-4-31b-it", 
    contents=[image, "Tell me about this Graph. Keep in mind that I am blind. Be very precise on the data down to 3 decimal points."]
)

print(response.text)           