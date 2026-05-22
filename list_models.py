import requests
import os
from dotenv import load_dotenv

load_dotenv()

r = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"}
)

models = r.json()["data"]
free = [m["id"] for m in models if ":free" in m["id"]]
for m in sorted(free):
    print(m)