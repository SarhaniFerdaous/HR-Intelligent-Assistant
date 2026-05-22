import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

models = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-coder:free",
    "google/gemma-4-31b-it:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
]

for model in models:
    try:
        llm = ChatOpenAI(
            model=model,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.2,
            max_tokens=4096,
        )
        result = llm.invoke("say hello")
        print(f"✅ Working model: {model}")
        print(result.content)
        break
    except Exception as e:
        print(f"❌ {model}: {e}")