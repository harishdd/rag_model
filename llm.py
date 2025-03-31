import os
from dotenv import load_dotenv
from langchain_together import Together

load_dotenv()

class LLM:
    def get_llm_together(self) -> Together:
        llm = Together(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            together_api_key=os.getenv("TOGETHER_API_KEY"),
        )
        return llm
