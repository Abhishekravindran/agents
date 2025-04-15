from typing import Callable, List
import time
import json
import requests
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage

class OllamaWrapper:
    def __init__(self, base_url: str = "http://192.168.4.168:11434", model: str = "llama2:latest"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"

    def __call__(self, messages: List[ChatMessage], stop: List[str] = [], replace_newline: bool = True) -> str:
        # Convert messages to a prompt format Ollama can understand
        prompt = ""
        for msg in messages:
            role = msg.type
            content = msg.content
            if role == "human":
                prompt += f"Human: {content}\n"
            elif role == "ai":
                prompt += f"Assistant: {content}\n"
            else:
                prompt += f"{content}\n"
        
        prompt += "Assistant: "

        # Make request to Ollama API
        for i in range(3):  # Retry logic
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "stop": stop if stop else None
                        }
                    },
                    timeout=60
                )
                response.raise_for_status()
                output = response.json()["response"].strip()
                break
            except (requests.exceptions.RequestException, KeyError) as e:
                print(f'\nRetrying {i}... Error: {str(e)}')
                time.sleep(1)
        else:
            raise RuntimeError('Failed to generate response from Ollama')

        if replace_newline:
            output = output.replace('\n', '')
        return output

class GPTWrapper:
    def __init__(self, llm_name: str, openai_api_key: str, long_ver: bool):
        self.model_name = llm_name
        if long_ver:
            llm_name = 'gpt-3.5-turbo-16k'
        self.llm = ChatOpenAI(
            model=llm_name,
            temperature=0.0,
            openai_api_key=openai_api_key,
        )

    def __call__(self, messages: List[ChatMessage], stop: List[str] = [], replace_newline: bool = True) -> str:
        kwargs = {}
        if stop != []:
            kwargs['stop'] = stop
        for i in range(6):
            try:
                output = self.llm(messages, **kwargs).content.strip('\n').strip()
                break
            except Exception as e:
                print(f'\nRetrying {i}... Error: {str(e)}')
                time.sleep(1)
        else:
            raise RuntimeError('Failed to generate response')

        if replace_newline:
            output = output.replace('\n', '')
        return output

def LLM_CLS(llm_name: str, openai_api_key: str = None, long_ver: bool = False) -> Callable:
    if llm_name == 'ollama':
        return OllamaWrapper(base_url="http://192.168.4.168:11434", model="llama2")
    elif 'gpt' in llm_name:
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for GPT models")
        return GPTWrapper(llm_name, openai_api_key, long_ver)
    else:
        raise ValueError(f"Unknown LLM model name: {llm_name}")
