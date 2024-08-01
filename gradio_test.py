import argparse

import gradio as gr
import requests
from openai import OpenAI

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Chatbot Interface with Customizable Parameters"
)
parser.add_argument(
    "--model-url", type=str, default="http://localhost:8000/v1", help="Model URL"
)
parser.add_argument(
    "-m", "--model", type=str, default=None, help="Model name for the chatbot"
)
parser.add_argument(
    "--temp", type=float, default=0.8, help="Temperature for text generation"
)
parser.add_argument(
    "--stop-token-ids", type=str, default="", help="Comma-separated stop token IDs"
)
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--port", type=int, default=8001)

# Parse the arguments
args = parser.parse_args()

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = args.model_url

# Create an OpenAI client to interact with the API server
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def get_first_model(url) -> str:
    response = requests.get(f"{url}/models")
    if response.status_code == 200:
        models = response.json().get("data", [])
        if models:
            model_name = models[0]["id"]
            return model_name
        else:
            raise ValueError("No models found at the specified model URL.")
    else:
        raise ValueError(f"Failed to fetch models from {args.model_url}: {response.text}")

    

def predict(message, history):
    # Convert chat history to OpenAI format
    history_openai_format = [
        # {"role": "system", "content": "You are a great ai assistant."}
    ]
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    # Create a chat completion request and send it to the API server
    stream = client.chat.completions.create(
        model=args.model if args.model else get_first_model(args.model_url),  # Model name to use, default to the first model
        messages=history_openai_format,  # Chat history
        temperature=args.temp,  # Temperature for text generation
        stream=True,  # Stream response
        extra_body={
            "repetition_penalty": 1,
            "stop_token_ids": (
                [int(id.strip()) for id in args.stop_token_ids.split(",") if id.strip()]
                if args.stop_token_ids
                else []
            ),
        },
    )

    # Read and return generated text from response stream
    partial_message = ""
    for chunk in stream:
        partial_message += chunk.choices[0].delta.content or ""
        yield partial_message


# Create and launch a chat interface with Gradio
gr.ChatInterface(predict).queue().launch(
    server_name=args.host, server_port=args.port, share=True
)
