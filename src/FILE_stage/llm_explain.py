import os
import json
import yaml
from random import randint

from mistral_inference.model import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


def load(model_path="/home/cuitianyu/cuitianyu/mistral_models/7B_instruct")-> tuple:
    tokenizer = MistralTokenizer.from_file(os.path.join(model_path, "tokenizer.model.v3"))
    model = Transformer.from_folder(model_path)
    return tokenizer,model

def predict(content: str)-> str:
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=content)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=1024, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    return result

if __name__ == "__main__":
    with open('logs.txt', 'r') as f:
        seq = f.readlines()
    tokenizer,model = load()
    prompt = "Task: The provided kernel logs contain fault-indicating kernel panic mixed with extraneous technical details. Summarize the key information while preserving essential context. Step 1: Kernel Panic Analysis. Details: Identify core fault-indicating kernel logs. Provide a structured response that captures the most relevant insights. Step 2: Noise Reduction. Details: Filter out low-level system diagnostics that do not contribute to kernel panic understanding. Ensure the explanation is clear and concise. Step 3: Concise Summary Generation. Details: Produce a human-readable summary of the kernel panic and its implications in a structured format. Maintain interpretability without introducing assumptions."
    print(predict(prompt+''.join(seq)))