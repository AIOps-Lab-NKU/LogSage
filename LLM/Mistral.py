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
    prompt = "Please describe and explain the faults in the following log sequence. Provide only a brief fault description without any solution to it."
    print(predict(prompt+''.join(seq)))
