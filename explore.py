import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2TokenizerFast, Qwen2ForCausalLM
from torch.optim import AdamW


# Setup
model_name = "Qwen/Qwen2.5-Coder-3B-instruct"
print(f"Loading model: {model_name}\n")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")
model = model.to(device)

messages = [
    {'role': 'system', 'content': "You are Qwen, created by Alibaba Cloud. You are a helpful assistant, not wasting the user's time, you ALWAYS answer VERY concisely, up to 10 words at MAX."},
    {'role': 'user', 'content': "What color is the sky?"},
    {'role': 'assistant', 'content': "Blue (it depends though)"},
    {'role': 'user', 'content': "What's the new name of Twitter?"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Caching inputs
tokens = tokenizer(text, return_tensors='pt').to(device)
cache_tokens = {}
cache_tokens['input_ids'] = tokens['input_ids'][:,:-1]
cache_tokens['attention_mask'] = tokens['attention_mask'][:,:-1]
model_output = model(**cache_tokens)

# Prepare cached inputs for generation
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']
cache_position = torch.tensor([input_ids.shape[1]], device=device)
past_key_values = model_output.past_key_values
result = model.generate(input_ids=input_ids,
                        attention_mask=attention_mask,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        num_logits_to_keep=1,
                        use_cache=True,
                        max_new_tokens=512)

result_text = tokenizer.decode(result.tolist()[0])
print(result_text)