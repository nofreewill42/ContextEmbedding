import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2TokenizerFast, Qwen2ForCausalLM
from torch.optim import AdamW


# Setup
#model_name = "Qwen/Qwen2.5-Coder-3B-instruct"
model_name = "checkpoints/final_model"
print(f"Loading model: {model_name}\n")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")
model = model.to(device)
print(f'Default model training mode for AutoModelForCausalLM: {model.training}')
model.train()
print(f'Updated mode: {model.training}')

embedding_text = "<|fim_middle|>"
system_prompts = {
    "default": {'role': 'system', 'content': "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    "concise": {'role': 'system', 'content': "You are Qwen, created by Alibaba Cloud. You are a helpful assistant, not wasting the user's time, you ALWAYS answer VERY concisely, up to 10 words at MAX."},
}
conversations = [
    [
        system_prompts['default'],
        {'role': 'user', 'content': "What is the color of the sky?\n" + embedding_text + "\n"},
    ],
]

embedding_tokens = tokenizer(embedding_text, return_tensors='pt')
emb_toks_len = embedding_tokens['input_ids'].shape[1]

def split_tokens(tokens_to_split, tokens_to_split_with):
    tokens_to_split_list = tokens_to_split['input_ids'].tolist()[0]  # shape of [1,:]
    tokens_to_split_with_tokens_list = tokens_to_split_with['input_ids'].tolist()[0]  # shape of [1,:]
    split_with_len = len(tokens_to_split_with_tokens_list)
    # [[2,3],[0,3],[4,2],[1,7]].index([4,2]) -> 2 (throws a ValueError if not found)
    token_lists_to_check = [tokens_to_split_list[i:i+split_with_len] for i in range(len(tokens_to_split_list) - split_with_len)]
    split_index = token_lists_to_check.index(tokens_to_split_with_tokens_list)
    # Split input_ids and attention_mask
    input_ids = tokens_to_split['input_ids']
    attention_mask = tokens_to_split['attention_mask']
    tokens_to_embed = {
        'input_ids': input_ids[:, :split_index + split_with_len],  # Include up to and including embedding tokens
        'attention_mask': attention_mask[:, :split_index + split_with_len],
    }
    tokens_to_respond = {
        'input_ids': input_ids[:, split_index:],  # Start after the embedding tokens
        'attention_mask': attention_mask[:, split_index:],
    }
    return tokens_to_embed, tokens_to_respond

for conversation in conversations:
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(text, return_tensors='pt').to(device)
    # Split at embedding_tokens as follows: [*system_and_user_tokens] + [*embedding_tokens] |+ [*assistant_tokens] ; cut at |
    tokens_to_embed, tokens_to_respond = split_tokens(tokens_to_split=tokens, tokens_to_split_with=embedding_tokens)

    # Caching inputs
    model_output = model(**tokens_to_embed)
    past_key_values = model_output.past_key_values
    embeddings_key_values = tuple((layer_cache[0][:,:,-emb_toks_len:,:],
                                   layer_cache[1][:,:,-emb_toks_len:,:])
                             for layer_cache in past_key_values)

    # Prepare cached inputs for generation
    input_ids = tokens_to_respond['input_ids'][:,:1+emb_toks_len]
    attention_mask = tokens_to_respond['attention_mask'][:,:1+emb_toks_len]
    past_key_values = embeddings_key_values
    result = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            num_logits_to_keep=1,
                            use_cache=True,
                            max_new_tokens=32,
                            eos_token_id=None)

    result_text = tokenizer.decode(result.tolist())
    print(result_text)

# Example output without fine-tuning:
    '''
tokenizer.decode(result.tolist())
'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n<|endoftext|><|endoftext|><|fim_prefix|>\n<|endoftext|><|fim_prefix|>\n<|endoftext|>Question: What is the purpose of the `__init__` method in the `Person` class?\n\nAnswer: The `__init__` method is a special method in Python that is called when an object of a class is created. It is used to initialize the attributes of the object. In the `Person` class, the `__init__` method takes two parameters: `name` and `age`, and assigns them to the corresponding attributes of the object. This allows the object to have a name and an age when it is created. The `__init__` method is a constructor in object-oriented programming, which is used to create new instances of a class. It is a fundamental part of object-oriented programming and is used to set up the initial state of an object. The `__init__` method is called automatically when a new object is created, and it is used to initialize the attributes of the object. The `__init__` method is a special method in Python that is called when an object of a class is created. It is used to initialize the attributes of the object. In the `Person` class, the `__init__` method takes two parameters: `name` and `age`, and assigns them to the corresponding attributes of the object. This allows the object to have a name and an age when it is created. The `__init__` method is a constructor in object-oriented programming, which is used to create new instances of a class. It is a fundamental part of object-oriented programming and is used to set up the initial state of an object. The `__init__` method is called automatically when a new object is created, and it is used to initialize the attributes of the object. The `__init__` method is a special method in Python that is called when an object of a class is created. It is used to initialize the attributes of the object. In the `Person` class, the `__init__` method takes two parameters: `name` and `age`, and assigns them to the corresponding attributes of the object. This allows the object to have a name and an age when it is created. The `__init__` method is a constructor in object-oriented programming, which is used to create new instances of a class. It is a fundamental part of object'
    '''