import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, Qwen2ForCausalLM


import random
from tqdm import tqdm
from pathlib import Path
import json

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


# Setup
model_name = "Qwen/Qwen2.5-Coder-14B-instruct"
#model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = model.to(device)
#model.train()

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device#"auto"
)

# Prepare for training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Training data
embedding_text = "<|fim_middle|>"
system_prompts = {
    "default": {'role': 'system', 'content': "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    "concise": {'role': 'system', 'content': "You are Qwen, created by Alibaba Cloud. You are a helpful assistant, not wasting the user's time, you ALWAYS answer VERY concisely, up to 10 words at MAX."},
}
conversations = json.loads(Path('conversations.json').read_text())
conversations = [
    [
        system_prompts['default'],
        {'role': 'user', 'content': f"{conversation['question']}\n{embedding_text}\n"},
        {'role': 'assistant', 'content': conversation['question']},#{'role': 'assistant', 'content': conversation['answer']},
    ] for conversation in conversations if 'question' in conversation
] + [
    [
        system_prompts['default'],
        {'role': 'user', 'content': f"{conversation['answer']}\n{embedding_text}\n"},
        {'role': 'assistant', 'content': conversation['answer']},#{'role': 'assistant', 'content': conversation['answer']},
    ] for conversation in conversations if 'answer' in conversation
] + [
    [
        system_prompts['default'],
        {'role': 'user', 'content': f"{conversation['text']}\n{embedding_text}\n"},
        {'role': 'assistant', 'content': conversation['text']},#{'role': 'assistant', 'content': conversation['answer']},
    ] for conversation in conversations if 'text' in conversation
]


def train_conversation(model, conversation, tokenizer, device, scaler):
    """Process single conversation with gradient scaling and KV caching"""
    text = tokenizer.apply_chat_template(conversation, tokenize=False)
    tokens = tokenizer(text, return_tensors='pt').to(device)
    
    # Find split point for embedding token
    embedding_text = "<|fim_middle|>"
    embedding_tokens = tokenizer(embedding_text, return_tensors='pt')
    emb_toks_len = embedding_tokens['input_ids'].shape[1]
    
    tokens_list = tokens['input_ids'].tolist()[0]
    embed_list = embedding_tokens['input_ids'].tolist()[0]
    split_index = [tokens_list[i:i+len(embed_list)] for i in range(len(tokens_list) - len(embed_list))].index(embed_list)
    
    # Split tokens
    tokens_to_embed = {
        'input_ids': tokens['input_ids'][:, :split_index + emb_toks_len],
        'attention_mask': tokens['attention_mask'][:, :split_index + emb_toks_len],
    }
    tokens_to_respond = {
        'input_ids': tokens['input_ids'][:, split_index + emb_toks_len:],
        'attention_mask': tokens['attention_mask'][:, split_index + 0:],  # not having (+ emb_toks_len) is important
    }

    with autocast(device_type=str(device), dtype=torch.bfloat16):
        # Get embeddings
        model_output = model(**tokens_to_embed, use_cache=True)
        past_key_values = model_output.past_key_values
        embeddings_key_values = tuple(
            (layer_cache[0][:,:,-emb_toks_len:,:], layer_cache[1][:,:,-emb_toks_len:,:])
            for layer_cache in past_key_values
        )
        
        # Forward pass with labels
        result = model(
            input_ids=tokens_to_respond['input_ids'],
            attention_mask=tokens_to_respond['attention_mask'],
            past_key_values=embeddings_key_values,
            labels=tokens_to_respond['input_ids']
        )
    
    return result.loss

def train(model, conversations, tokenizer, device,
          num_epochs=1,
          lr=3e-5,
          warmup_steps=100,
          grad_clip=1.0):

    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=num_epochs * len(conversations)
    )
    
    model.train()
    pbar = tqdm(total=num_epochs * len(conversations))
    for epoch in range(num_epochs):
        # randomize conversations' order
        random.shuffle(conversations)  # this is in-place
        for conversation in conversations:
            optimizer.zero_grad()
            
            loss = train_conversation(model, conversation, tokenizer, device, scaler)
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            pbar.update(1)
            pbar.set_description(f"Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
    
    pbar.close()
    return model

# Start training
train(model, conversations, tokenizer, device)

Path('checkpoints').mkdir(exist_ok=True)
model.save_pretrained("checkpoints/final_model")#model.save_pretrained(f"checkpoints/final_model")
tokenizer.save_pretrained(f"checkpoints/final_model")