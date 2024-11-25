import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model():
    # Load quantized base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-14B-instruct", 
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    )
    
    # Load adapter and tokenizer
    model = PeftModel.from_pretrained(base_model, "checkpoints/final_model")
    tokenizer = AutoTokenizer.from_pretrained("checkpoints/final_model") 
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, prompt, device):
    embedding_text = "<|fim_middle|>"
    conversation = [
        {'role': 'system', 'content': "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {'role': 'user', 'content': f"{prompt}\n{embedding_text}\n"},
    ]
    
    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(text, return_tensors='pt').to(device)
    
    # Find split point
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
    
    # Generate embeddings
    model_output = model(**tokens_to_embed)
    embeddings_key_values = tuple(
        (layer_cache[0][:,:,-emb_toks_len:,:], layer_cache[1][:,:,-emb_toks_len:,:])
        for layer_cache in model_output.past_key_values
    )
    
    # Generate completion
    input_ids = tokens['input_ids'][:, split_index:split_index+1+emb_toks_len]
    attention_mask = tokens['attention_mask'][:, split_index:split_index+1+emb_toks_len]
    
    result = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=embeddings_key_values,
        num_logits_to_keep=1,
        use_cache=True,
        max_new_tokens=32,
        eos_token_id=None
    )
    
    return tokenizer.decode(result[0])

def main():
    print("Loading model...")
    model, tokenizer = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Model loaded! Enter prompts (Ctrl+C to exit):")
    
    try:
        while True:
            prompt = input("\nPrompt> ")
            if not prompt:
                continue
            response = generate_response(model, tokenizer, prompt, device)
            print(f"\nResponse: {response}")
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()