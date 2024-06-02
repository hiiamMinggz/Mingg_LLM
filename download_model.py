from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import os
# Load model from HuggingFace Hub
if len(os.listdir('models/embedding')) == 0:
    tokenizer = AutoTokenizer.from_pretrained('keepitreal/vietnamese-sbert')
    model = AutoModel.from_pretrained('keepitreal/vietnamese-sbert')
    tokenizer.save_pretrained('models/embedding')
    model.save_pretrained('models/embedding')
# if len(os.listdir('models/vi-LLM')) == 0:
#     tokenizer = AutoTokenizer.from_pretrained('Viet-Mistral/Vistral-7B-Chat')
#     model = AutoModelForCausalLM.from_pretrained(
#         'Viet-Mistral/Vistral-7B-Chat')
#     tokenizer.save_pretrained('models/vi-LLM')
#     model.save_pretrained('models/vi-LLM')