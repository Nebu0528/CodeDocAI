from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
import numpy as np

class CodeDocDataset(Dataset):
    def __init__(self, code_texts, doc_texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.code_texts = code_texts
        self.doc_texts = doc_texts
        self.max_length = max_length

    def __len__(self):
        return len(self.code_texts)

    def __getitem__(self, idx):
        code = self.code_texts[idx]
        doc = self.doc_texts[idx]

        # Prepare input and target
        inputs = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer(
            doc,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

class code_doc:
    def __init__(self, model_name="t5-base"):  # Updated to t5-base for better performance
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Add special tokens for code structure
        special_tokens = ['<func>', '</func>', '<param>', '</param>', '<desc>', '</desc>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def train(self, code_data, doc_data, epochs=5, batch_size=8):
        # Create dataset
        dataset = CodeDocDataset(code_data, doc_data, self.tokenizer)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            save_strategy="epoch"
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )

        # Train the model
        trainer.train()
