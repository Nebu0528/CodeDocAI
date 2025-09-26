from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import os

# Add the src directory to the path for imports
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config import config

class CodeDocDataset(Dataset):
    def __init__(self, code_texts, doc_texts, tokenizer, max_length=None):
        self.tokenizer = tokenizer
        self.code_texts = code_texts
        self.doc_texts = doc_texts
        self.max_length = max_length or config.model_config.max_length

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
    def __init__(self, model_name=None):
        model_name = model_name or config.model_config.model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.config = config
        
        # Add special tokens for code structure
        special_tokens = ['<func>', '</func>', '<param>', '</param>', '<desc>', '</desc>']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def train(self, code_data, doc_data, val_code=None, val_doc=None, epochs=None, batch_size=None):
        epochs = epochs or self.config.training_config.epochs
        batch_size = batch_size or self.config.training_config.batch_size
        
        # Create datasets
        train_dataset = CodeDocDataset(code_data, doc_data, self.tokenizer)
        eval_dataset = None
        if val_code and val_doc:
            eval_dataset = CodeDocDataset(val_code, val_doc, self.tokenizer)
        
        # Define training arguments using config
        training_args = TrainingArguments(
            output_dir=self.config.training_config.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=self.config.training_config.warmup_steps,
            weight_decay=self.config.training_config.weight_decay,
            logging_dir=self.config.training_config.logging_dir,
            logging_steps=self.config.training_config.logging_steps,
            save_strategy=self.config.training_config.save_strategy,
            evaluation_strategy=self.config.training_config.evaluation_strategy if eval_dataset else "no",
            save_total_limit=self.config.training_config.save_total_limit,
            load_best_model_at_end=self.config.training_config.load_best_model_at_end if eval_dataset else False,
            metric_for_best_model=self.config.training_config.metric_for_best_model,
            greater_is_better=self.config.training_config.greater_is_better,
            learning_rate=self.config.training_config.learning_rate
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # Train the model
        trainer.train()
        
        return trainer
    
    def save_model(self, path=None):
        """Save the trained model and tokenizer."""
        path = path or self.config.model_config.model_path
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
