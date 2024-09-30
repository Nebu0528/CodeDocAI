from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
import torch
import os
from data_loader import load_python_code_data

class CodeToDocModel:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def train(self, code_data, doc_data, epochs=3, batch_size=4):
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        for epoch in range(epochs):
            for i in range(0, len(code_data), batch_size):
                batch_code = code_data[i:i+batch_size]
                batch_docs = doc_data[i:i+batch_size]
                
                inputs = self.tokenizer(batch_code, return_tensors="pt", padding=True, truncation=True)
                labels = self.tokenizer(batch_docs, return_tensors="pt", padding=True, truncation=True).input_ids
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=inputs.input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                #Display training process (testing purposes)
                #print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def save_model(self, path="models/t5_code_to_text/"):
        # Ensure the path exists before saving
        if not os.path.exists(path):
            os.makedirs(path)

        # Save the model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        #print(f"Model and tokenizer saved to {path}")

if __name__ == "__main__":
    # Load training data
    code_data, doc_data = load_python_code_data()

    # Initialize and train the model
    model = CodeToDocModel()
    model.train(code_data, doc_data, epochs=3, batch_size=2)

    # Output handling for generated documentation
    for code, doc in zip(code_data, doc_data):
        print(f"Input Code:\n{code}\n")
        print(f"Generated Documentation:\n{doc}\n")
        
    # Save the model
    model.save_model()
