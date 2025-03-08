
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
import torch
import os
from data_loader import load_python_code_data

class code_doc:
    def __init__(self, model_name="t5-small"):
        #load tokenizer and model
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def train(self, code_data, doc_data, epochs=3, batch_size=4):
        #set the optimizer using AdamW
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        for epoch in range(epochs):
            for i in range(0, len(code_data), batch_size):
                #prepare batch data
                batch_code = code_data[i:i+batch_size]
                batch_docs = doc_data[i:i+batch_size]

                #tokenize the input code and documentation for the model
                inputs = self.tokenizer(batch_code, return_tensors="pt", padding=True, truncation=True)
                labels = self.tokenizer(batch_docs, return_tensors="pt", padding=True, truncation=True).input_ids

                #zero gradients before backward pass
                #forward pass through the model
                #then compute loss and perform backpropagation
                optimizer.zero_grad()
                outputs = self.model(input_ids=inputs.input_ids, labels=labels)
                loss = outputs.loss
                #look into this, possibly not needed
                #loss.backward()
                #optimizer.step()

                #display training process (testing purposes)
                #print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def model_to_save(self, path="models/codetext_t5/"):
        # Ensure the path exists before saving
        if not os.path.exists(path):
            os.makedirs(path)

        # Save the model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        #print(f"Model and tokenizer saved to {path}")

if __name__ == "__main__":
    #load training data
    code_data, doc_data = load_python_code_data()

    #initialize the model and train it
    model = code_doc()
    model.train(code_data, doc_data, epochs=3, batch_size=2)

    #save model
    model.model_to_save()
