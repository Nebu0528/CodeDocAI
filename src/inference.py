from transformers import T5Tokenizer, T5ForConditionalGeneration
from data_loader import load_python_code_data 

class CodeToDocInference:
    #Initialization of T5 Model
    def __init__(self, model_path="models/t5_code_to_text/"):
        #load the model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

    def generate_doc(self, code_snippet):
        #define the inputs and outputs
        inputs = self.tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(inputs['input_ids'], max_new_tokens=200)
        #decode and return the results
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
