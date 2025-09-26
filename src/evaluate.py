from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu
import sys
import os

# Add the src directory to the path for imports
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from data_loader import load_python_code_data
from config import config

class CodeDocEvaluator:
    def __init__(self, model_path=None):
        # Use config if no model path provided
        model_path = model_path or config.model_config.model_path
        
        # Load the model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.config = config.model_config

    def evaluate(self, code_data, doc_data):
        """
        Evaluate the model using BLEU scores.
        :param code_data: List of code snippets
        :param doc_data: List of reference documentation
        :return: Average BLEU score
        """
        bleu_scores = []

        for code_snippet, reference_doc in zip(code_data, doc_data):
            # Generate documentation for the code snippet
            inputs = self.tokenizer(
                code_snippet, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.config.max_length
            )
            outputs = self.model.generate(
                inputs['input_ids'], 
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams
            )
            generated_doc = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Calculate BLEU score
            reference = [reference_doc.split()]  # Tokenize reference
            candidate = generated_doc.split()  # Tokenize generated doc
            bleu_score = sentence_bleu(reference, candidate)
            bleu_scores.append(bleu_score)

        # Return the average BLEU score
        return sum(bleu_scores) / len(bleu_scores)

if __name__ == "__main__":
    # Load test data
    code_data, doc_data = load_python_code_data()

    # Initialize evaluator
    evaluator = CodeDocEvaluator()

    # Evaluate the model
    avg_bleu_score = evaluator.evaluate(code_data, doc_data)
    print(f"Average BLEU Score: {avg_bleu_score}")