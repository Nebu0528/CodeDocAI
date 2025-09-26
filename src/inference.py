from transformers import T5Tokenizer, T5ForConditionalGeneration
import ast
import sys
import os

# Add the src directory to the path for imports
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config import config

class CodeToDocInference:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = config.model_config.model_path
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.config = config.model_config

    def parse_code(self, code_snippet):
        """Parse the input code to extract relevant information."""
        try:
            tree = ast.parse(code_snippet)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    params = [arg.arg for arg in node.args.args]
                    body = ast.unparse(node.body)
                    return {
                        'name': node.name,
                        'params': params,
                        'body': body,
                        'returns': node.returns and ast.unparse(node.returns)
                    }
        except:
            return None

    def generate_doc(self, code_snippet):
        # Parse the code first
        code_info = self.parse_code(code_snippet)
        if not code_info:
            return "Unable to parse the code snippet. Please ensure it's a valid Python function."

        # Create structured input
        structured_input = (
            f"<func>{code_info['name']}</func>\n"
            f"<param>{', '.join(code_info['params'])}</param>\n"
            f"Body:\n{code_info['body']}"
        )

        # Generate documentation
        inputs = self.tokenizer(
            structured_input, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.config.max_length
        )
        outputs = self.model.generate(
            inputs['input_ids'],
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
            length_penalty=self.config.length_penalty,
            early_stopping=self.config.early_stopping,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            do_sample=True if self.config.temperature > 1.0 else False
        )

        # Decode and format the output
        raw_doc = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Format the documentation
        formatted_doc = self.format_documentation(raw_doc, code_info)
        return formatted_doc

    def format_documentation(self, raw_doc, code_info):
        """Format the generated documentation in a structured way."""
        template = f"""
Function: {code_info['name']}

Description:
{raw_doc}

Parameters:
{chr(10).join([f'- {param}: ' + self.generate_param_desc(param) for param in code_info['params']])}

Returns:
{self.generate_return_desc(code_info['returns'])}
"""
        return template

    def generate_param_desc(self, param):
        """Generate specific description for a parameter."""
        inputs = self.tokenizer(
            f"Describe parameter {param}:", 
            return_tensors="pt",
            max_length=self.config.max_length
        )
        outputs = self.model.generate(
            inputs['input_ids'], 
            max_new_tokens=50,
            temperature=self.config.temperature
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_return_desc(self, return_type):
        """Generate description for the return value."""
        return_type = return_type if return_type else "None"
        inputs = self.tokenizer(
            f"Describe return value of type {return_type}:", 
            return_tensors="pt",
            max_length=self.config.max_length
        )
        outputs = self.model.generate(
            inputs['input_ids'], 
            max_new_tokens=50,
            temperature=self.config.temperature
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
