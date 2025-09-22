from transformers import T5Tokenizer, T5ForConditionalGeneration
import ast

class CodeToDocInference:
    def __init__(self, model_path="models/codetext_t5"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

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
        inputs = self.tokenizer(structured_input, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(
            inputs['input_ids'],
            max_length=200,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
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
        inputs = self.tokenizer(f"Describe parameter {param}:", return_tensors="pt")
        outputs = self.model.generate(inputs['input_ids'], max_length=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_return_desc(self, return_type):
        """Generate description for the return value."""
        return_type = return_type if return_type else "None"
        inputs = self.tokenizer(f"Describe return value of type {return_type}:", return_tensors="pt")
        outputs = self.model.generate(inputs['input_ids'], max_length=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
