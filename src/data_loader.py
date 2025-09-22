from datasets import load_dataset
from huggingface_hub import login
import ast

def extract_function_info(code_str):
    """Extract function name, parameters, and body from code."""
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                params = [arg.arg for arg in node.args.args]
                body = ast.unparse(node.body)
                return {
                    'name': node.name,
                    'params': params,
                    'body': body,
                    'returns': extract_return_type(node)
                }
    except:
        return None

def extract_return_type(node):
    """Extract return type hints if available."""
    if node.returns:
        return ast.unparse(node.returns)
    return None

def load_python_code_data():
    """Load and preprocess Python code snippets."""
    dataset = load_dataset('jtatman/python-code-dataset-500k', split='train')
    
    code_data = []
    doc_data = []
    
    for entry in dataset:
        if entry['language'] == 'Python' and 'code' in entry:
            code_content = entry['code']
            function_info = extract_function_info(code_content)
            
            if function_info:
                # Create structured input for the model
                processed_code = (
                    f"Function: {function_info['name']}\n"
                    f"Parameters: {', '.join(function_info['params'])}\n"
                    f"Returns: {function_info['returns'] if function_info['returns'] else 'None'}\n"
                    f"Body:\n{function_info['body']}"
                )
                
                # Create structured documentation template
                doc_template = (
                    f"Description: {entry.get('description', '')}\n"
                    f"Parameters:\n" + 
                    '\n'.join([f"- {param}: [description]" for param in function_info['params']]) +
                    f"\nReturns: [description]"
                )
                
                code_data.append(processed_code)
                doc_data.append(doc_template)
    
    return code_data, doc_data

if __name__ == "__main__":
    # Test the data loader
    code_data, doc_data = load_python_code_data()
    print(f"Loaded {len(code_data)} code-documentation pairs")