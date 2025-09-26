"""
API-based inference for CodeDocAI using external AI services.
Supports OpenAI, Anthropic, and Google Gemini APIs.
"""
import ast
import sys
import os
import requests
import json
from typing import Dict, Any, Optional

# Add the src directory to the path for imports
src_path = os.path.dirname(os.path.abspath(__file__))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from config import config

class APIInference:
    """API-based inference for code documentation generation."""
    
    def __init__(self, api_key: Optional[str] = None, api_provider: Optional[str] = None):
        self.config = config.model_config
        self.api_key = api_key or self.config.api_key
        self.api_provider = api_provider or self.config.api_provider
        
        if not self.api_key:
            raise ValueError("API key is required for API-based inference")
        
        # Set up API-specific configurations
        self.setup_api_config()
    
    def setup_api_config(self):
        """Set up API-specific configurations."""
        if self.api_provider == "openai":
            self.api_url = "https://api.openai.com/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self.model = self.config.api_model or "gpt-3.5-turbo"
            
        elif self.api_provider == "anthropic":
            self.api_url = "https://api.anthropic.com/v1/messages"
            self.headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            self.model = self.config.api_model or "claude-3-haiku-20240307"
            
        elif self.api_provider == "gemini":
            # Use the model name in the URL, default to gemini-2.0-flash
            model_name = self.config.api_model or "gemini-2.0-flash"
            self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            self.headers = {
                "Content-Type": "application/json",
                "X-goog-api-key": self.api_key
            }
            self.model = model_name
            
        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")
    
    def parse_code(self, code_snippet: str) -> Optional[Dict[str, Any]]:
        """Parse the input code to extract relevant information."""
        try:
            tree = ast.parse(code_snippet)
            
            # Look for functions first
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    params = []
                    for arg in node.args.args:
                        param_info = {"name": arg.arg}
                        if arg.annotation:
                            param_info["type"] = ast.unparse(arg.annotation)
                        params.append(param_info)
                    
                    # Get docstring if exists
                    docstring = ast.get_docstring(node)
                    
                    return {
                        'type': 'function',
                        'name': node.name,
                        'params': params,
                        'body': ast.unparse(node.body),
                        'returns': ast.unparse(node.returns) if node.returns else None,
                        'docstring': docstring,
                        'decorators': [ast.unparse(dec) for dec in node.decorator_list]
                    }
            
            # Look for classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append(item.name)
                    
                    return {
                        'type': 'class',
                        'name': node.name,
                        'methods': methods,
                        'body': ast.unparse(node.body),
                        'docstring': ast.get_docstring(node),
                        'bases': [ast.unparse(base) for base in node.bases]
                    }
            
            # If no functions or classes, treat as generic code
            return {
                'type': 'code',
                'body': code_snippet
            }
            
        except Exception as e:
            return None
    
    def create_prompt(self, code_info: Dict[str, Any]) -> str:
        """Create a prompt for the API based on parsed code information."""
        if code_info['type'] == 'function':
            params_str = ", ".join([p['name'] + (f": {p.get('type', '')}" if p.get('type') else "") 
                                   for p in code_info['params']])
            
            prompt = f"""Please generate comprehensive documentation for the following Python function:

Function Name: {code_info['name']}
Parameters: {params_str}
Returns: {code_info.get('returns', 'Not specified')}

Code:
```python
def {code_info['name']}({params_str}):
    {code_info['body']}
```

Please provide:
1. A clear, concise description of what the function does
2. Detailed parameter descriptions (purpose, type, constraints)
3. Return value description
4. Any important notes about usage, complexity, or edge cases
5. Example usage if helpful

Format the response as clean markdown documentation."""

        elif code_info['type'] == 'class':
            prompt = f"""Please generate comprehensive documentation for the following Python class:

Class Name: {code_info['name']}
Base Classes: {', '.join(code_info['bases']) if code_info['bases'] else 'None'}
Methods: {', '.join(code_info['methods']) if code_info['methods'] else 'None'}

Code:
```python
{code_info['body']}
```

Please provide:
1. A clear description of the class purpose and functionality
2. Overview of key methods and their purposes
3. Usage examples
4. Any important notes about the class design or usage patterns

Format the response as clean markdown documentation."""

        else:
            prompt = f"""Please generate documentation for the following Python code:

```python
{code_info['body']}
```

Please provide:
1. A description of what the code does
2. Key components and their purposes
3. Usage notes if applicable

Format the response as clean markdown documentation."""

        return prompt
    
    def call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API for documentation generation."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Python developer who writes clear, comprehensive documentation. Generate well-structured documentation that helps other developers understand and use the code effectively."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"OpenAI API request failed: {str(e)}")
        except KeyError as e:
            raise Exception(f"Unexpected OpenAI API response format: {str(e)}")
    
    def call_anthropic_api(self, prompt: str) -> str:
        """Call Anthropic API for documentation generation."""
        payload = {
            "model": self.model,
            "max_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": f"You are an expert Python developer who writes clear, comprehensive documentation. {prompt}"
                }
            ]
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['content'][0]['text'].strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Anthropic API request failed: {str(e)}")
        except KeyError as e:
            raise Exception(f"Unexpected Anthropic API response format: {str(e)}")
    
    def call_gemini_api(self, prompt: str) -> str:
        """Call Google Gemini API for documentation generation."""
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"You are an expert Python developer who writes clear, comprehensive documentation. {prompt}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_new_tokens,
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Gemini API request failed: {str(e)}")
        except KeyError as e:
            raise Exception(f"Unexpected Gemini API response format: {str(e)}")
    
    def generate_doc(self, code_snippet: str) -> str:
        """Generate documentation for the given code snippet."""
        # Parse the code
        code_info = self.parse_code(code_snippet)
        if not code_info:
            return "❌ Unable to parse the code snippet. Please ensure it contains valid Python code."
        
        # Create prompt
        prompt = self.create_prompt(code_info)
        
        try:
            # Call appropriate API
            if self.api_provider == "openai":
                documentation = self.call_openai_api(prompt)
            elif self.api_provider == "anthropic":
                documentation = self.call_anthropic_api(prompt)
            elif self.api_provider == "gemini":
                documentation = self.call_gemini_api(prompt)
            else:
                return f"❌ Unsupported API provider: {self.api_provider}"
            
            return documentation
            
        except Exception as e:
            return f"❌ Error generating documentation: {str(e)}\n\nPlease check your API key and try again."

class HybridInference:
    """Hybrid inference that can use either local model or API."""
    
    def __init__(self, use_api: bool = None, api_key: str = None):
        self.use_api = use_api if use_api is not None else config.model_config.use_api
        
        if self.use_api and (api_key or config.model_config.api_key):
            self.api_inference = APIInference(api_key=api_key)
            self.inference_type = "API"
        else:
            # Fall back to local model
            from inference import CodeToDocInference
            self.local_inference = CodeToDocInference()
            self.inference_type = "Local"
    
    def generate_doc(self, code_snippet: str) -> str:
        """Generate documentation using the configured inference method."""
        if hasattr(self, 'api_inference'):
            result = self.api_inference.generate_doc(code_snippet)
            # Add inference type info
            return f"*Generated using {self.inference_type} ({self.api_inference.api_provider} - {self.api_inference.model})*\n\n{result}"
        else:
            result = self.local_inference.generate_doc(code_snippet)
            return f"*Generated using {self.inference_type} Model*\n\n{result}"
    
    def parse_code(self, code_snippet: str):
        """Parse code using the appropriate parser."""
        if hasattr(self, 'api_inference'):
            return self.api_inference.parse_code(code_snippet)
        else:
            return self.local_inference.parse_code(code_snippet)