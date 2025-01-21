from datasets import load_dataset
from huggingface_hub import login



def load_python_code_data():
    """
    Load Python code snippets from the Python Code Dataset.
    This function retrieves all relevant code snippets without limiting the count.
    """
    # Load the Python Code Dataset
    dataset = load_dataset('jtatman/python-code-dataset-500k', split='train', download_mode='reuse_cache_if_exists')

    code_data = []
    doc_data = []

    # Iterate over the dataset
    for entry in dataset:
        #print(entry)  # Print the entry to check its structure
        if 'language' in entry and entry['language'] == 'Python' and 'code' in entry:
            code_content = entry['code']  # Accessing the code snippet
            comment = entry.get('description', '')  # Use .get() to avoid KeyError

            code_data.append(code_content)
            doc_data.append(comment)

    return code_data, doc_data

# For testing purposes
# if __name__ == "__main__":
#    code_data, doc_data = load_python_code_data()  # Load all files without limit
#    print(f"Loaded {len(code_data)} code snippets.")
#    for code in code_data:
#        print(code)  # Print each loaded code snippet for verification
