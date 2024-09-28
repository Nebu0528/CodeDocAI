from datasets import load_dataset

def load_python_code_data():

    # load the python dataset
    dataset = load_dataset('jtatman/python-code-dataset-500k', split='train')

    #lists that contains python lines of code and its associated content
    python_data = []
    documentation = []

    # Print every entry in the dataset
    for entry in dataset:
        print(entry)  # Print the entry to check its structure
        #Filter for python
        if 'language' in entry and entry['language'] == 'Python' and 'code' in entry:
            #Access the code snippets and documentation comments
            code_content = entry['code'] 
            comment = entry.get('description', '') 

            #Append to the list
            code_data.append(code_content)
            documentation.append(comment)

    return code_data, documentation

#Testing
if __name__ == "__main__":
    code_data, documentation = load_python_code_data() 