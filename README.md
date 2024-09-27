# NLP-Powered Code Documentation Generator

## Project Overview
Project aims to help programmers generate documentation for their code
## Features
- None 

## Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/Nebu0528/CodeDocAI.git
cd nlp_code_doc
```

### 2. Install Dependencies
Run the command below to install all the dependencies needed:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run interface/app.py
```

### 4. Access App
Once the app is running, open the URL provided by Streamlit 
```bash 
http://localhost:8501
```


### Example Code Snippet
```bash
def add(a, b):
    return a + b
# This function returns the sum two numbers
```

## Future Improvements
1. Fine tune the T5 model
2. Implement the data loader function to take the text data and turn it into manageable dataframes for the model
3. Handle different training dataset types (i.e .json, csv etc..)
4. Train the model on this [dataset](https://huggingface.co/datasets/bigcode/the-stack)