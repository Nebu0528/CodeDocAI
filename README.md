# CodeDocAI: An NLP-Powered Code Documentation Generator


## Project Overview
This project aims to help programmers write documentation for their code, using a fine-tuned T5 model to generate readable documentation from code chunks. The training is done on the command line and the documentation is generated on a web interface through Streamlit.  

## Features

- Documentation generation for code chunks.
- Interactive Web Interface made using Streamlit
- Model fine-tuned on code-documentation pairs.

## Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/Nebu0528/CodeDocAI
```

### 2. Install Dependencies
Run the command below to install all the dependencies needed:
```bash
pip install -r dependencies.txt
```
### 3. Train the Model 
This step is necessary to teach the model how to interpret code and generate corresponding documentation based on that training. The model is being trained on this:
```bash
https://huggingface.co/datasets/jtatman/python-code-dataset-500k
```
- 1. Train the model using:
```bash
python src/model.py
```
- 2. After training, the model will be saved in the models/codetext_t5/ directory.

### 4. Run the Streamlit App
Once the model is trained, run the app:
```bash
streamlit run interface/website.py
```

### 5. Access App
Once the app is running, open the URL provided by Streamlit 
```bash 
http://localhost:8501
```

## Usage
1. Paste a code snippet into the text area in the Streamlit interface.
2. Click "Generate Documentation" to generate the documentation for your code.

### Example Code Snippet
```bash
def multiply(a, b):
    #This function returns the product of variables a and b
    return a * b
```

## Future Improvements
1. Only works if you provide one function at a time with detailed comments
2. Integrate the Tranformers and BART Models to summarize the documentation
3. Handle different training dataset types (i.e .json, csv etc..)
4. Being able to download the generated documentation
5. Train the model on this [dataset](https://huggingface.co/datasets/bigcode/the-stack)

## Evaluate the Model

To evaluate the model's performance using BLEU scores, run the following command:

```bash
python src/evaluate.py
```

This will calculate the average BLEU score for the model's generated documentation compared to the reference documentation.
