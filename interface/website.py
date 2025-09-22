import sys
import os
import streamlit as st
import pandas as pd


# Reference: https://docs.streamlit.io/get-started/tutorials/create-an-app
# Set up the path for importing the inference module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from evaluate import CodeDocEvaluator
from inference import CodeToDocInference




st.title("NLP-Powered Code Documentation")

# Text area for user to put code
code_input = st.text_area("Enter code here")

# Button for document generation
if st.button("Generate Documentation"):
    infer = CodeToDocInference()
    documentation = infer.generate_doc(code_input)
    
    # Display generated documentation
    st.subheader("Generated Documentation:")
    st.write(documentation)

# Can show the original code if the user wants
if st.checkbox('Show code'):
    st.subheader('Code')
    st.code(code_input, language='python')

# File uploader for evaluation
uploaded_file = st.file_uploader("Upload a CSV file with 'code' and 'doc' columns for evaluation")

if uploaded_file:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    code_data = data['code'].tolist()
    doc_data = data['doc'].tolist()

    # Evaluate the model
    evaluator = CodeDocEvaluator()
    avg_bleu_score = evaluator.evaluate(code_data, doc_data)

    # Display the BLEU score
    st.subheader("Evaluation Results")
    st.write(f"Average BLEU Score: {avg_bleu_score}")
