import sys
import os
import streamlit as st

# Reference: https://docs.streamlit.io/get-started/tutorials/create-an-app
# Set up the path for importing the inference module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

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
