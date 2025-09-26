import sys
import os
import streamlit as st
import pandas as pd

# Set up the path for importing modules - ensure src directory is in Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from evaluate import CodeDocEvaluator
from inference import CodeToDocInference  
from config import config




st.set_page_config(
    page_title=config.web_config.title,
    page_icon="📚",
    layout="wide"
)

st.title(config.web_config.title)
st.markdown(config.web_config.description)

# Sidebar with configuration options
with st.sidebar:
    st.header("Configuration")
    
    # Model settings
    st.subheader("Generation Settings")
    temperature = st.slider(
        "Temperature", 
        min_value=0.1, 
        max_value=2.0, 
        value=config.model_config.temperature,
        step=0.1,
        help="Higher values make output more creative but less focused"
    )
    
    num_beams = st.slider(
        "Number of Beams", 
        min_value=1, 
        max_value=10, 
        value=config.model_config.num_beams,
        help="More beams generally produce better quality but slower generation"
    )
    
    max_new_tokens = st.slider(
        "Max New Tokens", 
        min_value=50, 
        max_value=500, 
        value=config.model_config.max_new_tokens,
        help="Maximum length of generated documentation"
    )

# Update config with user settings
config.model_config.temperature = temperature
config.model_config.num_beams = num_beams
config.model_config.max_new_tokens = max_new_tokens

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Code Input")
    
    # File upload option
    uploaded_code_file = st.file_uploader(
        "Upload a code file",
        type=config.web_config.allowed_extensions,
        help=f"Supported extensions: {', '.join(config.web_config.allowed_extensions)}"
    )
    
    code_input = ""
    if uploaded_code_file:
        if uploaded_code_file.size > config.web_config.max_file_size:
            st.error(f"File size exceeds limit of {config.web_config.max_file_size / (1024*1024):.1f}MB")
        else:
            code_input = str(uploaded_code_file.read(), "utf-8")
            st.success(f"Loaded {uploaded_code_file.name}")
    
    # Text area for manual code input
    manual_code = st.text_area(
        "Or enter code manually:",
        value=code_input,
        height=300,
        placeholder="def example_function(param1, param2):\n    \"\"\"Your function here\"\"\"\n    return param1 + param2"
    )
    
    code_to_process = manual_code or code_input

with col2:
    st.header("Generated Documentation")
    
    if st.button("Generate Documentation", type="primary", disabled=not code_to_process):
        if code_to_process:
            with st.spinner("Generating documentation..."):
                try:
                    infer = CodeToDocInference()
                    documentation = infer.generate_doc(code_to_process)
                    
                    # Display generated documentation
                    st.markdown("### Generated Documentation:")
                    st.markdown(documentation)
                    
                    # Option to download documentation
                    st.download_button(
                        label="Download Documentation",
                        data=documentation,
                        file_name="generated_documentation.md",
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating documentation: {str(e)}")
        else:
            st.warning("Please enter some code first!")

# Show original code option
if code_to_process and st.checkbox('Show Original Code'):
    st.subheader('Original Code')
    st.code(code_to_process, language='python')

# Evaluation section
st.header("Model Evaluation")

# File uploader for evaluation
uploaded_eval_file = st.file_uploader(
    "Upload a CSV file with 'code' and 'doc' columns for evaluation",
    type=['csv']
)

if uploaded_eval_file:
    with st.spinner("Evaluating model..."):
        try:
            # Read the uploaded file
            data = pd.read_csv(uploaded_eval_file)
            
            if 'code' not in data.columns or 'doc' not in data.columns:
                st.error("CSV file must contain 'code' and 'doc' columns")
            else:
                code_data = data['code'].tolist()
                doc_data = data['doc'].tolist()

                # Evaluate the model
                evaluator = CodeDocEvaluator()
                avg_bleu_score = evaluator.evaluate(code_data, doc_data)

                # Display the BLEU score
                st.subheader("Evaluation Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average BLEU Score", f"{avg_bleu_score:.4f}")
                with col2:
                    st.metric("Samples Evaluated", len(code_data))
                with col3:
                    quality = "Excellent" if avg_bleu_score > 0.7 else "Good" if avg_bleu_score > 0.5 else "Fair" if avg_bleu_score > 0.3 else "Poor"
                    st.metric("Quality Rating", quality)
                    
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")

# Footer with configuration info
with st.expander("Current Configuration"):
    st.json({
        "Model Path": config.model_config.model_path,
        "Max New Tokens": config.model_config.max_new_tokens,
        "Temperature": config.model_config.temperature,
        "Number of Beams": config.model_config.num_beams,
        "Max File Size (MB)": config.web_config.max_file_size / (1024*1024)
    })
