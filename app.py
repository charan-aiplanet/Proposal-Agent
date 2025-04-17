import streamlit as st
import os
import tempfile
import base64
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from docx import Document
from weasyprint import HTML
import markdown
import io
import uuid
import shutil
from PIL import Image
import PyPDF2
import fitz  # PyMuPDF

# Set page configuration
st.set_page_config(page_title="AI Proposal Generator", layout="wide")

# Initialize session state variables if they don't exist
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None
if "knowledge_files" not in st.session_state:
    st.session_state.knowledge_files = []
if "extracted_images" not in st.session_state:
    st.session_state.extracted_images = []
if "proposal_template" not in st.session_state:
    st.session_state.proposal_template = """
# Proposal

## Introduction
[Brief introduction about the company]

## Scope/Objectives
[Define the scope and objectives of the project]

## Proposal/Approach
[Detail the approach to meet the objectives]

## Proposed Workflow
```mermaid
graph TD
    A[Requirement Analysis] --> B[Design]
    B --> C[Development]
    C --> D[Testing]
    D --> E[Deployment]
    E --> F[Support]
```

## Deliverables from Client
[List what is expected from the client]

## Timelines
[Project timeline with milestones]

## Commercials
[Cost breakdown and payment terms]

## About AI Planet
[Information about AI Planet]
"""
if "requirements" not in st.session_state:
    st.session_state.requirements = ""
if "generated_proposal" not in st.session_state:
    st.session_state.generated_proposal = ""
if "template_option" not in st.session_state:
    st.session_state.template_option = "Use Default"

# Create a directory for storing images
IMAGE_DIR = "extracted_images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Helper functions
def extract_text_and_images_from_pdf(file_path):
    """Extract text and images from PDF file"""
    extracted_text = ""
    image_paths = []

    try:
        # Extract text using PyPDF2
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text() + "\n\n"

        # Extract images using PyMuPDF
        try:
            # Open the PDF with PyMuPDF
            pdf_document = fitz.Document(file_path)  # Use Document instead of open

            # Iterate through pages
            for page_num, page in enumerate(pdf_document):
                # Extract images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Generate a unique filename
                    image_filename = f"{IMAGE_DIR}/pdf_image_{uuid.uuid4().hex}.png"

                    # Save the image
                    with open(image_filename, "wb") as img_file:
                        img_file.write(image_bytes)

                    image_paths.append(image_filename)

            pdf_document.close()
        except Exception as e:
            # Fallback: just use PyPDF2 for text if PyMuPDF fails
            st.warning(f"Image extraction failed: {str(e)}. Only text will be extracted.")

    except Exception as e:
        st.error(f"Error extracting from PDF: {str(e)}")

    return extracted_text, image_paths

def extract_text_and_images_from_docx(file_path):
    """Extract text and images from DOCX file"""
    # Extract text
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    extracted_text = "\n\n".join([doc.page_content for doc in docs])

    # Extract images
    image_paths = []
    try:
        doc = Document(file_path)

        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract all the shapes (including images)
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    # Save the image
                    image_filename = f"{IMAGE_DIR}/docx_image_{uuid.uuid4().hex}.png"
                    with open(image_filename, "wb") as img_file:
                        img_file.write(rel.target_part.blob)
                    image_paths.append(image_filename)
    except Exception as e:
        st.warning(f"Image extraction from DOCX failed: {str(e)}. Only text will be extracted.")

    return extracted_text, image_paths

def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file based on file type"""
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        file_path = tmp_file.name

    try:
        # Use the appropriate loader based on file type
        if file_path.endswith(".pdf"):
            extracted_text, image_paths = extract_text_and_images_from_pdf(file_path)
            st.session_state.extracted_images.extend(image_paths)
        elif file_path.endswith(".docx"):
            extracted_text, image_paths = extract_text_and_images_from_docx(file_path)
            st.session_state.extracted_images.extend(image_paths)
        else:
            raise ValueError("Only PDF and DOCX files are supported")

        # Clean up the temporary file
        os.unlink(file_path)
        return extracted_text
    except Exception as e:
        os.unlink(file_path)  # Clean up on error
        raise e

def create_kb_from_texts(texts):
    """Create knowledge base from list of texts"""
    combined_text = "\n\n".join(texts)
    # Create text chunks for the vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(combined_text)

    # Create embeddings using Hugging Face
    embeddings = HuggingFaceEmbeddings(model_name=selected_embedding_model)
    return FAISS.from_texts(chunks, embeddings)

def convert_md_to_docx(md_text, output_filename, images=None):
    """Convert markdown to DOCX with images"""
    doc = Document()
    # Simple parsing of markdown headings and paragraphs
    lines = md_text.split('\n')

    # Add any available logos at the top
    if images:
        for img_path in images[:2]:  # Limit to first 2 images which are likely logos
            try:
                doc.add_picture(img_path, width=2000000)  # ~2 inches wide
            except Exception as e:
                pass  # Skip if image can't be added

    for line in lines:
        line = line.strip()
        if line.startswith('# '):
            doc.add_heading(line[2:], level=1)
        elif line.startswith('## '):
            doc.add_heading(line[3:], level=2)
        elif line.startswith('### '):
            doc.add_heading(line[4:], level=3)
        elif line.startswith('```') and line != '```':
            # Skip code blocks for now
            continue
        elif '```' in line:
            # Skip code blocks
            continue
        elif line:
            doc.add_paragraph(line)

    # Save the document
    doc.save(output_filename)
    return output_filename

def convert_md_to_pdf(md_text, output_filename, images=None):
    """Convert markdown to PDF with images"""
    # Convert markdown to HTML
    html = markdown.markdown(md_text, extensions=['extra', 'codehilite'])

    # Add image tags if images are available
    image_tags = ""
    if images:
        for img_path in images[:2]:  # Limit to first 2 images which are likely logos
            try:
                # Convert to base64 for embedding in HTML
                with open(img_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    img_ext = os.path.splitext(img_path)[1][1:]
                    image_tags += f'<img src="data:image/{img_ext};base64,{img_data}" style="max-width: 200px; margin: 10px 0;">'
            except Exception as e:
                pass  # Skip if image can't be processed

    styled_html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2cm; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; }}
            h3 {{ color: #555; }}
            code {{ background-color: #f8f8f8; padding: 2px 4px; }}
        </style>
    </head>
    <body>
        {image_tags}
        {html}
    </body>
    </html>
    """
    HTML(string=styled_html).write_pdf(output_filename)
    return output_filename

# Set up sidebar with API keys and model selection
st.sidebar.title("Configuration")

# Set up API key input for Groq
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
os.environ["GROQ_API_KEY"] = groq_api_key

# Set up embedding model selection
embedding_model_options = [
    "sentence-transformers/all-MiniLM-L6-v2",  # Fast and lightweight
    "sentence-transformers/all-mpnet-base-v2",  # Better quality
    "BAAI/bge-small-en-v1.5"  # Another good option
]
selected_embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    embedding_model_options,
    index=0
)

# Groq LLM model selection
groq_model_options = [
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229"
]
selected_llm_model = st.sidebar.selectbox(
    "Select Groq LLM Model",
    groq_model_options,
    index=0
)

# Main UI
st.title("🤖 AI Proposal Generator")

# Create columns for main layout
col1, col2 = st.columns([1, 1])

with col1:
    # Requirements section
    st.header("1️⃣ Project Requirements")

    # Input methods
    input_method = st.radio("Choose input method:", ["Text Input", "File Upload"], horizontal=True, key="req_input_method")

    if input_method == "Text Input":
        new_requirements = st.text_area(
            "Enter the project requirements:",
            height=150,
            value=st.session_state.requirements
        )
        # Update session state when text changes
        if new_requirements != st.session_state.requirements:
            st.session_state.requirements = new_requirements
    else:
        uploaded_file = st.file_uploader("Upload requirements document", type=["pdf", "docx"], key="req_file")
        if uploaded_file:
            try:
                st.session_state.requirements = extract_text_from_file(uploaded_file)
                st.success(f"Successfully extracted requirements from {uploaded_file.name}")

                # Show the extracted text
                with st.expander("View Extracted Requirements"):
                    st.text_area("Extracted content:", value=st.session_state.requirements, height=150)
            except Exception as e:
                st.error(f"Error extracting text: {str(e)}")

    # Knowledge base section
    st.header("2️⃣ Knowledge Base")

    # Allow uploading multiple files
    kb_files = st.file_uploader("Upload knowledge base documents", type=["pdf", "docx"], accept_multiple_files=True, key="kb_files")

    # Add files to knowledge base list
    if kb_files:
        new_files = [f for f in kb_files if f.name not in [kf.name for kf in st.session_state.knowledge_files]]
        st.session_state.knowledge_files.extend(new_files)

    # Display existing knowledge base files
    if st.session_state.knowledge_files:
        st.write("Knowledge Base Files:")
        cols = st.columns(3)
        for i, file in enumerate(st.session_state.knowledge_files):
            col_idx = i % 3
            with cols[col_idx]:
                st.write(f"📄 {file.name}")

    # Option to clear knowledge base
    if st.session_state.knowledge_files and st.button("Clear Knowledge Base"):
        st.session_state.knowledge_files = []
        st.session_state.knowledge_base = None
        st.session_state.extracted_images = []
        # Clean up image directory
        for filename in os.listdir(IMAGE_DIR):
            file_path = os.path.join(IMAGE_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                st.warning(f"Error deleting {file_path}: {e}")
        st.success("Knowledge base cleared.")

    # Create knowledge base button
    if st.button("Create Knowledge Base"):
        if not st.session_state.knowledge_files and not st.session_state.requirements:
            st.warning("Please add requirements or upload knowledge base files first.")
        else:
            # Show progress message
            with st.spinner(f"Creating knowledge base with {selected_embedding_model}..."):
                try:
                    # Extract text from all knowledge base files
                    kb_texts = []

                    # Add requirements if available
                    if st.session_state.requirements:
                        kb_texts.append(st.session_state.requirements)

                    # Add content from all files
                    for file in st.session_state.knowledge_files:
                        try:
                            file_content = extract_text_from_file(file)
                            kb_texts.append(file_content)
                        except Exception as e:
                            st.warning(f"Could not process {file.name}: {str(e)}")

                    # Create the vector database knowledge base
                    st.session_state.knowledge_base = create_kb_from_texts(kb_texts)
                    st.success("✅ Knowledge base created successfully!")
                except Exception as e:
                    st.error(f"Error creating knowledge base: {str(e)}")

    # Proposal template section
    st.header("3️⃣ Proposal Template")

    template_options = ["Use Default", "Edit Template", "Upload Template"]
    st.session_state.template_option = st.radio("Choose template method:", template_options, horizontal=True, key="template_method")

    if st.session_state.template_option == "Use Default":
        with st.expander("View Default Template"):
            st.code(st.session_state.proposal_template, language="markdown")
    elif st.session_state.template_option == "Edit Template":
        new_template = st.text_area(
            "Edit the proposal template below:",
            value=st.session_state.proposal_template,
            height=250
        )
        # Update session state when template changes
        if new_template != st.session_state.proposal_template:
            st.session_state.proposal_template = new_template
    else:  # Upload Template
        uploaded_template = st.file_uploader("Upload proposal template", type=["docx", "pdf"], key="template_file")
        if uploaded_template:
            try:
                content = extract_text_from_file(uploaded_template)
                st.session_state.proposal_template = content
                with st.expander("Template Preview"):
                    st.text_area("Template Content:", value=content, height=200)
            except Exception as e:
                st.error(f"Error extracting template: {str(e)}")

with col2:
    # Generation and results section
    st.header("4️⃣ Generate & Edit Proposal")

    # Generate proposal button
    if st.button("🚀 Generate Proposal", use_container_width=True):
        if not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar.")
        elif not st.session_state.requirements:
            st.warning("Please enter project requirements first.")
        elif not st.session_state.knowledge_base and st.session_state.knowledge_files:
            st.warning("Please create a knowledge base first.")
        else:
            try:
                with st.spinner(f"Generating proposal with {selected_llm_model}..."):
                    # Set up the Groq LLM
                    llm = ChatGroq(
                        model=selected_llm_model,
                        temperature=0.2,
                        api_key=groq_api_key
                    )

                    # Get the requirements from session state
                    requirements = st.session_state.requirements
                    

                    # Get relevant context from knowledge base if it exists
                    context = ""
                    if st.session_state.knowledge_base:
                        retriever = st.session_state.knowledge_base.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 5}
                        )
                        context_docs = retriever.get_relevant_documents(requirements)
                        context = "\n\n".join([doc.page_content for doc in context_docs])
                    print("context", context)
                    # Get the proposal template
                    proposal_template = st.session_state.proposal_template

                    # Set up the prompt template - FIX: Include input variable in prompt template
                    
                    system_template = """
                    You are a professional proposal writer tasked with creating a detailed business proposal.

                    # REQUIREMENTS
                    {requirements}

                    # RETRIEVED CONTEXT
                    {context}

                    # PROPOSAL TEMPLATE
                    {proposal_template}

                    Based on the requirements and using the provided knowledge base context, generate a complete proposal
                    that strictly follows the proposal template format. Be specific, professional, and thorough.
                    Fill in all sections with relevant information based on the requirements. Use the knowledge base to
                    provide accurate details. If any information is missing, make reasonable assumptions that would benefit
                    a business proposal.

                    YOUR PROPOSAL:
                    """

                    # Create the prompt with proper input variables
                    from langchain_core.prompts import ChatPromptTemplate
                    prompt = ChatPromptTemplate.from_template(system_template)
                    chain = prompt | llm
                    result = chain.invoke({"requirements":requirements, 
                                  "context" : context,
                                  "proposal_template" : proposal_template})
                    st.session_state.generated_proposal = result.content

                    # If we have a knowledge base, use the retrieval chain
                    

            except Exception as e:
                st.error(f"Error generating proposal: {str(e)}")

    # Display and edit generated proposal
    if st.session_state.generated_proposal:
        st.subheader("Generated Proposal")

        # Editable proposal
        edited_proposal = st.text_area(
            "Edit your proposal as needed:",
            value=st.session_state.generated_proposal,
            height=400
        )
        # Update session state when proposal changes
        if edited_proposal != st.session_state.generated_proposal:
            st.session_state.generated_proposal = edited_proposal

        # Preview in markdown
        with st.expander("Proposal Preview"):
            st.markdown(edited_proposal)

        # Create temporary files for download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as md_file:
            md_file.write(edited_proposal.encode())
            md_path = md_file.name

        docx_path = md_path.replace(".md", ".docx")
        pdf_path = md_path.replace(".md", ".pdf")

        try:
            # Convert to DOCX and PDF with extracted images
            convert_md_to_docx(edited_proposal, docx_path, st.session_state.extracted_images)
            convert_md_to_pdf(edited_proposal, pdf_path, st.session_state.extracted_images)

            # Create download buttons
            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    label="Download as DOCX",
                    data=open(docx_path, "rb").read(),
                    file_name="generated_proposal.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

            with col2:
                st.download_button(
                    label="Download as PDF",
                    data=open(pdf_path, "rb").read(),
                    file_name="generated_proposal.pdf",
                    mime="application/pdf"
                )

            # Clean up temporary files
            os.unlink(md_path)
            os.unlink(docx_path)
            os.unlink(pdf_path)

        except Exception as e:
            st.error(f"Error preparing downloads: {str(e)}")
            # Clean up on error
            if os.path.exists(md_path):
                os.unlink(md_path)
            if os.path.exists(docx_path):
                os.unlink(docx_path)
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

# Add helpful information in the sidebar
st.sidebar.title("Help")
st.sidebar.info("""
### How to use this app:
1. Enter project requirements via text or file upload
2. Upload PDF/DOCX documents to build your knowledge base
3. Create the knowledge base using the button
4. Choose a proposal template (default, edit, or upload)
5. Click "Generate Proposal" to create your proposal
6. Edit the generated proposal as needed
7. Download in your preferred format (DOCX or PDF)
""")

st.sidebar.title("About")
st.sidebar.info("""
This app uses:
- LangChain with Hugging Face embeddings for the vector database
- Groq for fast, high-quality text generation
- Streamlit for the interactive interface
- PyMuPDF and python-docx for document processing and image extraction

To get a Groq API key, sign up at https://console.groq.com/   Updated one at 23:44
""")
