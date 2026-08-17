# RBI Financial Inclusion RAG

A document-grounded Retrieval-Augmented Generation (RAG) application that allows users to ask questions about the RBI's National Strategy for Financial Inclusion 2019-2024.

The system retrieves relevant sections from the source document and uses a local LLM to generate grounded answers with document page references.

## Features

- PDF document ingestion
- Recursive text chunking
- Semantic embeddings using Sentence Transformers
- Chroma vector database for similarity search
- Top-k retrieval
- Local LLM inference using Ollama and Phi-3 Mini
- Context-grounded responses
- Hallucination fallback for unsupported questions
- Retrieved source/page references
- Streamlit web interface
- Retrieval evaluation across representative questions

## Architecture

RBI PDF
   |
   v
PDF Loader
   |
   v
Text Chunking
(800 chunk size / 100 overlap)
   |
   v
Embedding Model
all-MiniLM-L6-v2
   |
   v
Chroma Vector Database
   |
   v
User Question
   |
   v
Semantic Search
Top-k = 4
   |
   v
Retrieved Context
   |
   v
Phi-3 Mini
   |
   v
Grounded Answer
   |
   v
Retrieved Sources


## Tech Stack

- Python
- LangChain
- Sentence Transformers
- ChromaDB
- Ollama
- Phi-3 Mini
- Streamlit
- PyPDF


## Document Processing

The RBI document contains 50 pages.

The initial ingestion pipeline generated 135 chunks using:

- Chunk size: 800 characters
- Chunk overlap: 100 characters

A minimum-length filtering step was then introduced to remove very small/noisy chunks, resulting in 131 chunks.

Embeddings are generated using:

sentence-transformers/all-MiniLM-L6-v2

The resulting embedding vectors have 384 dimensions.


## Retrieval Experiment

Two retrieval configurations were evaluated:

- k = 2
- k = 4

A small evaluation set covering definitions, economic importance, challenges, customer protection, and strategic objectives was used.

For the query:

"What are the strategic objectives of financial inclusion in India?"

the k=2 configuration initially retrieved the table of contents and a generic chapter summary.

After increasing k to 4, the system retrieved the actual Strategic Objectives section from page 29.

Based on this experiment, k=4 was selected as the default retrieval configuration.


## Grounding

The application instructs the LLM to answer only from the retrieved document context.

If the requested information cannot be found in the provided document, the system responds:

"I don't know based on the provided document."

This behavior was tested using an unrelated question about the capital of France.


## Project Structure

rag-project/
|
├── data/
│   └── RBI.pdf
|
├── chroma_db/
|
├── ingest.py
├── query.py
├── evaluation.py
├── main.py
├── app.py
├── requirements.txt
└── README.md


## File Responsibilities

### ingest.py

Loads the PDF, splits the document into chunks, generates embeddings, and creates the Chroma vector database.

### query.py

Performs similarity retrieval, constructs the grounded prompt, generates the answer using Ollama, and returns retrieved source metadata.

### evaluation.py

Runs retrieval experiments across representative questions and different top-k configurations.

### main.py

Provides a command-line interface for querying the RAG system.

### app.py

Provides the Streamlit web interface.


## Running Locally

### 1. Activate the virtual environment

.\venv312\Scripts\Activate.ps1


### 2. Install dependencies

pip install -r requirements.txt


### 3. Pull the Ollama model

ollama pull phi3:mini


### 4. Build the vector database

python ingest.py


### 5. Run the CLI application

python main.py


### 6. Run the Streamlit application

streamlit run app.py


## Example

Question:

What are the strategic objectives of financial inclusion in India?

The system retrieves relevant sections from the RBI document and generates a concise answer using the retrieved context.

Retrieved sources include the corresponding RBI document pages.


## Limitations

- The current system works with a single PDF document.
- Retrieval quality depends on document chunking and embedding quality.
- The local Phi-3 Mini model has limited reasoning capability compared with larger hosted models.
- Source references currently represent retrieved chunks rather than sentence-level citations.
- Evaluation is currently based on a small manually curated question set.


## Future Improvements

- Support multiple documents
- Hybrid keyword and semantic retrieval
- Reranking
- Automated retrieval evaluation
- Improved citation attribution
- Conversation history
- Document upload through the UI
- Cloud deployment