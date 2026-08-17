# 🤖 Agentic RAG Financial Intelligence Assistant

An AI-powered financial intelligence assistant that combines **Agentic AI, Retrieval-Augmented Generation (RAG), SQL analytics, and calculator tools** to answer document-based and data-driven questions.

The system uses an LLM as an intelligent router that decides when to retrieve information from the RBI Financial Inclusion document, perform calculations, or query structured sales data.

---

## 🚀 Overview

Traditional RAG systems are primarily designed for document question answering.

This project extends that approach into an **agentic system** capable of selecting between multiple tools depending on the user's question.

The assistant can:

- Retrieve information from the RBI Financial Inclusion document
- Answer questions using grounded RAG
- Perform mathematical calculations
- Query structured sales data using SQL
- Select the appropriate tool automatically
- Return document sources and page numbers for RAG responses
- Evaluate answer correctness
- Evaluate retrieval coverage
- Evaluate answer grounding
- Measure agent latency
- Run locally using Ollama or use Groq as the hosted LLM backend

---

# 🧠 System Architecture

```text
                         ┌─────────────────────┐
                         │      User Query     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Agent / LLM Router│
                         │        Groq         │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┼─────────────────┐
                  │                 │                 │
                  ▼                 ▼                 ▼
           ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
           │   RBI RAG   │   │ Calculator  │   │  SQL Tool   │
           │    Tool     │   │    Tool     │   │             │
           └──────┬──────┘   └─────────────┘   └──────┬──────┘
                  │                                    │
                  ▼                                    ▼
        ┌────────────────────┐              ┌─────────────────┐
        │ HuggingFace        │              │ SQLite Database │
        │ Embeddings         │              │ Sales Data      │
        └─────────┬──────────┘              └─────────────────┘
                  │
                  ▼
        ┌────────────────────┐
        │     Chroma DB      │
        │   Vector Store     │
        └─────────┬──────────┘
                  │
                  ▼
        ┌────────────────────┐
        │    RBI PDF         │
        │ Financial Inclusion│
        └────────────────────┘
```

---

# 🔥 Key Features

## 1. Agentic Tool Selection

The LLM determines which tool should be used based on the user's question.

Examples:

```text
"What is financial inclusion?"
                ↓
          RBI RAG Tool
```

```text
"What is 125 multiplied by 48?"
                ↓
           Calculator
```

```text
"Which product has the highest revenue?"
                ↓
             SQL Tool
```

This allows the system to behave as a **decision-making assistant rather than a simple chatbot**.

---

# 📚 2. Retrieval-Augmented Generation

The RBI Financial Inclusion document is processed and stored in a Chroma vector database.

The RAG pipeline follows:

```text
User Question
      ↓
Query Embedding
      ↓
Chroma Similarity Search
      ↓
Top-K Relevant Chunks
      ↓
Context Construction
      ↓
LLM
      ↓
Grounded Answer
```

The system uses:

* HuggingFace sentence-transformer embeddings
* `sentence-transformers/all-MiniLM-L6-v2`
* Chroma vector database
* Top-K similarity retrieval

The generated answer is instructed to use only the retrieved document context.

If the information is not available in the retrieved context, the system can respond:

```text
I don't know based on the provided document.
```

---

# 📖 3. Source Attribution

RAG responses include the source document and page numbers.

Example:

```text
Answer:
Financial inclusion is the access to formal financial services...

Sources:
- data/RBI.pdf | Page 19
- data/RBI.pdf | Page 13
- data/RBI.pdf | Page 16
- data/RBI.pdf | Page 45
```

This makes the generated answers easier to verify.

---

# 🧮 4. Calculator Tool

The agent can delegate mathematical operations to a calculator tool instead of relying on the LLM to perform arithmetic.

Example:

```text
Question:
What is 125 multiplied by 48?

Tool:
Calculator

Result:
6000
```

Another example:

```text
Question:
What is 15% of 800?

Result:
120
```

---

# 🗄️ 5. SQL Analytics Tool

The project also contains a structured sales dataset stored in SQLite.

The SQL tool allows the agent to answer analytical questions using SQL queries.

Example:

```text
Question:
Which product has the highest revenue?
```

The agent generates and executes a SQL query such as:

```sql
SELECT
    product,
    SUM(revenue) AS total_revenue
FROM sales
GROUP BY product
ORDER BY total_revenue DESC
LIMIT 1;
```

Result:

```text
product | total_revenue
Laptop  | 900000.0
```

Other supported analytical questions include:

```text
What is the total revenue?

Which category generated the most revenue?

Which product has the highest revenue?
```

---

# 🛡️ SQL Safety

The SQL tool is designed for analytical read-only operations.

The system validates SQL queries and prevents unsafe database operations such as:

```text
INSERT
UPDATE
DELETE
DROP
ALTER
CREATE
TRUNCATE
```

This prevents the agent from modifying the underlying dataset.

---

# 🤖 LLM Architecture

The project supports two LLM execution modes.

## Hosted Environment

Groq is used as the hosted LLM backend.

The project currently uses:

```text
openai/gpt-oss-20b
```

## Local Environment

For local development, Ollama can be used as the fallback LLM backend.

Example local model:

```text
phi3:mini
```

This allows the same application architecture to work both locally and in a hosted environment.

---

# ⚡ Embedding Model Caching

The vector store and embedding model are cached using:

```python
@lru_cache(maxsize=1)
```

This prevents the embedding model from being repeatedly initialized for every question within the same application process.

Instead of:

```text
Question 1 → Load embedding model
Question 2 → Load embedding model
Question 3 → Load embedding model
```

the application follows:

```text
Application Start
       ↓
Load embedding model once
       ↓
Cache
       ↓
Question 1 → Reuse
Question 2 → Reuse
Question 3 → Reuse
```

This reduces unnecessary initialization overhead.

---

# 🧪 Evaluation Framework

The project contains multiple evaluation layers rather than relying only on whether the chatbot "looks correct."

## 1. Tool Selection Evaluation

Measures whether the correct tool is selected for a question.

Example:

```text
Financial Inclusion Question
        ↓
Expected: RBI RAG
        ↓
Agent: RBI RAG
        ↓
PASS
```

Tools evaluated include:

* RBI RAG
* Calculator
* SQL
* No tool

---

## 2. Answer Accuracy Evaluation

Expected keywords are defined for benchmark questions.

Example:

```text
Question:
Which product has the highest revenue?

Expected:
Laptop
900000
```

The evaluation system checks whether the generated answer contains the expected information.

---

## 3. Retrieval Evaluation

The project separately evaluates whether relevant information is present in the retrieved context.

The retrieval benchmark currently contains five test questions covering:

* Financial inclusion
* Economic growth
* Customer protection
* Strategic objectives
* Challenges to financial inclusion

### Current verified result

```text
Average Retrieval Term Coverage: 100%
```

This result is based on the current five-question retrieval benchmark.

---

## 4. Grounding Evaluation

The project also evaluates whether generated answers are supported by retrieved document context.

Example verified test:

```text
Question:
What is financial inclusion?

Grounded: True
Grounding Score: 100.00%
Supported Sentences: 1
Total Sentences: 1
```

The grounding evaluation helps identify cases where an answer may contain information that is not sufficiently supported by the retrieved context.

---

## 5. Latency Evaluation

Each evaluation run measures:

```text
Question processing time
```

and calculates:

```text
Average Latency
```

This allows future optimization of:

* Retrieval
* LLM inference
* Tool execution
* Agent loops
* Embedding initialization

---

# 📊 Evaluation Results

Current verified benchmark results:

| Metric                  | Result |
| ------------------------ | -----: |
| Retrieval Term Coverage |   100% |
| Verified Grounding Test |   100% |
| SQL Tool Direct Test    |   PASS |
| Calculator Tool Test    |   PASS |
| RBI RAG Tool Test       |   PASS |

The final end-to-end agent evaluation metrics are intentionally not hard-coded here until the complete evaluation can be rerun under a clean LLM quota.

This avoids reporting stale or incomplete benchmark results.

---

# 💬 Example Queries

## RBI RAG

```text
What is financial inclusion?
```

```text
What are the strategic objectives of financial inclusion in India?
```

```text
What is the role of customer protection in financial inclusion?
```

---

## Calculator

```text
What is 125 multiplied by 48?
```

```text
What is 15% of 800?
```

---

## SQL Analytics

```text
Which product has the highest revenue?
```

```text
What is the total revenue?
```

```text
Which category generated the most revenue?
```

---

# 🧩 Project Structure

```text
rag-project/
│
├── agent.py
├── query.py
├── evaluation.py
├── grounding.py
├── retrieval_evaluation.py
├── test_agent.py
│
├── data/
│   └── RBI.pdf
│
├── chroma_db/
│
├── requirements.txt
│
├── app.py
│
└── README.md
```

> The exact project structure may evolve as additional evaluation and deployment components are added.

---

# 🛠️ Tech Stack

## Programming

* Python

## AI / LLM

* Groq
* Ollama
* LangChain
* Large Language Models

## RAG

* HuggingFace Embeddings
* Sentence Transformers
* Chroma
* Retrieval-Augmented Generation

## Data

* SQLite
* SQL
* Pandas

## Application

* Streamlit

## Deployment

* Render

## Evaluation

* Retrieval evaluation
* Grounding evaluation
* Tool selection evaluation
* Answer accuracy evaluation
* Latency measurement

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/ms1706g/rbi-financial-inclusion-rag.git
```

Move into the project:

```bash
cd rbi-financial-inclusion-rag
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

For hosted Groq inference, configure:

```text
GROQ_API_KEY=your_groq_api_key
```

Do not commit API keys or other secrets to GitHub.

A `.env` file should be excluded from version control.

---

# ▶️ Running the Agent

Run the agent:

```bash
python agent.py
```

The assistant accepts questions interactively.

Example:

```text
You: What is financial inclusion?

Assistant:
Financial inclusion refers to access to formal financial services...
```

---

# 🧪 Running Tests

## Agent Test

```bash
python test_agent.py
```

## Evaluation

```bash
python evaluation.py
```

## Retrieval Evaluation

```bash
python retrieval_evaluation.py
```

## Grounding Evaluation

```bash
python grounding.py
```

---

# 🌐 Running the Streamlit Application

Start the application with:

```bash
streamlit run app.py
```

The Streamlit interface provides an interactive interface for querying the agent.

---

# 🚀 Deployment

The application can be deployed to a hosting platform such as Render.

The hosted architecture uses:

```text
Streamlit Application
        ↓
Agent
        ↓
Groq LLM
        ↓
Tools
        ↓
RAG / SQL / Calculator
```

Environment variables such as `GROQ_API_KEY` should be configured through the hosting platform's secret/environment-variable settings.

---

# 🔍 Engineering Highlights

This project focuses on several practical AI engineering concepts:

### Agentic Decision Making

The LLM decides whether a question requires:

* Document retrieval
* Calculation
* SQL analytics
* No external tool

### Grounded Generation

RAG answers are constrained by retrieved document context.

### Tool-Based Architecture

Instead of forcing one model to solve every task, specialized tools handle:

* Retrieval
* Arithmetic
* Structured data analysis

### Evaluation-Driven Development

The system includes dedicated evaluation for:

* Tool selection
* Answer correctness
* Retrieval coverage
* Grounding
* Latency

### Safety

The SQL tool restricts database operations to safe analytical queries.

### Performance

Embedding/vectorstore caching avoids unnecessary repeated initialization during the application's lifetime.

---

# 📌 Current Limitations

The current evaluation benchmark is relatively small and uses manually defined test cases.

Retrieval evaluation currently measures **term coverage within retrieved context**, rather than a formal IR benchmark such as precision@k or recall@k.

Agent latency also depends on:

* LLM provider
* Model load
* Network conditions
* Retrieval time
* Tool execution time

Future versions can introduce a larger evaluation dataset and more formal retrieval metrics.

---

# 🔮 Future Improvements

Potential improvements include:

* Larger automated evaluation datasets
* Formal retrieval metrics such as Recall@K and MRR
* Reranking retrieved documents
* Hybrid keyword + vector search
* Query rewriting
* Better conversational memory
* Streaming responses
* More SQL analytical capabilities
* Agent tracing and observability
* Prompt/version tracking
* Automated regression testing
* More robust multi-tool planning
* Model fallback and retry strategies
* Evaluation dashboards

---

# 🎯 Why This Project?

The goal was to move beyond a basic chatbot or single-purpose RAG pipeline and build a small **decision-oriented AI system**.

The assistant demonstrates the ability to:

```text
Understand a question
        ↓
Decide whether a tool is needed
        ↓
Select the appropriate tool
        ↓
Retrieve / calculate / query
        ↓
Generate a grounded response
        ↓
Evaluate the result
```

This architecture reflects practical AI engineering workflows involving:

**Data → Retrieval → Models → Agents → Tools → Applications → Evaluation**

---

# 👨‍💻 Author

**Manan Shiva**

Built as an AI Engineering / Agentic RAG project focused on practical LLM application development, retrieval systems, tool use, SQL analytics, and evaluation.