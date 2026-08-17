# JobHunterAgent: Automated AI Job Application Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green?style=for-the-badge)
![n8n](https://img.shields.io/badge/n8n-Orchestration-EA4B71?style=for-the-badge&logo=n8n)
[![Tests](https://github.com/pabloChantada/JobHunterAgent/actions/workflows/ci.yaml/badge.svg)](https://github.com/pabloChantada/JobHunterAgent/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/pabloChantada/JobHunterAgent/graph/badge.svg)](https://codecov.io/gh/pabloChantada/JobHunterAgent)

An end-to-end automated MLOps pipeline designed to streamline the job hunting process. The **JobHunterAgent** uses RAG (Retrieval-Augmented Generation) to evaluate job descriptions against a personal CV, score the match, and automatically generate tailored cover letters for highly compatible roles.

This project serves as a portfolio piece demonstrating production-ready MLOps practices, containerization, visual orchestration, and modern LLM application development.

## Architecture & Workflow

1. **Orchestration (n8n):** Scheduled triggers fetch new job listings, execute the Python container, and send email notifications with the results.
2. **Retrieval (ChromaDB):** The user's CV is embedded and queried to find matching skills and experiences.
3. **Evaluation (LangChain + LLM):** An agent scores the job description (0-100%) and applies exclusion rules using Pydantic Structured Outputs.
4. **Generation:** If the job is a match (e.g., Score >= 60%), a customized cover letter is automatically drafted.
5. **Reporting:** Results are logged in an Excel tracking file (`job_tracker.xlsx`) and an executive summary is emailed to the user.

## Technology Stack

* **AI Framework:** LangChain (utilizing LCEL architecture).
* **Scraping & Automation:** Apify (for job listing extraction) and n8n (for workflow orchestration).
* **Vector Store:** ChromaDB (PersistentClient).
* **Embeddings:** `sentence-transformers` (paraphrase-multilingual-MiniLM-L12-v2).
* **LLMs (Model-Agnostic):** Groq (openai/gpt-oss-120b) for high-speed cloud execution / Ollama (qwen2.5-coder) for local, zero-cost development.
* **MLOps & Infrastructure:** Docker, Github Actions, Pylint.
* **Data Processing:** Pandas, Openpyxl.

## Documentation

Full technical documentation is available on GitHub Pages:

**[Read the documentation](https://pablochantada.github.io/JobHunterAgent/)**

The documentation includes:

- **Architecture**: system architecture and data flow.
- **Setup**: installation and configuration.
- **Development**: development and extension guidelines.
- **API Reference**: automatically generated Python API documentation using Sphinx.

## Getting Started

### Prerequisites

* Python 3.10 or higher
* Docker and Docker Compose (for n8n and isolated execution)
* *(Optional)* [Ollama CLI](https://ollama.com/) installed and running locally if using local models. To configure this, run `ollama pull <model>` to download the model and add `OLLAMA_HOST` to your environment variables (Name: `OLLAMA_HOST`, Value: `0.0.0.0`).

### Installation

1. **Clone the repository:**

```bash
git clone [https://github.com/pabloChantada/JobHunterAgent.git](https://github.com/pabloChantada/JobHunterAgent.git)
cd JobHunterAgent
```

2. **Set up the virtual environment (Local Testing):**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **Upload your CV:**
   Place your personal CV in PDF format inside the `cv/` directory.
4. **Generate the ChromaDB embeddings:**

```bash
python agent/vectorstore.py
```

4. **Environment Variables:**
   Create a `.env` file in the root directory and configure your credentials following the `.env.example` template:

```env
GROQ_API_KEY="your_groq_api_key_here"
LLM_PROVIDER="groq" # or "ollama"
APIFY_API_TOKEN="your_apify_api_token_here"
```

The project it's model-agnostic, so you can switch between Groq and Ollama by changing the `LLM_PROVIDER` variable.
No current paid API intregrations are added, but can be easily integrated in the future.

### Running with Docker & n8n

For automated orchestration and containerized execution, deploy the infrastructure using Docker Compose:

```bash
docker-compose up -d --build
```

*Access the n8n visual interface at `http://localhost:5678` to monitor, configure, and trigger your workflows.*

## Roadmap

* 
* [ ] **Reporting Enhancements:** Style the Excel output (`job_tracker.xlsx`) with dynamic colors and conditional formatting using `openpyxl`.
* [ ] **Document Conversion:** Add a `.pdf` conversion pipeline for the final cover letter drafts.
* [ ] **Prompt Engineering:** Refine generation prompts to produce highly human-like, "anti-AI-cliché" cover letters.
* [ ] **LLM Evaluation Framework:** Implement automated testing metrics (e.g., LangSmith, RAGAS) to objectively measure the LLM's scoring accuracy against a dataset.
