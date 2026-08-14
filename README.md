# JobHunterAgent: Automated AI Job Application Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green?style=for-the-badge)
![n8n](https://img.shields.io/badge/n8n-Orchestration-EA4B71?style=for-the-badge&logo=n8n)

An end-to-end automated MLOps pipeline designed to streamline the job hunting process. The **JobHunterAgent** uses RAG (Retrieval-Augmented Generation) to evaluate job descriptions against a personal CV, score the match, and automatically generate tailored cover letters for highly compatible roles.

This project serves as a portfolio piece demonstrating production-ready MLOps practices, containerization, visual orchestration, and modern LLM application development.

## Architecture & Workflow

1. **Orchestration (n8n):** Scheduled triggers fetch new job listings, execute the Python container and send email notifications with the results.
2. **Retrieval (ChromaDB):** The user's CV is embedded and queried to find matching skills and experiences.
3. **Evaluation (LangChain + LLM):** An agent scores the job description (0-100%) and applies exclusion rules using Pydantic Structured Outputs.
4. **Generation:** If the job is a match (e.g., Score >= 60%), a customized Cover Letter is generated. *(WIP)*
5. **Reporting:** Results are logged in an Excel (`job_tracker.xlsx`) and emailed to the user.

## Tech Stack

* **AI Framework:** LangChain (using LCEL architecture)
* **Vector Store:** ChromaDB (PersistentClient)
* **Embeddings:** `sentence-transformers` (Multilingual MiniLM)
* **LLMs (Model-Agnostic):** Groq (Llama 3.3 70B) for high-speed JSON parsing / Ollama (Llama 3.1 8B) for local, zero-cost execution.
* **MLOps & Infrastructure:** Docker, Docker Compose, n8n.
* **Data Manipulation:** Pandas, Openpyxl.

## Getting Started

### Prerequisites

* Python 3.10 or higher
* Docker and Docker Compose (for n8n and isolated execution)
* *(Optional)* [Ollama CLI](https://ollama.com/) installed and running locally if using local models.

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/JobHunterAgent.git
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

4. **Environment Variables:**
Create a `.env` file in the root directory and configure your keys, following the .env.example template:
```env
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=groq  # Options: "groq" or "ollama"
```

### Running with Docker & n8n

For the automated orchestration and containerized execution, use Docker Compose:

```bash
docker-compose up -d --build
```
*Access the n8n visual interface at `http://localhost:5678` to monitor and trigger your workflows.*

## Roadmap / TODOs

* [x] **State Management (Idempotency):** Implement logic in `main.py` to ensure cover letters and evaluations are only generated for *new* job offers.
* [ ] **Reporting Enhancements:** Style the Excel output (`job_tracker.xlsx`) with dynamic colors and conditional formatting using `openpyxl` after Pandas generation.
* [ ] **Document Conversion:** Add a `.pdf` conversion pipeline for the final cover letter drafts.
* [ ] **Prompt Engineering:** Refine and optimize the generation prompts to produce even more human-like, "Anti-AI-cliché" cover letters.
* [ ] **LLM Evaluation Framework:** Implement automated testing and evaluation metrics (e.g., LangSmith, RAGAS) to objectively measure the LLM's scoring accuracy against a dataset.

---

*Built by Pablo Chantada - AI/ML Engineer*