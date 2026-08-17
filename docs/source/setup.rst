Setup
=====

Requirements
------------

* Python 3.11+
* Git
* Ollama or a Groq API key
* Apify API token
* Docker (optional)
* n8n (optional)

Python Environment
------------------

Create a virtual environment:

.. code-block:: powershell

    python -m venv .venv

Activate it on Windows:

.. code-block:: powershell

    .venv\Scripts\Activate.ps1

Install dependencies:

.. code-block:: powershell

    pip install -r requirements.txt


Environment Variables
---------------------

Create a ``.env`` file based on ``.env.example``.

Example:

.. code-block:: text

    LLM_PROVIDER=ollama
    OLLAMA_MODEL=llama3.1
    OLLAMA_BASE_URL=http://localhost:11434

    GROQ_API_KEY=
    GROQ_MODEL=openai/gpt-oss-120b

    APIFY_API_TOKEN=


Ollama
------

Ollama can be used for local inference.

Download the model:

.. code-block:: powershell

    ollama pull qwen2.5-coder

Start Ollama and verify that it is available on:

.. code-block:: text

    http://localhost:11434


CV Setup
--------

Place the candidate CV inside:

.. code-block:: text

    data/cv/

Build the vector database:

.. code-block:: powershell

    python agent/vectorstore.py


Running the Scraper
-------------------

Run:

.. code-block:: powershell

    python scraper/scraper.py


Running the Pipeline
--------------------

Run:

.. code-block:: powershell

    python main.py


Docker
------

Build and start the application:

.. code-block:: powershell

    docker compose up -d --build

Stop the containers:

.. code-block:: powershell

    docker compose down