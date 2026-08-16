Architecture
============

Overview
--------

JobHunterAgent is an automated job-search pipeline that combines job
scraping, profile filtering, Retrieval-Augmented Generation (RAG),
structured LLM evaluation, cover-letter generation and automated reporting.

High-Level Architecture
-----------------------

.. code-block:: text

    ChromaDB Embeddings
     |
     v
    n8n
     |
     v
    Job Scraper
     |
     v
    Profile Filtering
     |
     v
    Deduplication
     |
     v
    CV RAG (retrieval not storing)
     |
     v
    LLM Evaluation
     |
     +--------> Rejected
     |
     v
    Cover Letter Generation
     |
     v
    Excel Reporting and Tracking


Job Scraping
------------

The scraper currently uses Apify to retrieve LinkedIn job postings.

The scraper normalizes the results into the ``JobOffer`` Pydantic model.

The ``JobScraper`` abstract interface allows additional job sources to be
implemented without changing the rest of the pipeline.

RAG Pipeline
------------

The candidate CV is processed and stored in ChromaDB.

The pipeline performs:

1. PDF text extraction.
2. Text normalization.
3. Section detection.
4. Text chunking.
5. Multilingual embedding generation.
6. Persistent vector storage.
7. Semantic retrieval.

The retrieved CV context is passed to the LLM evaluator.

LLM Evaluation
--------------

The evaluator compares the retrieved CV context with the job description.

The output is validated using a Pydantic model containing:

* A numerical score.
* An application verdict.
* Three evaluation reasons.

Cover Letter Generation
-----------------------

Accepted job offers are passed to the cover-letter generation pipeline.

The generated letter is grounded in the retrieved CV context and the
original job description.

Reporting
---------

The final results are stored in Excel files for tracking and later review.
Both daily and all-time reports.

n8n orchestrates the complete process and can execute the pipeline on a
schedule.