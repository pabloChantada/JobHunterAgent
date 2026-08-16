Development
===========

Project Structure
-----------------

The main application components are:

.. code-block:: text

    agent/
        agent.py
        cover_letter.py
        llm_config.py
        vectorstore.py

    scraper/
        scraper.py

    n8n/
        workflow.json

    main.py


Adding a New Job Source
-----------------------

New job sources should implement the ``JobScraper`` interface.

For example:

.. code-block:: python

    class IndeedScraper(JobScraper):

        source_name = "indeed"

        def scrape(self) -> list[JobOffer]:
            ...


The scraper should return normalized ``JobOffer`` objects.


Adding an LLM Provider
----------------------

LLM configuration is centralized in:

.. code-block:: text

    agent/llm_config.py

The rest of the application should interact with the provider abstraction
rather than directly instantiating provider-specific models.


Testing Changes
---------------

Before committing changes, rebuild the Sphinx documentation:

.. code-block:: powershell

    sphinx-build -b html docs/source docs/build/html

Then run the application locally:

.. code-block:: powershell

    python main.py


Documentation
-------------

Python modules should contain docstrings for public classes and functions.

Example:

.. code-block:: python

    def generate_cover_letter_draft(
        job_description: str,
        cv_context: str,
        company_name: str,
        job_title: str,
    ) -> str:
        """Generate a cover letter draft.

        Args:
            job_description: Job posting description.
            cv_context: Retrieved CV context.
            company_name: Company name.
            job_title: Job title.

        Returns:
            Generated cover letter.
        """