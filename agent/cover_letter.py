"""Cover letter generation module for job applications."""
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from agent.llm_config import build_chat_model

COVER_LETTER_PROMPT = """
You are {user_name}, a direct, professional, technical AI/ML Engineer
applying for a job. Write a short cover letter for the role of {job_title}
at {company_name}, grounded strictly in MY CV.

CONTENT RULES:
- Add a brief introduction to who I am and why I'm applying, but do not
  include generic statements about being "passionate" or "thrilled".
- Only mention skills, tools, or experience that actually appear in MY CV.
  Never invent or exaggerate.
- Mention {company_name} by name at least once, naturally.
- Mention 1-2 technologies from the job posting that genuinely overlap
  with my CV: pick the strongest overlaps.
- Lead with the single most relevant project or experience for THIS
  specific posting, not a generic list.
- Use at least one concrete, specific detail from my CV (a project,
  a metric, a result) instead of a generic skill claim.
- Do not paraphrase or restate the job posting back at the reader.
  Just state the fit directly.
- End with one forward-looking sentence before the sign-off.

TONE & STYLE RULES:
- No emotional language or clichés ("thrilled", "passionate", etc.).
- Avoid starting every letter with the same template. Vary the opening.
- Direct, concise, plain sentences. No corporate filler phrases.
- Vary sentence length.
- Aim for close to 110-150 words rather than the bare minimum.
- 3 short paragraphs: (1) strongest specific match, (2) a second
  relevant point or value-add, (3) brief forward-looking close.

FORMAT:
- Plain text only — no markdown, no bullet points, no subject line.
- End with a short, plain sign-off appropriate to the job posting’s
  language, followed by {user_name}.
- Output only the letter body, ready to save. No preamble.

LANGUAGE:
- Write in the SAME language as the job posting.

JOB POSTING ({job_title} at {company_name}):
{job_description}

MY CV:
{cv_context}
"""

# New prompt for cover letter generation
cl_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", COVER_LETTER_PROMPT),
        ("human", "Redacta el mensaje de aplicación."),
    ]
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
# pylint: disable=too-many-arguments,too-many-positional-arguments
def generate_cover_letter_draft(
    job_description: str,
    cv_context: str,
    company_name: str,
    job_title: str,
    user_name: str = "Pablo Chantada",
    provider: str = None,
) -> str:
    """Generate a cover letter draft for a job application.

    Args:
        job_description: The job posting description.
        cv_context: The candidate’s CV context.
        company_name: The company name.
        job_title: The job title.
        user_name: The candidate’s name. Defaults to "Pablo Chantada".
        provider: LLM provider ('ollama' or 'groq'). Defaults to LLM_PROVIDER env var.

    Returns:
        The generated cover letter as a string.
    """
    llm_writer = build_chat_model(
        task="cover_letter",
        provider=provider,
        temperature=0.4,
        num_predict=600,  # Avoid infinite loops
        timeout=120,  # 120 seconds timeout
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        ),
    )

    # New chain for cover letter generation
    chain = cl_prompt | llm_writer | StrOutputParser()

    return chain.invoke(
        {
            "cv_context": cv_context,
            "job_description": job_description,
            "user_name": user_name,
            "company_name": company_name,
            "job_title": job_title,
        }
    )
