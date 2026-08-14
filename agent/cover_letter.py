import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

COVER_LETTER_PROMPT = """
You are {user_name}, a direct, professional, technical AI/ML Engineer applying for a job.
Write a short cover letter for the role of {job_title} at {company_name}, grounded strictly in MY CV.

CONTENT RULES:
- Add a brief introduction to who I am and why I'm applying, but do not include generic statements about being "passionate" or "thrilled" to apply.
- Only mention skills, tools, or experience that actually appear in MY CV. Never invent or exaggerate.
- Mention {company_name} by name at least once, naturally not just as a header.
- Mention 1-2 technologies from the job posting that genuinely overlap with my CV  pick the strongest overlaps, not just the first ones listed.
- Lead with the single most relevant project or experience for THIS specific posting, not a generic list of past roles. Vary which project/detail you foreground based on what this posting actually asks for.
- Use at least one concrete, specific detail from my CV (a project, a metric, a result) instead of a generic skill claim.
- Do not paraphrase or restate the job posting back at the reader ("I see you're looking for..."). Just state the fit directly.
- End with one forward-looking sentence (e.g. what you could contribute, or interest in discussing further) before the sign-off do not end on a bare list of skills.

TONE & STYLE RULES:
- No emotional language or clichés ("thrilled", "passionate", "entusiasmado", "me dirijo a usted", "tapestry", "delve", "leverage").
- Avoid starting every letter with the same template, e.g. "My experience as an AI Engineer at [company] allowed me to develop..." / "Mi experiencia como Ingeniero de IA en X me permitió...". Vary the opening: start from a project, a result, or a direct statement of fit instead.
- Direct, concise, plain sentences. No corporate filler phrases like "se alinean con los requisitos del puesto" / "aligns with the requirements of the role".
- Vary sentence length; avoid a string of same-length sentences, which reads as generated.
- Aim for close to the word limit (roughly 110-150 words) rather than the bare minimum — a two-sentence letter reads as low-effort.
- 3 short paragraphs: (1) strongest specific match, (2) a second relevant point or concrete value-add, (3) brief forward-looking close.

FORMAT:
- Plain text only — no markdown, no bullet points, no subject line.
- End with a short, plain sign-off appropriate to the job posting's language (e.g. "Un saludo," or "Best regards,") followed by {user_name}.
- Output only the letter body, ready to save as-is. No preamble like "Here's your cover letter:".

LANGUAGE:
- Write in the SAME language as the job posting.

JOB POSTING ({job_title} at {company_name}):
{job_description}

MY CV:
{cv_context}
"""

# New prompt for cover letter generation
cl_prompt = ChatPromptTemplate.from_messages([
    ("system", COVER_LETTER_PROMPT),
    ("human", "Redacta el mensaje de aplicación.")
])

# Docorator to retry in case the API fails
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def generate_cover_letter_draft(job_description: str, cv_context: str, company_name: str,
                                  job_title: str, user_name: str = "Pablo Chantada", provider: str = None) -> str:
    """Generate the cover letter in text format."""
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    
    # New llm instance with temperature for creative writing, not a lot to avoid hallucinations
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        llm_writer = ChatOllama(model="llama3.1", 
                                temperature=0.4,
                                num_predict=600, # Avoid infinite loops in case of a misbehaving prompt
                                timeout=120, # 30 seconds timeout for the request
                                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")) 
    elif provider == "groq":
        llm_writer = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)
        
    # New chain for cover letter generation
    chain = cl_prompt | llm_writer | StrOutputParser()
    
    return chain.invoke({
        "cv_context": cv_context, 
        "job_description": job_description,
        "user_name": user_name,
        "company_name": company_name,
        "job_title": job_title
    })