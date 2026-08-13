"""
This DB stores the vector embeddings of the CVs (English and Spanish),
chunked by section for precise retrieval.
"""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
import re
import unicodedata
from langdetect import detect

PATH_CV = Path("data/cv")
PERSIST_DIR = "data/chroma_cv"
SECTION_PATTERN = re.compile(
    r"^\s*(PERFIL|PROFILE|CONOCIMIENTOS T.*?CNICOS|TECHNICAL SKILLS|"
    r"EXPERIENCIA|EXPERIENCE|PROYECTOS|PROJECTS|EDUCACI.*?N|FORMACI.*?N|EDUCATION|IDIOMAS|LANGUAGES)\s*$",
    re.MULTILINE | re.IGNORECASE
)


# Embedding model for CVs, supporting both English and Spanish.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Initialize the Chroma vector store for CV embeddings
vectorstore = Chroma(
    collection_name="cv_embeddings",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

# Defines the chunking strategy for the CVs, prioritizing section/paragraph breaks.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". "],  
)

# Need an especific method for .pdf's
# NOTE: pypdf has problems with accents and special characters
def extract_text(pdf_path: Path) -> str:
    doc = pymupdf.open(pdf_path)
    raw_text =  "\n".join(page.get_text() or "" for page in doc)

    # Fix simbols and language issues with unicode
    normalized_text = unicodedata.normalize("NFKC", raw_text)
    return normalized_text

def chunk_by_sections(text: str) -> dict:
    sections = {}
    matches = list(SECTION_PATTERN.finditer(text))
    
    # Filter out the header, ussualy contact info, name, etc.
    if matches:
        sections["CONTACTO_HEADER"] = text[:matches[0].start()].strip()
        
    for i, match in enumerate(matches):
        section_name = match.group(1).strip().upper()
        start_idx = match.end()
        # The end index is either the start of the next match or the end of the text
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
        
        sections[section_name] = text[start_idx:end_idx].strip()
        
    return sections

# The convention we'll use is *_ES.pdf for Spanish CVs and *.pdf for English CVs.
# If other languages are needed, we can extend this function to detect them based on filename patterns or content.
def detect_language(filename: str) -> str:
    return "spanish" if filename.endswith("_ES.pdf") else "english"


def add_cv_to_db(cv_path: Path):
    language = detect_language(cv_path.name)
    text = extract_text(cv_path)
    
    # Obtain text by section: {PROFILE: text, EXPERIENCE: text, ...}
    sections = chunk_by_sections(text)
    
    final_chunks = []
    metadatas = []
    
    for section_name, section_content in sections.items():
        if not section_content:
            continue

        # Split first and then tag the chunks        
        sub_chunks = splitter.split_text(section_content)
        
        for i, chunk in enumerate(sub_chunks):
            # Section name + content for context in the embedding
            enriched_chunk = f"[{section_name}]\n{chunk}"
            final_chunks.append(enriched_chunk)
            metadatas.append({
                "language": language,
                "source": cv_path.name,
                "section": section_name, 
                "chunk_index": i
            })

    # Unique IDs for each chunk to avoid duplicates in the vector store
    ids = [f"{cv_path.stem}_{language}_{m['section']}_{m['chunk_index']}" for m in metadatas]
    
    vectorstore.add_texts(texts=final_chunks, metadatas=metadatas, ids=ids)
    print(f"Indexed {len(final_chunks)} chunks from {cv_path.name} ({language})")


def index_all_cvs():
    for pdf_file in PATH_CV.glob("*.pdf"):
        add_cv_to_db(pdf_file)

def detect_language_from_text(text: str) -> str:
    try:
        detected_lang = detect(text)
        if detected_lang == "es":
            return "spanish"
        elif detected_lang == "en":
            return "english"
        else:
            return "unknown"
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "unknown"
    
if __name__ == "__main__":
    index_all_cvs()
    
    print(f"\nTotal chunks in collection: {vectorstore._collection.count()}")

    query = "Interpretación de datos y visualización en Python"
    results = vectorstore.as_retriever(
        search_kwargs={
            "k": 5,
            # Need this to avoid retrieving English CVs when searching for Spanish ones
            "filter": {"language": detect_language_from_text(query)} 
        }
    ).invoke(query)

    print("\n--- Manual Test Results ---")
    for result in results:
        print(f"Section: {result.metadata.get('section', 'N/A')} ({result.metadata['language']})")
        print(f"Text: {result.page_content[:1500]}...\n")