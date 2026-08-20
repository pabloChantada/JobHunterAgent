"""Vector store for CV embeddings and retrieval.

This module stores vector embeddings of CVs (English and Spanish),
chunked by section for precise retrieval.
"""
import re
import unicodedata
from pathlib import Path

import pymupdf
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect
from langdetect import DetectorFactory
DetectorFactory.seed = 0

PATH_CV = Path("data/cv")
PERSIST_DIR = "data/chroma_cv"
SECTION_PATTERN = re.compile(
    r"^\s*(PERFIL|PROFILE|CONOCIMIENTOS T.*?CNICOS|TECHNICAL SKILLS|"
    r"EXPERIENCIA|EXPERIENCE|PROYECTOS|PROJECTS|EDUCACI.*?N|FORMACI.*?N|"
    r"EDUCATION|IDIOMAS|LANGUAGES)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Global variable to hold the Chroma vectorstore instance
# and avoid re-initializing it multiple times.
_vectorstore_instance = None

def get_vectorstore() -> Chroma:
    """Return the shared Chroma vectorstore instance, creating it on first use."""
    global _vectorstore_instance
    if _vectorstore_instance is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        _vectorstore_instance = Chroma(
            collection_name="cv_embeddings",
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
    return _vectorstore_instance

def get_splitter() -> RecursiveCharacterTextSplitter:
    # Defines the chunking strategy for the CVs, prioritizing section/paragraph breaks.
    return RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". "],
    )
    
def extract_text(pdf_path: Path) -> str:
    """Extract text from a PDF file with proper unicode normalization.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Normalized text content from the PDF.
    """
    doc = pymupdf.open(pdf_path)
    raw_text = "\n".join(page.get_text() or "" for page in doc)
    normalized_text = unicodedata.normalize("NFKC", raw_text)
    return normalized_text

def chunk_by_sections(text: str) -> dict[str, str]:
    """Split text into sections based on predefined section headers.

    Args:
        text: The full CV text.

    Returns:
        Dictionary with section names as keys and section content as values.
    """
    sections = {}
    matches = list(SECTION_PATTERN.finditer(text))

    # Filter out the header, usually contact info, name, etc.
    if matches:
        sections["CONTACTO_HEADER"] = text[: matches[0].start()].strip()

    for i, match in enumerate(matches):
        section_name = match.group(1).strip().upper()
        start_idx = match.end()
        # The end index is either the start of the next match or the end of text
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        sections[section_name] = text[start_idx:end_idx].strip()

    return sections

def detect_language(filename: str) -> str:
    """Detect the language of a CV based on its filename.

    Convention: ``*_ES.pdf`` for Spanish CVs, ``*.pdf`` for English CVs.

    Args:
        filename: The name of the CV file.

    Returns:
        Either "spanish" or "english".
    """
    return "spanish" if filename.endswith("_ES.pdf") else "english"


def add_cv_to_db(cv_path: Path) -> None:
    """Add a CV to the vector store, chunked by sections.

    Args:
        cv_path: Path to the CV PDF file.
    """
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
        sub_chunks = get_splitter().split_text(section_content)

        for i, chunk in enumerate(sub_chunks):
            # Section name + content for context in the embedding
            enriched_chunk = f"[{section_name}]\n{chunk}"
            final_chunks.append(enriched_chunk)
            metadatas.append(
                {
                    "language": language,
                    "source": cv_path.name,
                    "section": section_name,
                    "chunk_index": i,
                }
            )

    # Unique IDs for each chunk to avoid duplicates in the vector store
    ids = [
        f"{cv_path.stem}_{language}_{m['section']}_{m['chunk_index']}"
        for m in metadatas
    ]

    get_vectorstore().add_texts(texts=final_chunks, metadatas=metadatas, ids=ids)
    print(f"Indexed {len(final_chunks)} chunks from {cv_path.name} ({language})")


def index_all_cvs() -> None:
    """Index all CVs in the data directory into the vector store."""
    for pdf_file in PATH_CV.glob("*.pdf"):
        add_cv_to_db(pdf_file)

def detect_language_from_text(text: str) -> str:
    """Detect the language of text using langdetect.

    Args:
        text: The text to detect language for.

    Returns:
        Either "spanish", "english", or "unknown".
    """
    try:
        detected_lang = detect(text)
        if detected_lang == "es":
            return "spanish"
        if detected_lang == "en":
            return "english"
        return None  
    except Exception as error:
        print(f"Language detection failed: {error}")
        return None
    
if __name__ == "__main__":
    # pylint: disable=protected-access
    index_all_cvs()
    
    # Force initialization even if no PDFs were found
    db = get_vectorstore()

    if _vectorstore_instance is not None:
        print(f"\nVector store persisted at: {PERSIST_DIR}")
    else:
        raise RuntimeError("Vector store instance was not created. Check the indexing process.")