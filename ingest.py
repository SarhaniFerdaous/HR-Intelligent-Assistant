import os
import sys
import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

CHROMA_DIR = "chroma_db"
SUPPORTED  = {".pdf", ".txt", ".docx", ".csv"}

# ─────────────────────────────────────────────
# SHARED CACHED EMBEDDINGS — loaded once ever
# ─────────────────────────────────────────────

def _get_embeddings():
    """Always use the cached embeddings from tools.py — never reload."""
    from tools import _get_embeddings as _cached
    return _cached()


def _get_vectorstore():
    """Return a Chroma vectorstore using the cached embedding model."""
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=_get_embeddings())


# ─────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────

def load_file(file_path: str) -> list:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".csv":
            loader = CSVLoader(file_path, encoding="utf-8")
        else:
            return []
        return loader.load()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


# ─────────────────────────────────────────────
# INGEST — single file
# ─────────────────────────────────────────────

def ingest_file(file_path: str) -> dict:
    if not os.path.exists(file_path):
        return {"success": False, "chunks": 0, "message": f"File not found: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED:
        return {"success": False, "chunks": 0, "message": f"Unsupported type: {ext}"}

    # Load
    documents = load_file(file_path)
    if not documents:
        return {"success": False, "chunks": 0, "message": "Could not load file."}

    # Chunk — larger chunks = fewer embeddings = faster
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,       # was 400 — fewer chunks to embed
        chunk_overlap=50,     # was 60
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        return {"success": False, "chunks": 0, "message": "File produced no chunks."}

    # Embed and store — reuses cached model, no reload
    try:
        vectorstore = _get_vectorstore()
        # Add in one batch — faster than multiple calls
        vectorstore.add_documents(chunks)
        return {
            "success": True,
            "chunks": len(chunks),
            "message": f"Successfully ingested {len(chunks)} chunks."
        }
    except Exception as e:
        return {"success": False, "chunks": 0, "message": f"Storage error: {e}"}


# ─────────────────────────────────────────────
# INGEST FOLDER
# ─────────────────────────────────────────────

def ingest_folder(folder_path: str = "docs") -> dict:
    if not os.path.exists(folder_path):
        return {"success": False, "message": f"Folder not found: {folder_path}"}

    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    ]
    if not files:
        return {"success": False, "message": "No supported files found in folder."}

    results      = []
    total_chunks = 0

    for file_path in files:
        result = ingest_file(file_path)
        results.append({"file": os.path.basename(file_path), **result})
        if result["success"]:
            total_chunks += result["chunks"]

    successful = sum(1 for r in results if r["success"])
    return {
        "success":        True,
        "files_processed": successful,
        "total_files":    len(files),
        "total_chunks":   total_chunks,
        "details":        results,
        "message":        f"Ingested {successful}/{len(files)} files → {total_chunks} total chunks."
    }


# ─────────────────────────────────────────────
# DELETE
# ─────────────────────────────────────────────

def delete_document(filename: str) -> dict:
    try:
        vectorstore = _get_vectorstore()
        collection  = vectorstore.get()
        ids_to_delete = [
            collection["ids"][i]
            for i, meta in enumerate(collection.get("metadatas", []))
            if meta and filename in meta.get("source", "")
        ]
        if not ids_to_delete:
            return {"success": False, "message": f"No chunks found for: {filename}"}
        vectorstore.delete(ids=ids_to_delete)
        return {"success": True, "message": f"Deleted {len(ids_to_delete)} chunks for {filename}."}
    except Exception as e:
        return {"success": False, "message": f"Delete error: {e}"}


# ─────────────────────────────────────────────
# LIST
# ─────────────────────────────────────────────

def list_documents() -> list:
    try:
        vectorstore = _get_vectorstore()
        collection  = vectorstore.get()
        sources = {}
        for meta in collection.get("metadatas", []):
            if meta and "source" in meta:
                name = os.path.basename(meta["source"])
                sources[name] = sources.get(name, 0) + 1
        return [{"name": n, "chunks": c} for n, c in sorted(sources.items())]
    except Exception:
        return []


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "docs"
    print(f"Ingesting from: {folder}/")
    result = ingest_folder(folder)
    print(result["message"])