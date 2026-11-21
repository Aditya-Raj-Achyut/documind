"""Ingestion pipeline: PDFs -> clean text -> chunks -> embeddings -> ChromaDB.

Usage:
    python -m app.ingest                      # default chunk size from config
    python -m app.ingest --chunk-size 500 --overlap 75
"""
import argparse
from pathlib import Path

from app import config
from app.utils import clean_text, split_text


def load_pdf_pages(pdf_path: Path):
    """Yield (page_number, cleaned_text) for every page that has real text."""
    try:
        import pymupdf  # PyMuPDF >= 1.24.3
    except ImportError:  # older versions expose it as `fitz`
        import fitz as pymupdf

    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            text = clean_text(page.get_text())
            if len(text) > 50:  # skip blank / image-only pages
                yield page_no, text


def collection_exists(chunk_size: int) -> bool:
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    # list_collections() returns objects in old Chroma versions, names in new ones
    names = [getattr(c, "name", c) for c in client.list_collections()]
    return config.collection_name(chunk_size) in names


def build_index(
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    pdf_dir: Path = config.PDF_DIR,
) -> int:
    """(Re)build the vector index for one chunk size. Returns the chunk count."""
    import chromadb
    from tqdm import tqdm

    from app.models import get_embedder

    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}. Add some PDFs and run again.")

    ids, docs, metas = [], [], []
    for pdf in tqdm(pdfs, desc="Reading PDFs"):
        for page_no, text in load_pdf_pages(pdf):
            for j, chunk in enumerate(split_text(text, chunk_size, overlap)):
                ids.append(f"{pdf.stem}_p{page_no}_c{j}")
                docs.append(chunk)
                metas.append({"source": pdf.name, "page": page_no})
    if not docs:
        raise SystemExit("No text could be extracted. Are the PDFs scanned images?")

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    name = config.collection_name(chunk_size)
    try:
        client.delete_collection(name)
    except Exception:
        pass  # collection did not exist yet
    # Embeddings are normalised, so the default L2 distance ranks exactly like cosine.
    collection = client.create_collection(name)

    model = get_embedder()
    batch = 64
    for i in tqdm(range(0, len(docs), batch), desc="Embedding"):
        part = docs[i : i + batch]
        vectors = model.encode(part, normalize_embeddings=True, show_progress_bar=False)
        collection.add(
            ids=ids[i : i + batch],
            documents=part,
            metadatas=metas[i : i + batch],
            embeddings=vectors.tolist(),
        )

    print(f"Indexed {len(docs)} chunks from {len(pdfs)} PDFs into '{name}'.")
    return len(docs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the DocuMind vector index.")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    args = parser.parse_args()
    build_index(args.chunk_size, args.overlap)
