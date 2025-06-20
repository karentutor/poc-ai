#!/usr/bin/env python
# 02_build_index.py
"""
Build an index from widget_catalog.json that includes:
- Document information (name, heading, section headings)
- Widget information (name, context)
- Vector embeddings for similarity search
"""
import json, os, argparse, torch, time
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

# Available models for indexing
AVAILABLE_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "intfloat": "intfloat/multilingual-e5-large",
    "baai": "BAAI/bge-large-en-v1.5",
    "e5": "intfloat/e5-large-v2"
}

def encode_batch(batch_texts, model_path, device):
    """Encode a batch of texts using the model"""
    tqdm.write(f"Starting batch encoding of {len(batch_texts)} texts...")
    embedder = SentenceTransformer(model_path, device=device)
    result = embedder.encode(batch_texts, convert_to_numpy=True).astype("float32")
    tqdm.write(f"Completed batch encoding, shape: {result.shape}")
    return result

def build_index(catalog_file, out_json, model="minilm", gpu=True):
    """Build an index from the widget catalog with vector embeddings."""
    start_time = time.time()
    tqdm.write(f"\n📚 Reading catalog from: {os.path.abspath(catalog_file)}")
    
    with open(catalog_file, 'r', encoding='utf-8') as f:
        catalog = json.load(f)
    
    # Extract documents from the catalog, looking for either 'documents' or 'docs' key
    documents = catalog.get('documents', {})
    if not documents:
        documents = catalog.get('docs', {})
    
    tqdm.write(f"📊 Found {len(documents)} documents to process")
    
    # Check if we have any documents to process
    if not documents:
        tqdm.write("\n⚠️ No documents found in the catalog file! Please check your catalog format.")
        tqdm.write("The catalog should have a top-level 'documents' or 'docs' key containing document data.")
        return  # Exit the function early
    
    # Prepare for vector embeddings
    texts = []
    widget_metadata = []
    widget_id = 0  # Unique identifier for each widget
    
    # Check GPU availability
    device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        tqdm.write(f"Using GPU: {torch.cuda.get_device_name(0)}")
        batch_size = 128
        try:
            res = faiss.StandardGpuResources()
            tqdm.write("FAISS GPU support initialized successfully")
        except AttributeError:
            tqdm.write("Warning: FAISS GPU support not available. Please install faiss-gpu package.")
            tqdm.write("Falling back to CPU processing...")
            device = "cpu"
            batch_size = 32
            res = None
    else:
        tqdm.write("Using CPU for indexing (GPU not available)")
        batch_size = 32
        res = None
    
    # Load the model
    model_path = AVAILABLE_MODELS.get(model, model)
    tqdm.write(f"Loading model: {model_path}")
    
    # Process documents and prepare texts for encoding
    index = {
        "timestamp": datetime.now().isoformat(),
        "documents": []
    }
    
    for doc_name, doc_data in tqdm(documents.items(), desc="Processing documents", unit="doc"):
        doc_entry = {
            "documentName": doc_name,
            "heading": doc_data.get("heading", ""),
            "section_headings": [],
            "widgets": []
        }
        
        # Handle different formats for section headings
        if "section_headings" in doc_data:
            doc_entry["section_headings"] = doc_data["section_headings"]
        
        # Process each widget in the document
        for field in doc_data.get("fields", []):
            widget_entry = {
                "widgetName": field.get("widgetName", ""),
                "context": field.get("context", ""),
                "documentName": doc_name,
                "documentHeading": doc_data.get("heading", ""),
                "sectionContext": []
            }
            
            # Try to extract section context from field data directly if available
            if "sectionHeading" in field and field["sectionHeading"]:
                widget_entry["sectionContext"].append({
                    "text": field["sectionHeading"],
                    "page": field.get("page", 0)
                })
            # Otherwise look for section headings as before
            else:
                context_text = field.get("context", "").lower()
                for heading in doc_data.get("section_headings", []):
                    heading_text = heading.get("text", "").lower()
                    if heading_text in context_text or context_text in heading_text:
                        widget_entry["sectionContext"].append({
                            "text": heading.get("text", ""),
                            "page": heading.get("page", 0)
                        })
            
            doc_entry["widgets"].append(widget_entry)
            
            # Add to texts for vector encoding with enhanced metadata
            texts.append(field.get("context", ""))
            widget_metadata.append({
                "id": widget_id,  # Unique identifier
                "documentName": doc_name,
                "widgetName": field.get("widgetName", ""),
                "heading": doc_data.get("heading", ""),
                "sectionContext": widget_entry["sectionContext"],
                "context": field.get("context", ""),  # Store original context
                "documentIndex": len(index["documents"]),  # Index in documents array
                "widgetIndex": len(doc_entry["widgets"]) - 1  # Index in widgets array
            })
            widget_id += 1
        
        index["documents"].append(doc_entry)
    
    # Check if we have any texts to encode
    if not texts:
        tqdm.write("\n⚠️ No text content found to encode! Please check your catalog data.")
        return  # Exit the function early
    
    # Encode texts with vector embeddings
    tqdm.write(f"\n🔄 Encoding {len(texts)} texts for vector search...")
    vecs = []
    
    if len(texts) > 1000:
        # Use parallel processing
        num_workers = min(multiprocessing.cpu_count(), 4)
        tqdm.write(f"Using {num_workers} workers for parallel encoding")
        
        max_chunk_size = 1000
        chunk_size = min(max_chunk_size, len(texts) // num_workers)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(encode_batch, chunk, model_path, device) for chunk in chunks]
            for future in tqdm(futures, total=len(chunks), desc="Encoding chunks", unit="chunk"):
                result = future.result()
                vecs.append(result)
    else:
        # Single process encoding
        embedder = SentenceTransformer(model_path, device=device)
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
            batch_texts = texts[i:i + batch_size]
            batch_vecs = embedder.encode(batch_texts, convert_to_numpy=True).astype("float32")
            vecs.append(batch_vecs)
    
    # Stack vectors and normalize
    if not vecs:
        tqdm.write("\n⚠️ No vectors were generated! Cannot create index.")
        return  # Exit the function early
        
    vecs = np.vstack(vecs)
    faiss.normalize_L2(vecs)
    
    # Create and populate FAISS index
    tqdm.write("\n🔍 Building FAISS index...")
    faiss_index = faiss.IndexFlatIP(vecs.shape[1])
    
    if device == "cuda" and res is not None:
        faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
    
    for i in tqdm(range(0, len(vecs), batch_size), desc="Indexing vectors", unit="batch"):
        batch_vecs = vecs[i:i + batch_size]
        faiss_index.add(batch_vecs)
    
    if device == "cuda" and res is not None:
        faiss_index = faiss.index_gpu_to_cpu(faiss_index)
    
    # Save everything
    tqdm.write("\n💾 Saving index and metadata...")
    
    # Save the document index
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    # Save the vector index and metadata
    model_name = os.path.basename(model)
    out_prefix = f"widget_{model_name}"
    
    faiss.write_index(faiss_index, f"{out_prefix}.faiss")
    pickle.dump(widget_metadata, open(f"{out_prefix}_metadata.pkl", "wb"))
    
    # Save configuration
    config = {
        "model": model_path,
        "vector_dimension": vecs.shape[1],
        "num_vectors": len(vecs),
        "timestamp": datetime.now().isoformat(),
        "widget_count": widget_id
    }
    with open(f"{out_prefix}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    tqdm.write(f"\n✅ Successfully completed indexing")
    tqdm.write(f"📊 Indexed {len(index['documents'])} documents")
    total_widgets = sum(len(doc['widgets']) for doc in index['documents'])
    tqdm.write(f"📊 Total widgets indexed: {total_widgets}")
    tqdm.write(f"⏱️  Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="widget_catalog.json",
                   help="Input catalog file (default: widget_catalog.json)")
    ap.add_argument("-o", "--out", default="widget_index.faiss",
                   help="Output index file (default: widget_index.faiss)")
    ap.add_argument("--model", default="minilm",
                   help="Model to use for indexing. Can be one of the predefined models (minilm, mpnet, intfloat, baai, e5) or a path to a custom model.")
    ap.add_argument("--gpu", action="store_true", default=True,
                   help="Use GPU for processing (if available)")
    ap.add_argument("--no-gpu", action="store_false", dest="gpu",
                   help="Disable GPU usage and force CPU processing")
    args = ap.parse_args()
    
    build_index(catalog_file=args.catalog, out_json=args.out, model=args.model, gpu=args.gpu)   