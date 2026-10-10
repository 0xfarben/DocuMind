import os
import faiss
import numpy as np
import pickle
from typing import List, Any, Callable, Optional
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline


ProgressCallback = Optional[Callable[[str, float, str], None]]


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(
        self,
        documents: List[Any],
        progress_callback: ProgressCallback = None,
    ):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")

        if progress_callback:
            progress_callback("chunking", 0.10, "Preparing document chunks...")

        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = emb_pipe.chunk_documents(
            documents,
            progress_callback=progress_callback,
        )

        if not chunks:
            raise ValueError("No text chunks were created from the uploaded documents.")

        embeddings = emb_pipe.embed_chunks(
            chunks,
            progress_callback=progress_callback,
            batch_size=32,
        )

        metadatas = [
            {
                **chunk.metadata,
                "source": chunk.metadata.get("source", "Unknown source"),
                "page": chunk.metadata.get("page"),
                "text": chunk.page_content,
            }
            for chunk in chunks
        ]

        self.add_embeddings(np.asarray(embeddings, dtype="float32"), metadatas)

        if progress_callback:
            progress_callback("indexing", 0.93, "Building and saving FAISS index...")

        self.save()

        if progress_callback:
            progress_callback("complete", 1.0, "Knowledge base is ready.")

        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Embeddings must be a non-empty 2D array.")

        dim = embeddings.shape[1]
        print(f"Dimension: {dim}")

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "wb") as file:
            pickle.dump(self.metadata, file)

        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        self.index = faiss.read_index(faiss_path)

        with open(meta_path, "rb") as file:
            self.metadata = pickle.load(file)

        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.index is None:
            raise RuntimeError("Vector store is not loaded or built.")

        k = min(top_k, self.index.ntotal)
        if k <= 0:
            return []

        distances, indices = self.index.search(query_embedding, k)
        print("---------------")
        print(f"\nDISTANCE: {distances}")
        print(f"\nINDEX: {indices}")

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            results.append({
                "index": int(idx),
                "distance": float(distance),
                "metadata": self.metadata[idx],
            })

        print(f"RESULTS: {results}")
        print("---------------")
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_embedding = self.model.encode([query_text]).astype("float32")
        return self.search(query_embedding, top_k=top_k)
