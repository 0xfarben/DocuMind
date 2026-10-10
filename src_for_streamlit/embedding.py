from typing import List, Any, Callable, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


ProgressCallback = Optional[Callable[[str, float, str], None]]


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    
    def chunk_documents(
        self,
        documents: List[Any],
        progress_callback=None,
    ) -> List[Any]:
        if progress_callback:
            progress_callback(
                "chunking",
                0.15,
                "Splitting documents into chunks...",
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        print(
            f"[INFO] Split {len(documents)} documents "
            f"into {len(chunks)} chunks."
        )

        if progress_callback:
            progress_callback(
                "chunking",
                0.30,
                f"Created {len(chunks):,} chunks.",
            )

        return chunks


    def embed_chunks(
        self,
        chunks: List[Any],
        progress_callback: ProgressCallback = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        total = len(texts)

        if total == 0:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype="float32")

        print(f"[INFO] Generating embeddings for {total} chunks...")

        # Encode in batches so the UI can report genuine progress.
        embedding_batches = []
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = texts[start:end]

            vectors = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            embedding_batches.append(vectors)

            completed = end
            fraction = completed / total
            # Reserve 30%-90% of the overall operation for embedding.
            overall_progress = 0.30 + (0.60 * fraction)

            if progress_callback:
                progress_callback(
                    "embedding",
                    overall_progress,
                    f"Generating embeddings: {completed:,} / {total:,} chunks",
                )

        embeddings = np.vstack(embedding_batches).astype("float32")
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings
