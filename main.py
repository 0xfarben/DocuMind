import os
from src.search_updated import RAGSearch
from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents

if __name__ == "__main__":
    
    docs = load_all_documents("data")

    store = FaissVectorStore("faiss_store")

    # BUILD STORE -> for the first time
    # store.build_from_documents(docs)

    # NO NEED TO BUILD STORE -> for each and every run.
    # store.load()
    #print(store.query("Why Machine Learning is required??", top_k=3))

    if (
        os.path.exists(os.path.join("faiss_store", "faiss.index"))
        and os.path.exists(os.path.join("faiss_store", "metadata.pkl"))
    ):
        store.load()
    else:
        store.build_from_documents(docs)

    rag_search = RAGSearch()

    query = "Why Machine Learning is required?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    
    print("Summary:", summary)