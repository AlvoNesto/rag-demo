import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever

# Configuración inicial
PERSIST_DIR = "chroma_db_dir"
COLLECTION_NAME = "stanford_report_data"

# Inicializar modelos
embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_vector_db(agent_id: str) -> Chroma:
    collection_name = f"kb_{agent_id}"
    db_path = os.path.join(PERSIST_DIR, collection_name)

    if os.path.exists(os.path.join(db_path, "index")):
        return Chroma(
            persist_directory=db_path,
            embedding_function=embed_model,
            collection_name=collection_name
        )
    else:
        loader = TextLoader(f"docs/kb_{agent_id}.txt", encoding="utf-8")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)

        db = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory=db_path,
            collection_name=collection_name
        )
        db.persist()
        return db
    
def get_context(question: str, retriever: VectorStoreRetriever) -> str:
    docs = retriever.get_relevant_documents(question)
    return "\n\n".join(doc.page_content.strip()[:500] for doc in docs)