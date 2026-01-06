from langchain_community.document_loaders import PyPDFLoader,TextLoader,Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_documents():
    docs = []

    pdf_path = r"C:\Users\sad\Downloads\sample.pdf"
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_docs = pdf_loader.load()
    for d in pdf_docs:
        d.metadata["source"] = "PDF"
        d.metadata["file"] = "sample.pdf"
    docs.extend(pdf_docs)

    text_path = r"C:\Users\sad\Downloads\AI_ML_Streamlit_text.txt"
    text_loader = TextLoader(text_path, encoding="utf-8")
    text_docs = text_loader.load()
    for d in text_docs:
        d.metadata["source"] = "TEXT"
        d.metadata["file"] = "AI_ML_Streamlit_text.txt"
    docs.extend(text_docs)

    docx_path = r"C:\Users\sad\Downloads\Activity.docx"
    docx_loader = Docx2txtLoader(docx_path)
    docx_docs = docx_loader.load()
    for d in docx_docs:
        d.metadata["source"] = "DOCX"
        d.metadata["file"] = "Activity.docx"
    docs.extend(docx_docs)

    return docs

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=50)
    return splitter.split_documents(documents)

def build_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


def query_vector_db(vector_db, query, k=15):
    results = vector_db.similarity_search_with_score(query, k=k)

    grouped_results = {}

    for doc, score in results:
        source = doc.metadata.get("source")

        if source not in grouped_results:
            grouped_results[source] = {
                "file": doc.metadata.get("file"),
                "page": doc.metadata.get("page", "N/A"),
                "score": score,
                "text": doc.page_content
            }

    print(f"Query: {query}")

    for source, data in grouped_results.items():
        print(f"Source: {source}")
        print("File:", data["file"])
        print("Page:", data["page"])
        print("Text:", data["text"])

documents = load_documents()
chunks = chunk_documents(documents)
vector_db = build_vector_db(chunks)

query = "Explain one concept of Machine Learning"
query_vector_db(vector_db, query)

# def for_PDF():
#     pdf_path = os.path.join("C:\\Users", "sad", "Downloads", "sample.pdf")

#     loader = PyPDFLoader(pdf_path)
#     documents = loader.load()
#     print("Total pages loaded:", len(documents))

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150)

#     chunks = text_splitter.split_documents(documents)
#     print("Total chunks created:", len(chunks))

#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vector_db = FAISS.from_documents(documents=chunks, embedding=embeddings)

#     query = "Explain deep learning"
#     docs = vector_db.similarity_search_with_score(query, k=1)

#     for i, (doc, score) in enumerate(docs):
#         print("Chunk text:\n", doc.page_content)
#         print("Metadata:", doc.metadata)

# def for_Text():
#     text_path = r"C:\Users\sad\Downloads\AI_ML_Streamlit_text.txt"

#     loader = TextLoader(text_path, encoding="utf-8")
#     documents = loader.load()
#     print("Documents loaded:", len(documents))

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150)

#     chunks = text_splitter.split_documents(documents)
#     print("Total chunks:", len(chunks))    

#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_db = FAISS.from_documents(documents=chunks,embedding=embeddings)

#     query = "What is Streamlit used for?"
#     results = vector_db.similarity_search_with_score(query,k=1)

#     for i, (doc, score) in enumerate(results):
#         print("Chunk ID:", doc.metadata)
#         print("Text:", doc.page_content)

# def for_Doc():
#     docx_path=r"C:\Users\sad\Downloads\Activity.docx"

#     loader = Docx2txtLoader(docx_path)
#     documents = loader.load()
#     print("Documents loaded:", len(documents))

#     text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=400,
#     chunk_overlap=80)

#     chunks = text_splitter.split_documents(documents)
#     print("Total chunks:", len(chunks))

#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_db = FAISS.from_documents(documents=chunks,embedding=embeddings)

#     query = "What is Streamlit used for?"
#     results = vector_db.similarity_search_with_score(query,k=1)
#     for i, (doc, score) in enumerate(results):
#         print("Chunk ID:", doc.metadata)
#         print("Text:", doc.page_content)



