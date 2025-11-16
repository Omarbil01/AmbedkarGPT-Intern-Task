# AI Intern Assignment by Omar Bilgrami
# Github link: https://github.com/Omarbil01/AmbedkarGPT-Intern-Task

from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma            
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

TEXT_FILE_PATH = "speech.txt"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_DIRECTORY = "chroma_db"
LLM_MODEL_NAME = "mistral"

def main():

    loader = TextLoader(TEXT_FILE_PATH)
    documents = loader.load()

    print("Splitting document into chunks")
    text_splitter = CharacterTextSplitter(
        chunk_size=500,  # Chunk size of 500 chars
        chunk_overlap=50   # Overlap of 50 chars between chunks helps in maintaining context
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    print(f"Initializing embedding model '{EMBEDDING_MODEL_NAME}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Change to 'cuda' if you could use gpu
    )

    print(f"Creating and persisting vector store at '{VECTOR_DB_DIRECTORY}'")
    # This creates the vector store from the chunks and saves it to disk
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=VECTOR_DB_DIRECTORY
    )
    vector_store.persist()
    print("Vector store created successfully.")

    print(f"Initializing Ollama LLM with model '{LLM_MODEL_NAME}'")
    llm = OllamaLLM(model=LLM_MODEL_NAME)

    retriever = vector_store.as_retriever()

    # Create a prompt template for RAG
    template = """Answer the question based only on the following context:

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)

    # Format documents function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    print("Creating the RAG chain")
    # Build the RAG chain using LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain is ready.\n")

    print("Ask a question based on the provided text. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            # Get user input
            query = input("\nQuestion: ")
            
            if query.lower() in ['exit', 'quit']:
                print("Exiting, Thank You!")
                break
            
            if not query.strip():
                continue

            answer = rag_chain.invoke(query)

            # Print the answer
            print("\nAnswer:")
            print(answer)

        except Exception as e:
            print(f"\nAn error occurred: {e}")
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break

if __name__ == "__main__":
    main()