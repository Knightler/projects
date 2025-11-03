from dotenv import load_dotenv
from langchain_community.retrievers import ArxivRetriever
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import tool

# Loads the API key
load_dotenv()

# Loads the LLM - gpt-4o-mini
llm = ChatOpenAI(
        model="gpt-4o-mini"
        )

# Loads the retriever - for Arxiv
retriever = ArxivRetriever(
        load_max_docs=1,
        get_full_documents=True,
        arxiv_search="all",
        arxiv_exceptions=True
        )

def fetch_doc(u: str):
    """
    Load the exact document mentioned by the number from the Arxiv and retrieve the information so we can start the Q&A on it

    Args:
        u: the user input containing the question
    """

    doc = retriever.invoke(u)
    print("Document fetched successfully")

    return doc

def load_doc(doc: str):
    response = llm.invoke(f'Document: {doc}\nSummarize the main points of the document in a concise manner.')
    return response

title = input("\nDocument Title: ")
fetched = fetch_doc(title)
docs_content = "\n".join([doc.page_content for doc in fetched])

while True:
    user = input("\nAsk: ")
    if user == "quit":
        print("bye")
        break

    else:
        docs = load_doc(docs_content)
        for chunk in llm.stream(f"Document: {docs_content}\nQuestion: {user}"):
            print(chunk.text, end="", flush=True)
