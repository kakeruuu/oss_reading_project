import os
from dotenv import load_dotenv

import openai

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from embedding import Embedding

# git clone https://github.com/Significant-Gravitas/Auto-GPT.git

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def main():
    root_dir = '/content/Auto-GPT'
    persist_directory = "langchain_vectordb"

    embeddings = Embedding(root_dir=root_dir, persist_directory=persist_directory)
    embeddings.init_vector()

    # 初期化したベクトルデータをそのまま利用しても良い
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings)

    retriever = vectordb.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10

    model = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

    questions = [
        "Please give an overview of Auto-GPT in about 100 characters",
        "Tell me in 100 characters how the response to a particular command is generated",
    ]
    chat_history = []

    for question in questions:
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result['answer']))
        print(f"-> **Question**: {question} \n")
        print(f"**Answer**: {result['answer']} \n")

if __name__ == "__main__":
    main()