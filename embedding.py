import os
import platform

import openai
import chromadb
import langchain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader


class Embedding:

    def __init__(self, root_dir, persist_directory) -> None:
        self.root_dir = root_dir
        self.docs = []
        self.texts = []
        self.persist_directory = persist_directory
        self.embedding = OpenAIEmbeddings()
        
    def load(self):
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for file in filenames:
                try:
                    loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                    self.docs.extend(loader.load_and_split())
                except Exception as e:
                    pass
    
    def split(self):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = text_splitter.split_documents(self.docs)

    def init_vector(self):
        self.load()
        self.split()
        # ベクトルデータの初期化
        chroma_index = Chroma.from_documents(self.docs, embedding=self.embedding, persist_directory=self.persist_directory)
        # ベクトルデータをディレクトリに保存
        chroma_index.persist()
    
    

    
    
