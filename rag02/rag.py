# pip install langchain
# pip install langchain_community
# pip install sentence-transformers

import os
import sys
import logging
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import AsyncChromiumLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.document_transformers import Html2TextTransformer
from langchain.vectorstores import FAISS
from langchain.schema.document import Document

#ngpus = FAISS.get_num_gpus()
#print("number of GPUs:", ngpus)


class Ragger:
    def __init__(self,):
        pass

    def make_embedding(self, text, model_name, chunk_size=500, chunk_overlap=200):
        self.chunked_documents = None
        docs_transformed = []
        print("creating docs..")
        doc = Document(page_content=text)
        docs_transformed.append(doc)
        print("RecursiveCharacterTextSplitter..")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunked_documents = text_splitter.split_documents(docs_transformed)
        print("HuggingFaceEmbeddings..")
        hfe = HuggingFaceEmbeddings(model_name=model_name)
        print("FAISS.from_documents..")
        self.db = FAISS.from_documents(self.chunked_documents, hfe)

    def load_model(self, model_config):
        self.inference = Inference()
        self.inference.set_model_and_tokenizer(model_config)

    def sim_search(self, query, topk):
        docs = self.db.similarity_search(query=query, k=topk)
        context = ""
        for doc in docs:
            context += doc.page_content + "\n"

        return context

    #def inference(self, sim_query, llm_prompt, llm_quest, k):
        #llm_prompt = llm_prompt.replace('<Kr>quest<Kr>', llm_quest).replace('<Kr>context<Kr>', context)
        #output = self.inference.ask_model(llm_prompt)
        #return output


if __name__ == "__main__":

    #import torch
    #print("torch.cuda.is_available:", torch.cuda.is_available())
    #print("device_name:", torch.cuda.get_device_name(0))

    rag = Ragger()

    #rag_docs_folder = "rag_docs"
    #rag.download_docs_from_s3(rag_docs_folder)
    #with open(os.path.join(rag_docs_folder, "wring_test.txt")) as fp:
    #    text = fp.read()
    #print(f"text size: {len(text)}")

    with open("horoscope.txt") as fp:
        text = fp.read()

    model_name = "sentence-transformers/all-mpnet-base-v2"
    rag.make_embedding(text, model_name)

    query = "Какой гороскоп на неделю?"
    context = rag.sim_search(query, topk=4)
    print("CONTEXT:\n", context)

    # Call a LLM model
    #output = call_llm_model(context)