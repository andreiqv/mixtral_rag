from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from pinecone import Pinecone, PodSpec
from dotenv import load_dotenv
import os

class ChatBot():
  load_dotenv()
  loader = TextLoader('./horoscope.txt', encoding='utf8')
  documents = loader.load()
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
  docs = text_splitter.split_documents(documents)

  HUGGINGFACE_API_KEY=os.getenv('HUGGINGFACE_API_KEY')
  repo_id = "NousResearch/Llama-2-7b-chat-hf"
  embeddings = HuggingFaceEmbeddings(model_name=repo_id)

  pc = Pinecone(api_key= os.getenv('PINECONE_API_KEY'), environment='gcp-starter')
  
  index_name = "langchain-grad"

  if index_name not in pc.list_indexes():
    pc.create_index(name=index_name, metric="cosine", dimension=768, spec=PodSpec(environment="gcp-starter",pod_type="s1.x1"))
    docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
  else:
    docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

  llm = HuggingFacePipeline.from_model_id(
      model_id=repo_id,task="text-generation", pipeline_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 10}
  )

  from langchain_core.prompts import PromptTemplate

  template = """
    Ты гадалка. Люди задают тебе вопросы о своей будующей жизни и судьбе. 
    Используй следующий фрагмент контекста, чтобы ответить на вопрос. 
    Если ты не знаете ответа, просто скажи, что не знаешь. 
    Ответ должен быть кратким и не превышать четырех предложений. 

  Context: {context}
  Question: {question}
  Answer: 

  """

  prompt = PromptTemplate(template=template, input_variables=["context", "question"])

  from langchain_core.runnables import RunnablePassthrough
  from langchain_core.output_parsers import StrOutputParser

  rag_chain = (
    {"context": docsearch.as_retriever(),  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
  )

bot = ChatBot() 
input = input ( "Спросите меня о чем угодно: " ) 
result = bot.rag_chain.invoke( input ) 
print (result)