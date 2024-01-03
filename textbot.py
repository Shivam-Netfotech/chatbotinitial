import openai
import os
import sys
import unstructured
import tiktoken
import chromadb
import azure.cognitiveservices.speech as speechsdk
import simpleaudio as sa

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper


os.environ["OPENAI_API_KEY"] = "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"

os.environ["SPEECH_KEY"] = "81548d38a58b434eb81354e6ba88db37"
os.environ["SPEECH_REGION"] = "centralindia"


# This function is used to pass the argument with query.

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]


#Load the custom Dataset and split into chunks, You can load data as (pdf, text file, html file and WebBaseLoader.)
loader = DirectoryLoader("mydata/")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# This part is used for embeddings the docs and store it into Vector DB and intialize the retriever.
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

# Create the RetrievalQA object
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

# This part is used to build a chat or Q&A application having the capability of both conversational capabilities and document retrieval
chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  #if query in ['quit', 'q', 'exit']:
    #sys.exit()
  result = chain({"question": query, "chat_history": chat_history, "temperature": 0.5})
  print(result['answer'])

  chat_history.append((query, result['answer']))
  query = None

# ... (previous code)

# This part is used to build a chat or Q&A application having the capability of both conversational capabilities and document retrieval
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")

    # Get the response from the chat model
    result = chain({"question": query, "chat_history": chat_history, "temperature": 0.5})
    response_text = result['answer']

    # Print the response
    print(response_text)

    # Speak the response
    speak(response_text)

    # Append the query and response to chat history
    chat_history.append((query, response_text))
    query = None
