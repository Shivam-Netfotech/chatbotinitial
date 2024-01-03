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

def recognize_speech():
    speech_config = speechsdk.SpeechConfig(subscription="81548d38a58b434eb81354e6ba88db37", region="centralindia")
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
    print("Say something...")
    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    else:
        print("Could not understand audio input. Please try again.")
        return None

def speak(text):
    speech_config = speechsdk.SpeechConfig(subscription="81548d38a58b434eb81354e6ba88db37", region="centralindia")
    speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        temp_wav_file = os.path.join("./Output", "temp.wav")
        with open(temp_wav_file, "wb") as f:
            f.write(result.audio_data)
        play_obj = sa.WaveObject.from_wave_file(temp_wav_file).play()
        play_obj.wait_done()
        os.remove(temp_wav_file)
    else:
        print("Speech synthesis failed: {}".format(result.reason))

loader = DirectoryLoader("mydata/")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
    query = recognize_speech()
    if query:
        result = chain({"question": query, "chat_history": chat_history, "temperature": 0.5})
        response_text = result['answer']
        speak(response_text)
        chat_history.append((query, response_text))
