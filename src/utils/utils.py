##------------------------------------
## utils.py contains all necessary utils
## for creating a LLM chatbot to discuss
## a self created vectorstore
## -----------------------------------


import os
import shutil

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain


def get_pdfs_docs_list(datadir: str) -> list:
    """This function searches the datadir for pdfs, and then uses PyPDFLoader to 
    create a list of langchain.schema.Document. These will eventually be split into 
    chunks and then converted with embeddings and stored in a vectorstore."""
    docs = []
    pdfs = [i for i in os.listdir(datadir) if 'pdf' in i]
    for i in pdfs:
        loader = PyPDFLoader(datadir+i)
        docs.extend(loader.load())
    
    return docs


def load_vectordb_retriever(persist_directory: str) -> Chroma:

    # define embedding
    embeddings = OpenAIEmbeddings()
    # load vectordb from disk
    vectordb = Chroma(persist_directory = persist_directory, embedding_function = embeddings)
    # define retriever
    return vectordb.as_retriever()

def save_vectordb(datadir:str, persist_dir: str, text_splitter: RecursiveCharacterTextSplitter, embeddings: OpenAIEmbeddings, allowed_special_kwargs: list) -> None:
    
    docs = get_pdfs_docs_list(datadir)

    splits = text_splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
        allowed_special={allowed_special_kwargs}
    )

    vectordb.persist()
    vectordb = None

    return

##if vectorstore exists, load vectorstore. If vectorstore DOESN'T exist, create vectorstore and then load it. 

def retrieve_context_vectordb(config: dict) -> Chroma:
    basedir = config['basedir']
    retrain_str = config['retrain_str']
    persist_dir = config['persist_dir']
    datadir = config['datadir']
    text_splitter = config['text_splitter']
    embeddings = config['embeddings']
    allowed_special_kwargs = config['allowed_special_kwargs']


    if os.path.isdir(basedir) == False:
        os.mkdir(basedir)

    if retrain_str == False:
        if os.path.isdir(persist_dir) == True:
            print('loading vectordb')
        else:
            print(f"no vectordb found, saving and then loading in vectordb using docs found in {datadir} and saving in {persist_dir}")
            save_vectordb(datadir, persist_dir, text_splitter, embeddings, allowed_special_kwargs)
        vectordb = load_vectordb_retriever(persist_dir)


    elif retrain_str == True:

        if os.path.isdir(persist_dir) == True:
            print('removing old vectordb')
            shutil.rmtree(persist_dir)

        print('generating vectordb on data')
        os.mkdir(persist_dir)
        save_vectordb(datadir, persist_dir, text_splitter, embeddings, allowed_special_kwargs)
        vectordb = load_vectordb_retriever(persist_dir)

    return vectordb

def get_conversational_llm(config):
    llm = config['llm']
    retriever = config['retriever']

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 

