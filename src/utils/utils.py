##------------------------------------
## utils.py contains all necessary utils
## for creating a LLM chatbot to discuss
## a self created vectorstore
## -----------------------------------


import os
import shutil
import param
import panel as pn

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain


def get_openai_api_key(api_txt_dir:str) -> str:
    """This function looks for a txt file which contains a single
    string indicating the users openai_api_key. It loads the api key
    to be set as an environment variable.
    """
    f = open(api_txt_dir, 'r')
    key = f.readline()
    return key

def get_pdfs_docs_list(datadir: str) -> list:
    """This function searches the datadir for pdfs, and then uses PyPDFLoader to 
    create a list of langchain.schema.Document. These will eventually be split into 
    chunks and then converted with embeddings and stored in a vectorstore.
    
    Returns a list of document names"""
    docs = []
    pdfs = [i for i in os.listdir(datadir) if 'pdf' in i]
    for i in pdfs:
        loader = PyPDFLoader(datadir+i)
        docs.extend(loader.load())
    
    return docs


def load_vectordb_retriever(persist_directory: str) -> Chroma:
    """This function looks into the persist_directory and loads
    the chroma vectordatabase returning it as a retriever. The
    function assumes the emebddings are OpenAIEmbeddings and that 
    the vectorstore we want is Chroma
    
    Returns a Chroma as a retriever for immediate use in an LLM chain"""

    # define embedding
    embeddings = OpenAIEmbeddings()
    # load vectordb from disk
    vectordb = Chroma(persist_directory = persist_directory, embedding_function = embeddings)
    # define retriever
    return vectordb.as_retriever()

def save_vectordb(datadir:str, persist_dir: str, text_splitter: RecursiveCharacterTextSplitter, embeddings: OpenAIEmbeddings, allowed_special_kwargs: list) -> None:
    """This function generates and persists the vectorstore. It assumes the vectorstore 
    is chroma, embeddings are openai, and the text splitting is recursive as identified 
    by type hinting. Experimentation shows other embeddings and splitters can work, however
    the vectorstore function internal is chroma. Change things at your own risk. 
    
    Returns None
    """
    
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

def retrieve_context_vectordb(config: dict) -> Chroma:
    """
    This function handles the vector store saving and loading; 
    and document handling for the LLM. If config has 'retrain_str'
    set to True this will assume the current vectorstore is outdated
    and needs to be recreated, usually beecause of new documents. If
    this is set to false, the existing vectorstore will be loaded. 

    On the off chance the directory is broken or DNE a new vectorstore
    can be created, assuming config[datadir] is properly pointing to 
    the pdfs one wants to use for the vectorstore. 

    This returns a loaded Chroma vectordatabase as a retriever
    """

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
    """
    This function applies the retriever and llm set in the config
    and returns a ConversationalRetrievalChain which will utilize
    the retriever loaded by the retrieve_context_vectordb function.
    """
    llm = config['llm']
    retriever = config['retriever']

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 

class cbfs(param.Parameterized):
    """
    This class initializes and calls the Docubot to be
    used in a frontend dashboard
    """
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    
    def __init__(self,  config, **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        self.config = config
        self.qa = get_conversational_llm(self.config)#, self.retriever)#,"stuff", 4)
    
    def call_load_db(self, count):
        """
        Loads the conversational LLM and sets up the widgets for the front end
        """
        if count == 0 or file_input.value is None:  # init or no file specified :
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")  # local copy
            self.loaded_file = file_input.filename
            button_load.button_style="outline"
            self.qa = load_db("temp.pdf", "stuff", 4)
            button_load.button_style="solid"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        """
        Utilizes widgets and helps parse the input and outputs to their
        proper location in the front end
        """
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer'] 
        self.panels.extend([
            pn.Row('User:', pn.pane.Markdown(query, width=600)),
            pn.Row('DocuBot:', pn.pane.Markdown(self.answer, width=600, style={'background-color': '#F6F6F6'}))
        ])
        inp.value = ''  #clears loading indicator when cleared
        return pn.WidgetBox(*self.panels,scroll=True)

    @param.depends('db_query ', )
    def get_lquest(self):
        if not self.db_query :
            return pn.Column(
                pn.Row(pn.pane.Markdown(f"Last question to DB:", styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("no DB accesses so far"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.db_query )
        )

    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return 
        rlist=[pn.Row(pn.pane.Markdown(f"Result of DB lookup:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    @param.depends('convchain', 'clr_history') 
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No History Yet")), width=600, scroll=True)
        rlist=[pn.Row(pn.pane.Markdown(f"Current Chat History variable", styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    def clr_history(self,count=0):
        self.chat_history = []
        return 
