# LLMDocExplorer

## Project Description:

A Jupyter front end for a LangChain powered LLM. LLM uses local documents for context and can be used for conversational analysis and retrieval. Context is created through pdfs and saved to disk as a Chroma Vectorstore. Currently ChatGPT 3.5 is the LLM supported.

## Installation
1. Clone the Repo in the command line: `git clone git@github.com:wolfgangjblack/LLMDocExplorer.git`
2. Install the required dependences: `pip install -r requirements.txt`


## Usage:

Assuming users have jupyter notebooks installed, the user can follow these steps to quickly get talking with their documents. 

1. users view /src/config/set_config.py to set their local directories.
	- basedir and persist_dir can be ignored
	- retrain can be ignored for the first pass
	- datadir should be set to where ever users have the pdfs of interest
2. users should generate a txt file called "api_key.txt" in /src/
	- this file should contain a single string that is their openai_api_key found in their user profile on openai.com
3. user should open the front end at /src/main.ipynb and run all cells
4. at the bottom of the main.ipynb notebook users can converse with DocuBot about their context

## Features

- Document Analysis: The chatbot utilizes Langchain to analyze the contents of uploaded documents, extract key information, and generate insights.
- Natural Language Interaction: Users can converse with the chatbot using natural language queries to retrieve specific information from their documents.
- Document Retrieval: The chatbot provides intelligent document retrieval based on user queries, enabling quick access to relevant sections or summaries.
- Currently this utilizes ChatGPT 3.5 Turbo and utilizes recursive text splitting with some overlap. Users can change some of these settings by digging into /src/utils/utils.py and /src/main.ipynb
- This utilizes as single ConversationalRetrievalChain currently

## Contributing

Contributions to this project are Welcome! If you encounter any issues or have suggestions for improvement, please submit an issue or PR. 

Current work on a seperate project will include a router based multi-sequential chain exploring LLMs with different topic retrievers for subject matter expert LLMs to discuss and help plan research. 
