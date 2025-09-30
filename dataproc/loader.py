def prepare_data():
    return


'''

def prepare_data(pdf_path="data/raw/spri_ai_brief.pdf"):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 4})
'''