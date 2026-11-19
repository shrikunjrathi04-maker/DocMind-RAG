from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document loaders/rcrm.pdf")

docs = data.load()

print(len(docs))
