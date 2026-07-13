from langchain_community.document_loaders import TextLoader
loader = TextLoader("poem.txt",autodetect_encoding=True)
docs = loader.load()
print(type(docs))
print(len(docs))
print(docs[0].metadata)
print(type(docs[0]))

