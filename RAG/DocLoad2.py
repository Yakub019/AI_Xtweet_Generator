from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-8B",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=2048,
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="write a summary of the following poem in points: \n{poem}",
    input_variables=["poem"]
)

loader = TextLoader("poem.txt", encoding="utf-8")

docs = loader.load()


parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"poem": docs[0].page_content})
print(result)

