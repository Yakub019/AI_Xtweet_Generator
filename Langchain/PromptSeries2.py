from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv
load_dotenv()

import streamlit as st 

llm1= HuggingFaceEndpoint(repo_id="Qwen/Qwen3-8B",task="text generation",temperature=0.7)
model = ChatHuggingFace(llm=llm1)

topic= "artificial intelligence"
explanation_style = "The emotions give me in points"
explanation_length = "Two lines"

Template = load_prompt('Template.json')
chain = Template | model
result = chain.invoke({
    'topic':topic,
    'explanation_style':explanation_style,
    'explanation_length':explanation_length
})
print(result.content)

