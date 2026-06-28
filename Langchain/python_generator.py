from langchain_core.prompts import PromptTemplate
Template = PromptTemplate(
    template ="""
Give me the five emotions of joke of topic,{topic}
    explanation_style:{explanation_style}
    explanation_length:{explanation_length}

""",
input_variables=['topic','explanation_style','explanation_length']
)

Template.save('Template.json')