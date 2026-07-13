from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import operator
import json

load_dotenv()

llm1 = HuggingFaceEndpoint(model="Qwen/Qwen3-8B", task="text-generation", temperature=0.5)
model1 = ChatHuggingFace(llm=llm1)

essay = "Climate change is the defining crisis of our generation. Rising temperatures, melting glaciers, and extreme weather events are clear indicators that human activity is disrupting the planet's natural balance. Governments, corporations, and individuals all share responsibility for reducing carbon emissions. Transitioning to renewable energy, adopting sustainable practices, and investing in green technology are no longer optional — they are urgent necessities for survival."

class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed Feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

class UPSC_State(TypedDict):
    essay: str
    cot_feedback: str
    analysis_feedback: str
    langguage_feedback: str
    overall_feedback: str
    individual_score: Annotated[list[int], operator.add]
    avg_score: float

def invoke_structured(prompt: str) -> EvaluationSchema:
    result = model1.invoke(prompt)
    raw = result.content.strip()
    print("RAW:", raw[:300])
    # Strip <think>...</think> tags (Qwen3 thinking mode)
    if "<think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    # Strip markdown code blocks
    if "```" in raw:
        raw = raw.split("```json")[-1].split("```")[0].strip()
    # Extract JSON object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    raw = raw[start:end]
    parsed = json.loads(raw)
    return EvaluationSchema(**parsed)

def evaluate_thought(state: UPSC_State):
    prompt = f"""Evaluate the clarity of thought of the following essay.
Respond ONLY with this exact JSON, no other text:
{{"feedback": "your feedback here", "score": 7}}

Essay: {state["essay"]}"""
    output = invoke_structured(prompt)
    return {'cot_feedback': output.feedback, 'individual_score': [output.score]}

def evaluate_analysis(state: UPSC_State):
    prompt = f"""Evaluate the depth of analysis of the following essay.
Respond ONLY with this exact JSON, no other text:
{{"feedback": "your feedback here", "score": 7}}

Essay: {state["essay"]}"""
    output = invoke_structured(prompt)
    return {'analysis_feedback': output.feedback, 'individual_score': [output.score]}

def evaluate_langguage(state: UPSC_State):
    prompt = f"""Evaluate the language quality of the following essay.
Respond ONLY with this exact JSON, no other text:
{{"feedback": "your feedback here", "score": 7}}

Essay: {state["essay"]}"""
    output = invoke_structured(prompt)
    return {'langguage_feedback': output.feedback, 'individual_score': [output.score]}

def final_evaluation(state: UPSC_State):
    prompt = f"""Summarize these feedbacks into one overall feedback.
Respond ONLY with this exact JSON, no other text:
{{"feedback": "your feedback here", "score": 7}}

Language feedback: {state["langguage_feedback"]}
Analysis feedback: {state["analysis_feedback"]}
Clarity feedback: {state["cot_feedback"]}"""
    output = invoke_structured(prompt)
    scores = state["individual_score"]
    avg = sum(scores) / len(scores)
    return {'overall_feedback': output.feedback, 'individual_score': [output.score], 'avg_score': avg}

graph = StateGraph(UPSC_State)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_langguage', evaluate_langguage)
graph.add_node('final_evaluation', final_evaluation)

graph.add_edge(START, 'evaluate_thought')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_langguage')
graph.add_edge('evaluate_thought', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_langguage', 'final_evaluation')
graph.add_edge('final_evaluation', END)

workflow = graph.compile()
result = workflow.invoke({'essay': essay})

print("\n--- RESULTS ---")
print("COT Feedback:", result["cot_feedback"])
print("Analysis Feedback:", result["analysis_feedback"])
print("Language Feedback:", result["langguage_feedback"])
print("Overall Feedback:", result["overall_feedback"])
print("All Scores:", result["individual_score"])
print("Avg Score:", result["avg_score"])