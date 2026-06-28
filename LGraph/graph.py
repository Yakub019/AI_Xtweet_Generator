from langgraph.graph import StateGraph, START, END
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import operator
import json

load_dotenv()

llm1 = HuggingFaceEndpoint(model="Qwen/Qwen3-235B-A22B", task="text-generation", temperature=0.5)
llm2 = HuggingFaceEndpoint(model="Qwen/Qwen3-8B", task="text-generation", temperature=0.5)
model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

essay = "Climate change is the defining crisis of our generation. Rising temperatures, melting glaciers, and extreme weather events are clear indicators that human activity is disrupting the planet's natural balance."

class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

class UPSC_State(TypedDict):
    essay: str
    cot_feedback: str
    analysis_feedback: str
    language_feedback: str
    overall_feedback: str
    individual_score: Annotated[list[int], operator.add]
    avg_score: float

def invoke_structured(prompt: str) -> EvaluationSchema:
    """Helper: invoke model and parse JSON manually"""
    result = model2.invoke(prompt)  # use model2 (8B) - more reliable on free tier
    raw = result.content.strip()
    print("RAW:", raw[:200])  # debug
    if "```" in raw:
        raw = raw.split("```json")[-1].split("```")[0].strip()
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
    return {"cot_feedback": output.feedback, "individual_score": [output.score]}

def evaluate_analysis(state: UPSC_State):
    prompt = f"""Evaluate the depth of analysis of the following essay.
Respond ONLY with this exact JSON, no other text:
{{"feedback": "your feedback here", "score": 7}}

Essay: {state["essay"]}"""
    output = invoke_structured(prompt)
    return {"analysis_feedback": output.feedback, "individual_score": [output.score]}

def evaluate_language(state: UPSC_State):
    prompt = f"""Evaluate the language quality of the following essay.
Respond ONLY with this exact JSON, no other text:
{{"feedback": "your feedback here", "score": 7}}

Essay: {state["essay"]}"""
    output = invoke_structured(prompt)
    return {"language_feedback": output.feedback, "individual_score": [output.score]}

def final_evaluation(state: UPSC_State):
    prompt = f"""Based on these feedbacks, write a summarized overall feedback.
Respond ONLY with this exact JSON, no other text:
{{"feedback": "summarized feedback here", "score": 7}}

Language feedback: {state["language_feedback"]}
Analysis feedback: {state["analysis_feedback"]}
Clarity feedback: {state["cot_feedback"]}"""
    output = invoke_structured(prompt)
    scores = state["individual_score"]
    avg = sum(scores) / len(scores) if scores else 0
    return {"overall_feedback": output.feedback, "individual_score": [output.score], "avg_score": avg}

# Build graph
graph = StateGraph(UPSC_State)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("final_evaluation", final_evaluation)

# Parallel edges
graph.add_edge(START, "evaluate_thought")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_language")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()
initial_state = {"essay": essay}
result = workflow.invoke(initial_state)
print("Overall Feedback:", result["overall_feedback"])
print("All Scores:", result["individual_score"])
print("Avg Score:", result["avg_score"])