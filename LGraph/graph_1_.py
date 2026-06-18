from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from typing import TypedDict
load_dotenv()
class BMIState(TypedDict):
    weight: float
    height: float
    bmi: float

graph = StateGraph(BMIState)
def calculate_bmi(state:BMIState)->BMIState:
    weight=state["weight"]
    height=state["height"]
    bmi=weight/(height**2)
    state["bmi"]=round(bmi,2)
    return state 
graph.add_node("calculate_bmi",calculate_bmi)
graph.add_edge(START,"calculate_bmi")
graph.add_edge("calculate_bmi",END)
workflow = graph.compile()
input_data = {"weight": 70, "height": 1.75}
result = workflow.invoke(input_data)
print(result)