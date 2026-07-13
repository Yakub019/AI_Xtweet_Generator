from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict,Literal,Annotated
from pydantic import BaseModel,Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

llm1 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text generation"
    ,temperature=0.7)
model1 = ChatHuggingFace(llm=llm1)

llm2 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    task="text generation"
    ,temperature=0.7)
model2 = ChatHuggingFace(llm=llm2)

llm3 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    task="text generation"
    ,temperature=0.7)
model3 = ChatHuggingFace(llm=llm3)


class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        description="Final evaluation result."
    )
    feedback: str = Field(
        description="Constructive feedback explaining why the tweet was approved or rejected."
    )


parser = PydanticOutputParser(pydantic_object=TweetEvaluation)

# state
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved","needs_improvement"]
    feedback: str
    iteration: int
    max_iterations: int

def generate_tweet(state: TweetState) -> TweetState:
    # prompt -> generator llm to generate detailed or desired tweet -> return response
    messages = [
    SystemMessage(
        content="""
You are a top AI educator and Twitter/X content creator.

Your goal is to write tweets that:
- Get high engagement (likes, reposts, bookmarks)
- Sound like a real developer documenting their journey
- Teach one useful idea clearly
- Create curiosity without using clickbait
- Are concise, conversational, and easy to read

Never sound like AI.
"""
    ),

    HumanMessage(
        content=f"""
Today's topic:

{state['topic']}

Write ONE tweet.

Rules:

- Maximum 280 characters.
- Start with a strong hook.
- Explain one idea only.
- Use simple English.
- Sound like a developer sharing what they learned today.
- Add a practical takeaway.
- Avoid emojis unless they improve the post.
- Avoid hashtags unless they genuinely help.
- Do NOT use question-answer format.
- Do NOT use marketing language.
- Avoid generic motivational quotes.
- Make readers want to bookmark or repost the tweet.
- Write naturally, as if posted by an engineer.

This is version {state['iteration'] + 1}.
"""
    )
     ]
    
    response = model1.invoke(messages).content
    return {'tweet': response}
    

def evaluate_tweet(state: TweetState) -> TweetState:

    messages = [
        SystemMessage(
            content=f"""
You are an expert Twitter/X content strategist specializing in AI, Machine Learning, and Software Engineering.

Your ONLY task is to evaluate tweets.

Be strict and objective.

Approve only tweets that are:
- Educational
- Technically accurate
- Engaging
- Natural sounding
- Likely to perform well on Twitter/X

Do NOT rewrite the tweet.

{parser.get_format_instructions()}
"""
        ),

        HumanMessage(
            content=f"""
Evaluate the following tweet.

Topic:
{state['topic']}

Tweet:
{state['tweet']}

Evaluation Criteria

1. Hook
- Does the first line grab attention?

2. Clarity
- Easy to understand?
- Avoids unnecessary jargon?

3. Educational Value
- Does the reader learn something useful?

4. Technical Accuracy
- Is the information correct?

5. Engagement Potential
- Likely to receive likes, reposts or bookmarks?

6. Readability
- Concise
- Easy to scan
- Natural English

7. Human Tone
- Sounds like a real developer documenting their learning.
- Does NOT sound AI generated.

8. Format
- Under 280 characters
- No clickbait
- No unnecessary emojis
- No unnecessary hashtags
- No question-answer format

Automatically reject if:
- Technical inaccuracies
- Weak hook
- Generic content
- AI sounding
- Over 280 characters
- Provides little educational value

Return your response using the required JSON format.
"""
        )
    ]

    response = model2.invoke(messages)

    result = parser.parse(response.content)

    return {"evaluation": result.evaluation, "feedback": result.feedback}



def optimize_tweet(state: TweetState) -> TweetState:
    messages = [
        SystemMessage(
            content="""
You are an expert Twitter/X content writer specializing in AI, Machine Learning, LangChain, LangGraph, LLMs, RAG, and AI Agents.

Your ONLY task is to improve an existing tweet based on reviewer feedback.

Do not change the topic.
Do not invent new facts.
Preserve the original meaning.

Your goal is to make the tweet:

- More engaging
- More educational
- Easier to read
- More likely to get likes, reposts, and bookmarks
- Natural and human-written
- Technically accurate

Return ONLY the improved tweet.
"""
        ),

        HumanMessage(
            content=f"""
Topic:
{state['topic']}

Current Tweet:
{state['tweet']}

Reviewer Feedback:
{state['feedback']}

Improve the tweet by addressing every point in the feedback.

Requirements:

- Keep it under 280 characters.
- Write like a real AI engineer sharing today's learning.
- Start with a stronger hook.
- Improve clarity.
- Increase educational value.
- Make the takeaway memorable.
- Remove unnecessary words.
- Avoid robotic language.
- Avoid clickbait.
- Avoid excessive emojis.
- Avoid unnecessary hashtags.
- Do NOT use question-answer format.
- Keep all technical information accurate.

Return ONLY the improved tweet.
"""
        )
    ]

def route_evaluation(state : TweetState) -> TweetState:
    if state["evaluation"] == "approved" or state["iteration"] >= state["max_iterations"]:
        return 'approved'
    else:
        return 'needs_improvement'

graph = StateGraph(TweetState)
graph.add_node('generate',generate_tweet)
graph.add_node('evaluate',evaluate_tweet)
graph.add_node('optimize',optimize_tweet)
graph.add_edge(START,'generate')
graph.add_edge('generate','evaluate')
graph.add_conditional_edges('evaluate',route_evaluation,{'approved':END,'needs_improvement':'optimize'})
graph.add_edge('optimize','evaluate')
workflow = graph.compile()


initial_state = {
    'topic': " differnce between langchain and langgraph in points",
    'iteration': 1,
    'max_iterations': 5
}
result = workflow.invoke(initial_state)
print(result)