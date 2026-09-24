import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


# ============================================================
# 1. DEFINE CAREERPILOT TOOLS
# ============================================================

@tool
def generate_dsa_questions(topic: str) -> str:
    """Generate placement-level DSA questions based on a topic."""

    prompt = f"""
You are CareerPilot AI, an AI placement preparation assistant.

Generate 5 DSA questions on the following topic:

{topic}

Difficulty:
- 2 Easy
- 2 Medium
- 1 Interview-level

For every question provide:
1. Question
2. Difficulty
3. Expected concept

Do not provide solutions.

Keep the questions suitable for college placement preparation.
"""

    response = llm.invoke(prompt)
    return response.text


@tool
def generate_java_questions(topic: str) -> str:
    """Generate placement-level Java interview questions."""

    prompt = f"""
You are CareerPilot AI, an expert Java placement interviewer.

Generate 5 Java interview questions on:

{topic}

Difficulty:
- 2 Easy
- 2 Medium
- 1 Interview-level

For every question provide:
1. Question
2. Difficulty
3. Expected concept

Focus on concepts commonly tested in Java technical interviews.

Do not provide solutions.
"""

    response = llm.invoke(prompt)
    return response.text


@tool
def evaluate_answer(question: str, student_answer: str) -> str:
    """Evaluate a student's answer to a placement interview question."""

    prompt = f"""
You are a technical interviewer evaluating a student.

Question:
{question}

Student's Answer:
{student_answer}

Evaluate the answer using this structure:

1. Correctness
2. What the student did well
3. What is wrong or missing
4. Correct explanation
5. Interview tip
6. Score out of 10

Be honest but beginner-friendly.
"""

    response = llm.invoke(prompt)
    return response.text


@tool
def create_study_plan(goal: str, days: int) -> str:
    """Create a practical day-by-day placement preparation plan."""

    prompt = f"""
You are CareerPilot AI, an AI placement preparation assistant.

Create a practical placement preparation study plan.

Goal:
{goal}

Number of days:
{days}

For each day include:

1. Topics to study
2. Practice tasks
3. Coding/DSA practice
4. Revision task

Keep the plan realistic for a college student.
Gradually increase the difficulty.
"""

    response = llm.invoke(prompt)
    return response.text


# ============================================================
# 2. INITIALIZE GEMINI
# ============================================================

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 3. CREATE CAREERPILOT AGENT
# ============================================================

tools = [
    generate_dsa_questions,
    generate_java_questions,
    evaluate_answer,
    create_study_plan
]


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are CareerPilot AI, an AI placement preparation assistant.

Your purpose is to help students prepare for technical placements.

You have access to these capabilities:

1. DSA Question Generator
   - Arrays
   - Strings
   - Linked Lists
   - Stacks
   - Queues
   - Trees
   - Graphs
   - Sorting
   - Searching
   - Algorithms
   - Coding questions

2. Java Interview Question Generator
   - OOP
   - Classes and objects
   - Inheritance
   - Polymorphism
   - Abstraction
   - Encapsulation
   - Collections
   - Exceptions
   - Multithreading
   - Strings
   - Other Java interview concepts

3. Answer Evaluator
   - Evaluate technical interview answers
   - Identify mistakes
   - Explain missing concepts
   - Give interview tips
   - Give a score out of 10

4. Study Planner
   - Placement preparation plans
   - Day-by-day schedules
   - DSA + Java preparation plans
   - Revision plans

Use the appropriate tool whenever the student's request matches one
of these capabilities.

For general placement-related questions, answer directly.

Keep responses clear, practical and beginner-friendly.
"""
)


# ============================================================
# 4. FORMAT INPUT FOR THE AGENT
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(description="Your message to CareerPilot")


def format_for_agent(x) -> dict:

    user_input = x["input"] if isinstance(x, dict) else x.input

    return {
        "messages": [
            ("user", user_input)
        ]
    }


# ============================================================
# 5. EXTRACT FINAL RESPONSE
# ============================================================

def extract_text_response(agent_output: dict) -> str:

    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:

        for value in agent_output.values():

            if isinstance(value, dict) and "messages" in value:

                messages = value["messages"]
                break

    if messages:

        last = messages[-1]

        return getattr(
            last,
            "content",
            str(last)
        )

    return str(agent_output)


# ============================================================
# 6. CREATE LANGSERVE CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 7. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="CareerPilot AI",
    version="1.0",
    description=(
        "An AI placement preparation agent "
        "using Gemini, LangChain tools and LangServe."
    )
)


@app.get("/")
def root():

    return {
        "message": "CareerPilot AI is running!",
        "website": "/agent/playground/",
        "docs": "/docs"
    }


# ============================================================
# 8. ADD LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# ============================================================
# 9. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )