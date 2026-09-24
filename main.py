import os

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from typing import TypedDict


# =========================
# Gemini Configuration
# =========================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)


# =========================
# CareerPilot Tools
# =========================

@tool
def generate_dsa_questions(topic: str):
    """Generate 5 placement-level DSA questions."""

    prompt = f"""
You are CareerPilot AI, an AI placement preparation assistant.

Generate 5 DSA questions on {topic}.

Difficulty:
- 2 Easy
- 2 Medium
- 1 Interview-level

For every question provide:
1. Question
2. Difficulty
3. Expected concept

Do not provide solutions.
"""

    response = llm.invoke(prompt)
    return response.text


@tool
def generate_java_questions(topic: str):
    """Generate 5 placement-level Java interview questions."""

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

Focus on concepts commonly tested in technical interviews.

Do not provide solutions.
"""

    response = llm.invoke(prompt)
    return response.text


@tool
def evaluate_answer(question: str, student_answer: str):
    """Evaluate a student's placement interview answer."""

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
def create_study_plan(goal: str, days: int):
    """Create a day-by-day placement preparation plan."""

    prompt = f"""
You are CareerPilot AI, an AI placement preparation assistant.

Create a practical study plan.

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
"""

    response = llm.invoke(prompt)
    return response.text


# =========================
# State
# =========================

class CareerPilotState(TypedDict):
    user_request: str
    question: str
    student_answer: str
    response: str


# =========================
# Manager
# =========================

def careerpilot_manager(state: CareerPilotState):

    user_request = state["user_request"]

    prompt = f"""
You are CareerPilot AI, an AI placement preparation manager.

Student request:
{user_request}

Decide which capability is needed.

Available capabilities:

1. DSA_QUESTIONS
2. JAVA_QUESTIONS
3. EVALUATE_ANSWER
4. STUDY_PLAN
5. GENERAL_GUIDANCE

Return ONLY one of:

DSA_QUESTIONS
JAVA_QUESTIONS
EVALUATE_ANSWER
STUDY_PLAN
GENERAL_GUIDANCE
"""

    response = llm.invoke(prompt)

    return {
        "response": response.text.strip()
    }


# =========================
# Agents
# =========================

def dsa_agent(state: CareerPilotState):

    result = generate_dsa_questions.invoke({
        "topic": state["user_request"]
    })

    return {"response": result}


def java_agent(state: CareerPilotState):

    result = generate_java_questions.invoke({
        "topic": state["user_request"]
    })

    return {"response": result}


def evaluator_agent(state: CareerPilotState):

    result = evaluate_answer.invoke({
        "question": state["question"],
        "student_answer": state["student_answer"]
    })

    return {"response": result}


def study_planner_agent(state: CareerPilotState):

    prompt = f"""
You are CareerPilot AI.

The student wants a study plan.

Student request:
{state["user_request"]}

Create a practical placement preparation study plan.

If the student gives a number of days, use that number.
If no number of days is given, create a 7-day plan.

For each day include:
1. Study
2. Practice
3. Coding
4. Revision

Keep it realistic for a college student.
"""

    response = llm.invoke(prompt)

    return {"response": response.text}


def general_agent(state: CareerPilotState):

    prompt = f"""
You are CareerPilot AI, a placement preparation assistant.

Student request:
{state["user_request"]}

Give a clear, practical and beginner-friendly answer.
Focus on placement preparation.
"""

    response = llm.invoke(prompt)

    return {"response": response.text}


# =========================
# Router
# =========================

def route_request(state: CareerPilotState):

    decision = state["response"].strip()

    if decision == "DSA_QUESTIONS":
        return "dsa_agent"

    elif decision == "JAVA_QUESTIONS":
        return "java_agent"

    elif decision == "EVALUATE_ANSWER":
        return "evaluator_agent"

    elif decision == "STUDY_PLAN":
        return "study_planner_agent"

    return "general_agent"


# =========================
# LangGraph
# =========================

workflow = StateGraph(CareerPilotState)

workflow.add_node("manager", careerpilot_manager)
workflow.add_node("dsa_agent", dsa_agent)
workflow.add_node("java_agent", java_agent)
workflow.add_node("evaluator_agent", evaluator_agent)
workflow.add_node("study_planner_agent", study_planner_agent)
workflow.add_node("general_agent", general_agent)

workflow.add_edge(START, "manager")

workflow.add_conditional_edges(
    "manager",
    route_request,
    {
        "dsa_agent": "dsa_agent",
        "java_agent": "java_agent",
        "evaluator_agent": "evaluator_agent",
        "study_planner_agent": "study_planner_agent",
        "general_agent": "general_agent"
    }
)

workflow.add_edge("dsa_agent", END)
workflow.add_edge("java_agent", END)
workflow.add_edge("evaluator_agent", END)
workflow.add_edge("study_planner_agent", END)
workflow.add_edge("general_agent", END)

careerpilot = workflow.compile()


# =========================
# FastAPI
# =========================

app = FastAPI(
    title="CareerPilot AI",
    description="AI Placement Preparation Agent",
    version="1.0"
)


class CareerPilotRequest(BaseModel):
    user_request: str
    question: str = ""
    student_answer: str = ""


@app.get("/")
def home():
    return {
        "message": "CareerPilot AI is running!",
        "status": "online"
    }


@app.post("/ask")
def ask_careerpilot(request: CareerPilotRequest):

    result = careerpilot.invoke({
        "user_request": request.user_request,
        "question": request.question,
        "student_answer": request.student_answer,
        "response": ""
    })

    return {
        "response": result["response"]
    }