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
# 1. GEMINI
# ============================================================

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 2. TOOLS
# ============================================================

@tool
def generate_dsa_questions(topic: str) -> str:
    """Generate 5 placement-level DSA questions on a topic."""

    prompt = f"""
Generate 5 DSA questions on {topic}.

Difficulty:
- 2 Easy
- 2 Medium
- 1 Interview-level

For each question provide:
1. Question
2. Difficulty
3. Expected concept

Do not provide solutions.
"""

    response = llm.invoke(prompt)
    return response.content


@tool
def generate_java_questions(topic: str) -> str:
    """Generate 5 Java interview questions on a topic."""

    prompt = f"""
Generate 5 Java interview questions on {topic}.

Difficulty:
- 2 Easy
- 2 Medium
- 1 Interview-level

For each question provide:
1. Question
2. Difficulty
3. Expected concept

Do not provide solutions.
"""

    response = llm.invoke(prompt)
    return response.content


@tool
def create_study_plan(goal: str) -> str:
    """Create a placement preparation study plan."""

    prompt = f"""
Create a practical placement preparation plan for:

Goal:
{goal}

Include:
1. Topics to study
2. Practice tasks
3. DSA/coding practice
4. Revision

Keep it realistic for a college student.
"""

    response = llm.invoke(prompt)
    return response.content


@tool
def evaluate_answer(answer: str) -> str:
    """Evaluate a student's technical interview answer."""

    prompt = f"""
You are a technical interviewer.

Evaluate this student's answer:

{answer}

Give:
1. Correctness
2. What was done well
3. What is wrong or missing
4. Correct explanation
5. Interview tip
6. Score out of 10

Be honest and beginner-friendly.
"""

    response = llm.invoke(prompt)
    return response.content


tools = [
    generate_dsa_questions,
    generate_java_questions,
    create_study_plan,
    evaluate_answer
]


# ============================================================
# 3. CAREERPILOT AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are CareerPilot AI, an AI placement preparation assistant.

Help students prepare for technical placements.

You can:
- Generate DSA questions
- Generate Java interview questions
- Evaluate interview answers
- Create placement study plans

Use the appropriate tool when needed.

For normal placement questions, answer directly.

Keep responses clear, practical and beginner-friendly.
"""
)


# ============================================================
# 4. INPUT
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
# 5. OUTPUT
# ============================================================

def extract_text_response(agent_output: dict) -> str:

    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages:
        last = messages[-1]
        content = getattr(last, "content", None)

        if content:
            return str(content)

    return str(agent_output)


# ============================================================
# 6. LANGSERVE CHAIN
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
# 7. FASTAPI
# ============================================================

app = FastAPI(
    title="CareerPilot AI",
    version="1.0",
    description="An AI placement preparation agent using Gemini, LangChain tools and LangServe."
)


@app.get("/")
def root():
    return {
        "message": "CareerPilot AI is running!",
        "website": "/agent/playground/",
        "docs": "/docs"
    }


# ============================================================
# 8. LANGSERVE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# ============================================================
# 9. RUN
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
