import logging
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from tavily import TavilyClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

load_dotenv()
tavily_client = TavilyClient()


@tool
def web_search(query: str) -> dict[str, Any]:
    """tool for searching the web for information"""
    logging.info(f"searching for {query}")
    return tavily_client.search(query)


@tool
def print_document(name: str) -> str:
    """use this tool to generate the proper document for your student"""
    return f"this is the document of {name}"


agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search],
)

response = agent.invoke(
    {"messages": [HumanMessage(content="give me the latest news of the day")]}
)
print("AI response")
print(response["messages"][-1].content)

print("--------------details -------------------")
print(response["messages"])
