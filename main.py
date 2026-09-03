from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI as ChatAzureOpenAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
import os


tavily_search = TavilySearch(max_results=5)


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query : The query to search for

    Returns:
        The search results as a string

    """
    print(f"Searching for : {query}")
    return str(tavily_search.invoke({"query": query}))

llm = ChatAzureOpenAI(azure_deployment=os.environ.get("AZURE_DEPLOYMENT_NAME"),
    azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
    api_key=os.environ.get("AZURE_API_KEY"),
    api_version="2024-02-01",
    temperature=0.3,)
#llm = ChatOllama(model="gemma3:270m", temperature=0.3)

tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?")})
    print(f"Agent Result: {result}")


if __name__ == "__main__":
    main()
