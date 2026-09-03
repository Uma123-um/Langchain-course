from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel,Field

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI as ChatAzureOpenAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
import os

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources"""

    answer : str = Field(description="The agent's answer to the query")
    sources : List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")








llm = ChatAzureOpenAI(azure_deployment=os.environ.get("AZURE_DEPLOYMENT_NAME"),
    azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
    api_key=os.environ.get("AZURE_API_KEY"),
    api_version="2024-02-01",
    temperature=0.3)
#llm = ChatOllama(model="gemma3:270m", temperature=0.3)

tools = [TavilySearch(max_results=5)]
agent = create_agent(model=llm, tools=tools,response_format=AgentResponse)



def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?")})
    print(f"Agent Result: {result}")


if __name__ == "__main__":
    main()
