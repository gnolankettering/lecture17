from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
import config

# model
model = ChatOpenAI(model="gpt-4", openai_api_key=config.OPENAI_API_KEY)

# tools
def search_on_google(query: str):
    return f"Jason Derulo doesn't have a wife or partner."

search_tool = Tool.from_function(
    func=search_on_google,
    name="FakeSearchTool",
    description="Useful for questions about Jason Derulo's partner.",
)
tools = [search_tool] 

# prompt
prompt = hub.pull("hwchase17/openai-functions-agent")

# agent
agent = create_openai_functions_agent(llm=model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"input": "Does Jason Derulo have a partner?"})
print(result)

