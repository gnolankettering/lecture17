import langchain
from langchain.chains import LLMMathChain
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import StructuredTool
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import ToolException
import config

langchain.debug = True

# Initialize the ChatOpenAI with temperature set to 0:
model = ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY)

# Create a LLMMathChain instance using the ChatOpenAI model:
llm_math_chain = LLMMathChain.from_llm(llm=model, verbose=True)

# Download the prompt from the hub:
prompt = hub.pull("hwchase17/openai-functions-agent")

# Define a calculator function that uses the LLMMathChain to answer math questions:
def calculator_function(question: str) -> str:
    """Useful for when you need to answer questions about math."""
    return llm_math_chain.run(question)

# Create a StructuredTool from the calculator function:
tools = [
    StructuredTool.from_function(
        func=calculator_function,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        return_direct=True,
    ),
]

# Create an agent using the ChatOpenAI model and the tools:
agent = create_openai_functions_agent(llm=model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({"input": "What is 5 + 5?"})
# result = agent_executor.invoke({"input": "What is the product of the 998th, 999th, and 1000th prime numbers?"})
print(result)