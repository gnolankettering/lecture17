from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
import config

# 1. Create the model and memory:
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=config.OPENAI_API_KEY)
memory = ConversationBufferMemory(return_messages=True)

# 2. Create the tools:
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

tools = [get_word_length]

# 3. Create the Prompt:
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events and isn't good at calculating word length.",
        ),
        # This is where the agent will write/read its messages from
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ]
)

# 4. Formats the python function tools into JSON schema and binds them to the model:
llm_with_tools = model.bind_tools(tools=[convert_to_openai_tool(t) for t in tools])

from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

# 5. Setting up the agent chain:
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "history": RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent = create_openai_tools_agent(llm_with_tools, tools, prompt)

# Create an agent executor by passing in the agent, tools and memory:
memory = ConversationBufferMemory(return_messages=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

agent_executor.invoke({"input": "My name is James"})

