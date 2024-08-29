from langchain.agents import create_openai_tools_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain.tools import StructuredTool
from langchain_community.utilities.serpapi import SerpAPIWrapper
from pydantic.v1 import BaseModel
from typing import Optional, Type, Union, Literal, List
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import ToolException
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
from langchain.agents import AgentExecutor
import tempfile
import requests
import config

SERPAPI_API_KEY = config.SERPAPI_API_KEY

# Define Tools:
def google_search(query: str):
    """Tool to search Google"""
    return SerpAPIWrapper().run(query)

def save_interview(raw_interview_text: str):
    """Tool to save the interview. You must pass the entire interview and conversation in here.
    The interview will then be saved to a local file. Remember to include all of the previous chat messages.
    Include all of the messages with the user and the AI, here is a good response:
    AI: some text
    Human: some text
    ...
    ---
    """
    # Save to local file:
    with open("interview.txt", "w") as f:
        f.write(raw_interview_text)
    return f"Interview saved! Content: {raw_interview_text}. File: interview.txt. You must tell the user that the interview is saved."

def get_pokemon(pokemon_name: str):
    """Tool to get a single Pokemon"""
    import requests
    clean_pokemon_name = pokemon_name.lower()
    # Get the Pokemon
    try:
        pokemon = requests.get(f"https://pokeapi.co/api/v2/pokemon/{clean_pokemon_name}").json()
        if pokemon:
            return f"Pokemon found! Name: {pokemon['name']}. Height: {pokemon['height']}. Weight: {pokemon['weight']}."
        else:
            return "Pokemon not found!"
    except Exception as e:
        print(e)
        return "Pokemon not found! Try a different pokemon."

# Convert functions to Structured Tools:
google_search = StructuredTool.from_function(google_search)
save_interview = StructuredTool.from_function(save_interview)
get_single_pokemon = StructuredTool.from_function(get_pokemon)

class ArgumentType(BaseModel):
    url: str
    file_type: Union[Literal["pdf"], Literal["txt"]]

class SummarizeFileFromURL(BaseTool):
    name = "SummarizeFileFromURL"
    description = "Summarize a file from a URL."
    args_schema: Type[ArgumentType] = ArgumentType

    def _run(
        self,
        url: str,
        file_type: Union[Literal["pdf"], Literal["txt"]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
        from langchain.chains.summarize import load_summarize_chain

        response = requests.get(url)
        if response.status_code != 200:
            raise ToolException("The URL is not valid. Please try another URL.")

        if file_type == "pdf":
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                tmp.write(response.content)
                pdf_loader = PyPDFLoader(file_path=tmp.name)
                documents: List[Document] = pdf_loader.load()

        else:
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=True, mode="w+t") as tmp:
                tmp.write(response.text)
                text_loader = TextLoader(tmp.name)
                documents: List[Document] = text_loader.load()

        chain = load_summarize_chain(llm=ChatOpenAI(), chain_type="map_reduce")
        return chain.run(documents)

    async def _arun(
        self,
        url: str,
        file_type: Union[Literal["pdf"], Literal["txt"]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("This tool does not support async yet!")


def _handle_error(error: ToolException) -> str:
    return (
        "The following errors occurred during tool execution:"
        + error.args[0]
        + "Please try another tool."
    )

file_summarizer = SummarizeFileFromURL()
file_summarizer.handle_tool_error = _handle_error

# Tools:
tools = [google_search, save_interview, get_single_pokemon, file_summarizer]

# Create the llm:
llm = ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY)

# Create the ChatPromptTemplate:
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define the system message:
system_message = SystemMessage(
    content="""Act as a helpful AI interview assistant,
               Always ask a follow-up question, especially after returning results.
               Ask one question at a time.
               Always be in interview mode or tool mode.
               Ask the user about the desired topic.
               """
)

# Initialize memory:
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Create the Prompt:
prompt = ChatPromptTemplate.from_messages(
    [
        system_message,
        MessagesPlaceholder(variable_name="history"), # This is where the user will write/read their messages from as memory
        ("user", "{input}"),
        MessagesPlaceholder(
            variable_name="agent_scratchpad"
        ),  # This is where the agent will write/read its messages from
    ]
)

agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

# Formats the python function tools into JSON schema and binds them to the model:
llm_with_tools = llm.bind_tools(tools)

from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser


# Setting up the agent chain:
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

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

agent_executor.invoke({"input": "My name is James, I'm a computer programmer and I like to play cosy games."})
agent_executor.invoke({"input": "What is my name and my profession?"})
agent_executor.invoke({"input": "Get information about Pikachu"})
agent_executor.invoke({"input": "Get information about Charmander"})
agent_executor.invoke(
    {
        "input": "I want a comprehensive summary of our conversation! Please can you save it?"
    }
)

try:
    agent_executor.invoke({"input":"I want to summarize this https://storage.googleapis.com/oreilly-content/NutriFusion%20Foods%20Marketing%20Plan%202022.docx"})
except Exception as e:
    error_message = str(e)
    chat = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY) 
    clean_error_message = chat.invoke([SystemMessage(content=error_message + 'Clean the error message, this will be shown to a user.')])
    print(clean_error_message.content)

# This line will take several minutes as we are summarizing a 41 page .pdf file:
agent_executor.invoke(
    {
        "input": "I want to summarize this https://storage.googleapis.com/oreilly-content/pokemon-rules.pdf "
    }
)
