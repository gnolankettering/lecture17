from langchain.memory import ConversationBufferMemory
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
import config

model = ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY)
memory = ConversationBufferMemory(return_messages=True)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a very powerful assistant, but don't know current events
and aren't good at calculating word length.""",
        ),
        # This is where the agent will write/read its messages from
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ]
)
chain = (
    {
        "input": lambda x: x["input"],
        "history": RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
    }
    | prompt
    | model
    | StrOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,memory=memory)

inputs = {"input": "Hi my name is James!"}
result = chain.invoke(inputs)
memory.save_context(inputs, {"outputs": result})

memory.load_memory_variables({})

inputs = {"input": "What is my name?"}
second_result = chain.invoke(inputs)
print(second_result)


