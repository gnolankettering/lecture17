from langchain.memory import ConversationBufferMemory
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter
import config

# memory = ConversationBufferMemory(return_messages=True)
# memory.save_context({"input": "hi"}, {"output": "whats up"})
# memory.load_memory_variables({})

memory = ConversationBufferMemory(return_messages=True)

model = ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Act as a chatbot that helps users with their queries."),
        MessagesPlaceholder(variable_name="history"), # The history of the conversation
        ("human", "{input}"),
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

inputs = {"input": "Hi my name is James!"}
result = chain.invoke(inputs)
memory.save_context(inputs, {"outputs": result})

memory.load_memory_variables({})

inputs = {"input": "What is my name?"}
second_result = chain.invoke(inputs)
print(second_result)


