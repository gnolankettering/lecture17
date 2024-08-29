from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_openai.chat_models import ChatOpenAI
import config

db = SQLDatabase.from_uri("sqlite:///./data/demo.db")
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY))

agent_executor = create_sql_agent(
    llm=ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

agent_executor.invoke("Identify all of the tables")
