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
# agent_executor.invoke("Identify all of the tables")

user_sql = agent_executor.invoke(
    "Add 5 new users to the database. Their names are: John, Mary, Peter, Paul, and Jane. \
        Run the following SQL query against the database and add the users."
)
print(user_sql)

SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
If the question does not seem related to the database, just return "I don't know" as the answer.
"""

# Updating the agent_executor to include DML statements:
agent_executor = create_sql_agent(
    llm=ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY),
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    prefix=SQL_PREFIX,
)

agent_executor.invoke(user_sql)
agent_executor.invoke("Do we have a Peter in the database?")
