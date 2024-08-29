# Importing the relevant packages:
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai.chat_models import ChatOpenAI
import config

# Creating a CSV Agent:
agent = create_csv_agent(
    ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY),
    "data/heart_disease_uci.csv",
    verbose=True,
    allow_dangerous_code=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent.invoke("How many rows of data are in the file?")
# '920'

agent.invoke("What are the columns within the dataset?")
# "'id', 'age', 'sex', 'dataset', 'cp', 'trestbps', 'chol', 'fbs',
# 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'"

agent.invoke("Create a correlation matrix for the data and save it to a file.")
# "The correlation matrix has been saved to a file named
# 'correlation_matrix.csv'."
