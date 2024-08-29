import re
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate
import config

text = """
Action: search_on_google
Action_Input: Tom Hanks current wife

action: search_on_wikipedia
action_input: How old is Rita Wilson in 2023

action : search_on_google
action input: some other query
"""

def extract_last_action_and_input(text):
    # Compile regex patterns
    action_pattern = re.compile(r"(?i)action\s*:\s*([^\n]+)", re.MULTILINE)
    action_input_pattern = re.compile(
        r"(?i)action\s*_*input\s*:\s*([^\n]+)", re.MULTILINE
    )

    # Find all occurrences of action and action_input
    actions = action_pattern.findall(text)
    action_inputs = action_input_pattern.findall(text)

    # Extract the last occurrence of action and action_input
    last_action = actions[-1] if actions else None
    last_action_input = action_inputs[-1] if action_inputs else None

    return {"action": last_action, "action_input": last_action_input}

print(extract_last_action_and_input(text))

def extract_final_answer(text):
    final_answer_pattern = re.compile(
        r"(?i)I've found the answer:\s*([^\n]+)", re.MULTILINE
    )
    final_answers = final_answer_pattern.findall(text)
    if final_answers:
        return final_answers[0]
    else:
        return None

final_answer_text = "I've found the answer: final_answer"
print(extract_final_answer(final_answer_text))

chat = ChatOpenAI(
    stop=["tool_result:"], openai_api_key=config.OPENAI_API_KEY
)

tools = {}

def search_on_google(query: str):
    return f"Jason Derulo doesn't have a wife or partner."

tools["search_on_google"] = {
    "function": search_on_google,
    "description": "Searches on google for a query",
}


base_prompt = """
You will attempt to solve the problem of finding the answer to a question.
Use chain of thought reasoning to solve through the problem, using the following pattern:

1. Observe the original question:
original_question: original_problem_text
2. Create an observation with the following pattern:
observation: observation_text
3. Create a thought based on the observation with the following pattern:
thought: thought_text
4. Use tools to act on the thought with the following pattern:
action: tool_name
action_input: tool_input

Do not guess or assume the tool results. Instead, provide a structured output that includes the action and action_input.

You have access to the following tools: {tools}.

original_problem: {question}
"""

model_output = chat.invoke(
    SystemMessagePromptTemplate.from_template(template=base_prompt).format_messages(
        tools=tools, question="Is Jason Derulo with a partner?"
    )
)
print(model_output)


# Extract the tool_name and tool_input from the model_output
tool_name = extract_last_action_and_input(model_output.content)["action"]
tool_input = extract_last_action_and_input(model_output.content)["action_input"]
tool_result = tools[tool_name]["function"](tool_input)

print(
    f"""
----------
The agent has opted to use the following tool:
tool_name: {tool_name}
tool_input: {tool_input}
tool_result: {tool_result}
----------
"""
)

current_prompt = """
Based on the provided tool result:
tool_result: {tool_result}

Either provide the next observation, action, action_input, or the final answer if available.
If you are providing the final answer, you must return the following pattern:
"I've found the answer: final_answer" """

print("The second prompt shows", current_prompt)

model_output = chat(
    SystemMessagePromptTemplate.from_template(template=current_prompt).format_messages(
        tool_result=tool_result
    )
)

print("----------\n\nThe model output is:", model_output.content)
# See if there is a final answer:
final_answer = extract_final_answer(model_output.content)
if final_answer:
    print(f"answer: {final_answer}")
else:
    print("No final answer found.")

