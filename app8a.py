from langchain.tools import StructuredTool

def save_interview(raw_interview_text: str):
    """Tool to save the interview. You must pass the entire interview and
    conversation in here. The interview will then be saved to a local file.
    Remember to include all of the previous chat messages. Include all of
    the messages with the user and the AI, here is a good response:
    AI: some text
    Human: some text
    ...
    ---
    """
    # Save to local file:
    with open("interview.txt", "w") as f:
        f.write(raw_interview_text)
    return f'''Interview saved! Content: {raw_interview_text}. File:
    interview.txt. You must tell the user that the interview is saved.'''

save_interview = StructuredTool.from_function(save_interview)

