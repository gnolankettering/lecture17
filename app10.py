import asyncio
from langchain.callbacks import get_openai_callback
from langchain_core.messages import SystemMessage
from langchain_openai.chat_models import ChatOpenAI
import config


model = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY)

with get_openai_callback() as cb:
    model.invoke([SystemMessage(content="My name is James")])
total_tokens = cb.total_tokens
print(total_tokens)
assert total_tokens > 0

with get_openai_callback() as cb:
    model.invoke([SystemMessage(content="My name is James")])
    model.invoke([SystemMessage(content="My name is James")])
assert cb.total_tokens == total_tokens * 2

# Async callbacks:
with get_openai_callback() as cb:
    await asyncio.gather(
        model.agenerate(
            [
                [SystemMessage(content="Is the meaning of life 42?")],
                [SystemMessage(content="Is the meaning of life 42?")],
            ],
        )
    )
print(cb.__dict__)
# {'successful_requests': 2, 'total_cost': 0.000455,
# 'total_tokens': 235, 'prompt_tokens': 30,
# 'completion_tokens': 205}



