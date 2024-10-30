from ai import client
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage


def __init__():

    completion: ChatCompletion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "hello world!",
            }
        ],
        model="gpt-3.5-turbo",
    )
    reply: ChatCompletionMessage = completion.choices[0].message
    print(reply.to_json())
