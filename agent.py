import os
from dataclasses import dataclass
import json
import html
import stackexchange
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.shared_params.function_definition import FunctionDefinition
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
    Function,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat_model import ChatModel
from ai import client


@dataclass
class AgentParams:
    name: str
    model: ChatModel
    max_questions: int


class AgentState:
    gas: int
    prompt: str
    messages: list[ChatCompletionMessageParam]

    def __init__(
        self,
        gas: int,
        prompt: str,
    ):
        self.gas = gas
        self.prompt = prompt
        self.messages = [ChatCompletionUserMessageParam(role="user", content=prompt)]


class AgentException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class Agent:
    params: AgentParams
    state: AgentState

    def __init__(self, params: AgentParams, state: AgentState) -> None:
        self.params = params
        self.state = state

    def run(self):
        tools: list[ChatCompletionToolParam] = [
            ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name="query_stackoverflow",
                    description="Makes a query to StackOverflow for related questions and answers.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query_string": {
                                "type": "string",
                                "description": "The query string to send to StackOverflow. This should just be a space-separated list of the most important keywords rather than a fully-formed question.",
                            }
                        },
                        "required": ["query_string"],
                        "additionalProperties": False,
                    },
                    strict=True,
                ),
            )
        ]

        while self.state.gas > 0:
            self.state.gas -= 1

            completion: ChatCompletion = client.chat.completions.create(
                model=self.params.model, messages=self.state.messages, tools=tools
            )

            if len(completion.choices) == 0:
                raise AgentException("len(completion.choices) == 0")
            choice: Choice = completion.choices[0]

            message: ChatCompletionMessage = choice.message

            print(f"[assistant]\n{message}\n")

            tool_calls: list[ChatCompletionMessageToolCallParam] = []
            if message.tool_calls is not None:
                for tool_call in message.tool_calls:
                    function = tool_call.function
                    tool_calls.append(
                        ChatCompletionMessageToolCallParam(
                            id=tool_call.id,
                            function=Function(
                                name=function.name, arguments=function.arguments
                            ),
                            type="function",
                        )
                    )

            self.state.messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=message.content,
                    tool_calls=tool_calls,
                )
            )

            if message.tool_calls is not None:
                for tool_call in message.tool_calls:
                    print(f"[tool_call]\n{tool_call.to_json()}\n")
                    if tool_call.function.name == "query_stackoverflow":
                        args = json.loads(tool_call.function.arguments)
                        query_string = args["query_string"]
                        question_list = stackexchange.query_questions(query_string)[
                            "items"
                        ]
                        if len(question_list) == 0:
                            tool_call_message = ChatCompletionToolMessageParam(
                                content="No questions matched that query.",
                                role="tool",
                                tool_call_id=tool_call.id,
                            )
                        else:
                            diagnostics: list[str] = []

                            question_list = question_list[: self.params.max_questions]

                            answer_id_list = []
                            for question in question_list:
                                answer_id_list.append(question["accepted_answer_id"])

                            answer_list = stackexchange.get_answers_by_ids(
                                answer_id_list
                            )["items"]

                            for i, (question, answer) in enumerate(
                                zip(question_list, answer_list)
                            ):
                                print("question", question)
                                print("answer", answer)
                                diagnostics.append(
                                    f"""
# Question {i + 1}

## Problem Statement
{html.unescape(question['body_markdown'])}

## Solution
{html.unescape(answer['body_markdown'])}
""".strip()
                                )
                            content = f"""
The following are the top question that matched the query, along with their accepted answers. 

{"\n\n".join(diagnostics)}
""".strip()
                            print(f"[diagnostics]\n{"\n\n".join(diagnostics)}")
                            tool_call_message = ChatCompletionToolMessageParam(
                                content=content, role="tool", tool_call_id=tool_call.id
                            )
                    else:
                        raise AgentException(
                            f"unrecognized tool_call function name: {tool_call.function.name}"
                        )
                    self.state.messages.append(tool_call_message)


if __name__ == "__main__":
    agent = Agent(
        AgentParams(name="test", model="gpt-3.5-turbo", max_questions=1),
        state=AgentState(
            gas=1,
            prompt='please seach StackExchange for stuff relating to "type hinting syntax"',
        ),
    )
    agent.run()
