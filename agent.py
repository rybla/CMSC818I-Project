import os
from dataclasses import dataclass
import json
import html
from typing import Literal
from pybughive import ProjectIssue, checkout_project_at_issue
import stackexchange
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
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

# ==============================================================================


@dataclass
class AgentParams:
    name: str
    model: ChatModel
    max_questions: int
    project_issue: ProjectIssue
    gas: int


# ==============================================================================


# class AgentState:
#     gas: int
#     prompt: str
#     messages: list[ChatCompletionMessageParam]

#     def __init__(
#         self,
#         gas: int,
#         prompt: str,
#     ):
#         self.gas = gas
#         self.prompt = prompt
#         self.messages = [ChatCompletionUserMessageParam(role="user", content=prompt)]


# ==============================================================================


class AgentException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


# ==============================================================================


class AgentAction:
    tag: str


class ToolCallAgentAction(AgentAction):
    parameters: dict[str, tuple[type, dict[Literal["type", "description"], str]]]

    def to_chat_completion_tool_param() -> ChatCompletionToolParam:
        return ChatCompletionToolParam(
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


@dataclass
class QueryStackoverflow(ToolCallAgentAction):
    tag = "QueryStackoverflow"
    tool_call_id: str
    query_string: str


class NextMainMessage(AgentAction):
    tag = "NextMainMessage"


def from_tool_call_to_agent_tool_call_action(
    tool_call: ChatCompletionMessageToolCall,
) -> ToolCallAgentAction:
    args = json.loads(tool_call.function.arguments)
    query_string = args["query_string"]
    return QueryStackoverflow(tool_call_id=tool_call.id, query_string=query_string)


def from_tool_call_to_tool_call_param(
    tool_call: ChatCompletionMessageToolCall,
) -> ChatCompletionMessageToolCallParam:
    function = tool_call.function
    return ChatCompletionMessageToolCallParam(
        id=tool_call.id,
        function=Function(name=function.name, arguments=function.arguments),
        type="function",
    )


# ==============================================================================


@dataclass
class Conversation:
    model: ChatModel
    messages: list[ChatCompletionMessageParam]
    tools: list[ChatCompletionToolParam]

    def next_message(self) -> ChatCompletionMessage:
        completion = client.chat.completions.create(
            model=self.model, messages=self.messages, tools=self.tools
        )

        if len(completion.choices) == 0:
            raise AgentException("len(completion.choices) == 0")
        choice = completion.choices[0]

        return choice.message


class Agent:
    params: AgentParams
    transcript: list[AgentAction]
    gas: int
    main_convo: Conversation
    splitter_convo: Conversation

    def __init__(self, params: AgentParams) -> None:
        self.params = params
        self.transcript = []
        self.gas = self.params.gas
        prompt = ""  # TODO: prompt to start things off
        self.main_convo = Conversation(
            model=self.params.model,
            messages=[
                ChatCompletionUserMessageParam(role="user", content=prompt),
            ],
            tools=[
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
            ],
        )
        self.splitter_convo = Conversation(
            model=self.params.model, messages=[], tools=[]
        )

    def next_main_message(self, action: NextMainMessage) -> ChatCompletionMessage:
        self.transcript.append(action)
        return self.main_convo.next_message()

    def handle_tool_call_agent_action(self, action: AgentAction):
        self.transcript.append(action)
        if isinstance(action, QueryStackoverflow):
            # args = json.loads(tool_call.function.arguments)
            # query_string = args["query_string"]
            query_string = action.query_string
            question_list = stackexchange.query_questions(query_string)["items"]
            if len(question_list) == 0:
                tool_call_message = ChatCompletionToolMessageParam(
                    content="No questions matched that query.",
                    role="tool",
                    tool_call_id=action.tool_call_id,
                )
            else:
                diagnostics: list[str] = []

                question_list = question_list[: self.params.max_questions]

                answer_id_list = []
                for question in question_list:
                    answer_id_list.append(question["accepted_answer_id"])

                answer_list = stackexchange.get_answers_by_ids(answer_id_list)["items"]

                for i, (question, answer) in enumerate(zip(question_list, answer_list)):
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
                self.main_convo.messages.append(
                    ChatCompletionToolMessageParam(
                        content=content, role="tool", tool_call_id=action.tool_call_id
                    )
                )
        else:
            raise Exception(f"unhandled AgentAction with tag: {action.tag}")

    def run(self):
        checkout_project_at_issue(self.params.project_issue)

        while self.gas > 0:
            self.gas -= 1
            message = self.next_main_message(NextMainMessage())
            if message.tool_calls is not None:
                for tool_call in message.tool_calls:
                    self.handle_tool_call_agent_action(
                        from_tool_call_to_agent_tool_call_action(tool_call)
                    )
            # TODO: agent decides what to do next
            pass


if __name__ == "__main__":
    agent = Agent(
        AgentParams(
            name="test",
            model="gpt-3.5-turbo",
            max_questions=1,
            project_issue=ProjectIssue(
                username="psf", repository="black", issue_index=0
            ),
            gas=1,
        ),
    )
    agent.run()
    pass
