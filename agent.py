import os
from abc import abstractmethod
from dataclasses import asdict, dataclass
import copy
import json
import html
from typing import Any, Literal, Self, TypeVar

from openai import NOT_GIVEN
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
from utilities import debug_log
import random

random.seed()

Cls = TypeVar("Cls")


def asdict_super(x):
    return dict((k, getattr(x, k)) for k in dir(x) if not k.startswith("_"))


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


@dataclass
class ToolTemplate:
    name: str
    description: str
    parameters: dict[str, dict[Literal["type", "description"], str]]

    def to_chat_completion_tool_param(self) -> ChatCompletionToolParam:
        return ChatCompletionToolParam(
            type="function",
            function=FunctionDefinition(
                name=self.name,
                description=self.description,
                parameters={
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys()),
                    "additionalProperties": False,
                },
                strict=True,
            ),
        )


@dataclass
class ToolUse:
    template: ToolTemplate


all_tool_use_classes: list[type[ToolUse]] = []


def toolclass(cls):
    all_tool_use_classes.append(cls)
    template: ToolTemplate = cls.template
    template_parameter_keys = set(template.parameters.keys())
    class_annotation_keys = set(cls.__annotations__.keys())

    if not template_parameter_keys == class_annotation_keys:
        raise Exception(
            f"""
Mismatch in ToolMeta template parameters and class annotations:
  - template parameters : {template_parameter_keys}
  - class annotations   : {class_annotation_keys}
""".strip()
        )
    return cls


query_stackoverflow = ToolTemplate(
    name="query_stackoverflow",
    description="Query StackOver for related questions and answers.",
    parameters={
        "query_string": {
            "type": "string",
            "description": "The query string to send to StackOverflow. This should just be a concise collection of the most relevant keywords.",
        }
    },
)


# ==============================================================================


class AgentAction:
    _tag: str


@dataclass
class ToolUseAgentAction(AgentAction):
    tag = "ToolUse"
    tool_call_id: str
    _tool_use: ToolUse


def from_ChatCompletionMessageToolCall_to_ToolUse(
    tool_call: ChatCompletionMessageToolCall,
) -> ToolUse:
    for tool_use in all_tool_use_classes:
        function = tool_call.function
        if tool_use.template.name == function.name:
            return tool_use(tool_use.template, **json.loads(function.arguments))
    raise Exception(f'unrecognized tool call with function name: "{function.name}"')


@dataclass
@toolclass
class QueryStackOverflow(ToolUse):
    template = ToolTemplate(
        name="query_stackoverflow",
        description="Query StackOver for related questions and answers.",
        parameters={
            "query_string": {
                "type": "string",
                "description": "The query string to send to StackOverflow. This should just be a concise collection of the most relevant keywords.",
            }
        },
    )

    query_string: str


@dataclass
class NextMainMessage(AgentAction):
    _tag = "NextMainMessage"


@dataclass
class MainToolMessage(AgentAction):
    _tag = "MainToolMessage"
    msg: ChatCompletionToolMessageParam


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
            model=self.model,
            messages=self.messages,
            tools=NOT_GIVEN if len(self.tools) == 0 else self.tools,
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
        # TODO: actual prompt
        prompt = """
Search StackOverflow for how to use type hints in Python.
""".strip()
        self.main_convo = Conversation(
            model=self.params.model,
            messages=[
                ChatCompletionUserMessageParam(role="user", content=prompt),
            ],
            tools=[
                QueryStackOverflow.template.to_chat_completion_tool_param(),
            ],
        )
        self.splitter_convo = Conversation(
            model=self.params.model, messages=[], tools=[]
        )

    def transcribe_action(self, action: AgentAction):
        self.transcript.append(action)

    def save_transcript(self):
        n = len(
            list(
                filter(
                    lambda fn: fn.startswith(self.params.name),
                    os.listdir("../../transcripts/"),
                ),
            )
        )
        transcript_filename = f"transcripts/{self.params.name}_{str(n)}.json"
        debug_log(f"writing transcript to file: {transcript_filename}")
        for action in self.transcript:
            print(asdict_super(action))
        json.dump(
            [asdict_super(action) for action in self.transcript],
            open(f"../../{transcript_filename}", "w+"),
        )

    def next_main_message(self, action: NextMainMessage) -> ChatCompletionMessage:
        self.transcribe_action(action)
        return self.main_convo.next_message()

    def append_main_tool_message(self, msg: ChatCompletionToolMessageParam):
        self.transcribe_action(MainToolMessage(msg))
        self.main_convo.messages.append(msg)

    def handle_tool_call_agent_action(self, action: ToolUseAgentAction):
        self.transcribe_action(action)
        if isinstance(action._tool_use, QueryStackOverflow):
            query_string = action._tool_use.query_string
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

                self.append_main_tool_message(
                    ChatCompletionToolMessageParam(
                        content=content,
                        role="tool",
                        tool_call_id=action.tool_call_id,
                    )
                )

    def run(self):
        checkout_project_at_issue(self.params.project_issue)

        while self.gas > 0:
            self.gas -= 1
            message = self.next_main_message(NextMainMessage())
            debug_log(f"message.tool_calls: {str(message.tool_calls)}")
            # if message.tool_calls is not None:
            for tool_call in message.tool_calls or []:
                self.handle_tool_call_agent_action(
                    ToolUseAgentAction(
                        tool_call_id=tool_call.id,
                        _tool_use=from_ChatCompletionMessageToolCall_to_ToolUse(
                            tool_call
                        ),
                    )
                )
            # TODO: agent decides what to do next
            pass

        self.save_transcript()


if __name__ == "__main__":
    agent = Agent(
        AgentParams(
            name="test",
            model="gpt-3.5-turbo",
            # model="gpt-4-turbo",
            max_questions=1,
            project_issue=ProjectIssue(
                username="psf", repository="black", issue_index=0
            ),
            gas=1,
        ),
    )
    agent.run()
    pass
