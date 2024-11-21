import ast
import os
from abc import abstractmethod
from dataclasses import asdict, dataclass
import copy
import json
import html
import subprocess
from typing import Any, Literal, Self, TypeVar

from openai import NOT_GIVEN
from pybughive import (
    DiffBlock,
    ProjectIssue,
    checkout_project_at_issue,
    project_issue_diff_blocks,
)
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
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat_model import ChatModel
from ai import client
from utilities import debug_log, find_innermost_scope
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


class AgentException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


# ==============================================================================


@dataclass
class ToolTemplate:
    name: str
    description: str
    parameters: dict[str, Any]

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
                    "strict": True,
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

    def __str__(self):
        return f"ToolUseAgentAction({self._tool_use.template.name}, ...)"


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
class EnumerateBugs(ToolUse):
    template = ToolTemplate(
        name="enumerate_potential_bugs",
        description="",
        parameters={
            "potential_bugs": {
                "type": "array",
                "description": "TODO",
                "items": {"type": "string"},
            }
        },
    )

    potential_bugs: list[str]

    def __str__(self):
        return f"EnumerateBugs(...)"


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

    def __str__(self):
        return "QueryStackOverflow(...)"


@dataclass
class NextMainMessage(AgentAction):
    _tag = "NextMainMessage"

    def __str__(self):
        return "NextMainMessage(...)"


@dataclass
class AppendMainMessage(AgentAction):
    _tag = "AppendMainMessage"
    msg: ChatCompletionUserMessageParam

    def __str__(self):
        return "AppendMainMessage(...)"


@dataclass
class EnumeratePotentialBugsMessage(AgentAction):
    _tag = "AppendEnumeratorMessage"
    msg: ChatCompletionUserMessageParam

    def __str__(self):
        return "EnumeratePotentialBugsMessage(...)"


@dataclass
class MainToolMessage(AgentAction):
    _tag = "MainToolMessage"
    msg: ChatCompletionToolMessageParam

    def __str__(self):
        return "MainToolMessage"


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
    cumulative: bool

    def next_message(self) -> ChatCompletionMessage:
        completion = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=NOT_GIVEN if len(self.tools) == 0 else self.tools,
        )

        if len(completion.choices) == 0:
            raise AgentException("len(completion.choices) == 0")
        choice = completion.choices[0]

        if self.cumulative:
            self.messages.append(
                ChatCompletionMessageParam(
                    content=choice.message.content, role=choice.message.role
                )
            )

        return choice.message


class Agent:
    params: AgentParams
    transcript: list[AgentAction]
    gas: int
    main_convo: Conversation
    enumerator_convo: Conversation

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
            cumulative=True,
        )
        self.enumerator_convo = Conversation(
            model=self.params.model,
            messages=[ChatCompletionSystemMessageParam(role="system", content=prompt)],
            tools=[EnumerateBugs.template.to_chat_completion_tool_param()],
            cumulative=False,
        )

    def transcribe_action(self, action: AgentAction):
        debug_log(f"[action] {str(action)}")
        self.transcript.append(action)
        self.gas -= 1
        if self.gas == 0:
            self.save_transcript()
            raise Exception("ran out of gas")

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

    def enumerate_potential_bugs(self, action: EnumeratePotentialBugsMessage):
        self.transcribe_action(action)
        self.enumerator_convo.messages.append(action.msg)
        message = self.enumerator_convo.next_message()
        for tool_call in message.tool_calls if message.tool_calls is not None else []:
            tooluse_action = ToolUseAgentAction(
                tool_call_id=tool_call.id,
                _tool_use=from_ChatCompletionMessageToolCall_to_ToolUse(tool_call),
            )
            self.transcribe_action(tooluse_action)
            if isinstance(tooluse_action._tool_use, EnumerateBugs):
                raise Exception("TODO")
            else:
                raise Exception(
                    f"enumerator_convo should use only tool EnumerateBugs: {tooluse_action._tool_use}"
                )

    def next_main_message(self, action: NextMainMessage) -> ChatCompletionMessage:
        self.transcribe_action(action)
        return self.main_convo.next_message()

    def append_main_message(self, action: AppendMainMessage):
        self.transcribe_action(action)
        self.main_convo.messages.append(action.msg)

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
                    # print("question", question)
                    # print("answer", answer)
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
                # debug_log(f"[diagnostics]\n{"\n\n".join(diagnostics)}")

                tool_call_message = ChatCompletionToolMessageParam(
                    content=content,
                    role="tool",
                    tool_call_id=action.tool_call_id,
                )

            self.append_main_tool_message(tool_call_message)

    def run(self):
        global innermost
        checkout_project_at_issue(self.params.project_issue)
        diff_blocks = project_issue_diff_blocks(self.params.project_issue)
        diff_blocks_pp_buggy: list[tuple[DiffBlock, str]] = []
        for diff_block in diff_blocks:
            source_line_range, _ = diff_block.line_ranges()
            source_file = diff_block.patched_file.source_file.strip("a/")

            # only care about changes in python files
            if not source_file.endswith(".py"):
                continue

            debug_log(f"extracting diff block from {source_file}")
            with open(source_file, "r") as f:
                source = f.read()
                tree = ast.parse(source)

            innermost = find_innermost_scope(tree, source_line_range)
            if innermost is None:
                raise Exception("failed to find innermost class or function scope")
            else:
                innermost_pp = ast.unparse(innermost)
                diff_blocks_pp_buggy.append((diff_block, innermost_pp))

        diff_block_pp_s = "\n\n".join(
            [
                f"""
file: {diff_block.patched_file.source_file.strip("a/")}
```
{innermost_pp}
```
                """.strip()
                for (diff_block, innermost_pp) in diff_blocks_pp_buggy
            ]
        )

        self.enumerate_potential_bugs(
            EnumeratePotentialBugsMessage(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"""
Consider the following Python code from various files in a project.

{diff_block_pp_s}

These files exist in a larger project that you don't have access to, but try to infer what that context would probably be in order to make sense of this code that you do have access to.

Now, use the "enumerate_potential_bugs" tool to enumerate the most likely potential bugs that could be present in the sections of code you've been presented with above.
Note that there may be NO bugs, or at most A FEW (around 1-3) bugs.
""".strip(),
                )
            )
        )
        message = self.next_main_message(NextMainMessage())
        debug_log(f"message.tool_calls: {str(message.tool_calls)}")
        for tool_call in message.tool_calls if message.tool_calls is not None else []:
            self.handle_tool_call_agent_action(
                ToolUseAgentAction(
                    tool_call_id=tool_call.id,
                    _tool_use=from_ChatCompletionMessageToolCall_to_ToolUse(tool_call),
                )
            )
        # TODO: agent decides what to do next

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
            gas=10,
        ),
    )
    agent.run()
    pass
