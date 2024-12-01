import ast
import os
from abc import abstractmethod
from dataclasses import asdict, dataclass
import copy
import json
import html
import subprocess
from typing import Any, Literal, Self, TypeVar
import itertools

from openai import NOT_GIVEN
from pybughive import (
    DiffBlock,
    ProjectIssue,
    checkout_project_at_issue,
    project_issue_diff_blocks,
)
import pybughive
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

_DEBUG_MODE = False

random.seed()

Cls = TypeVar("Cls")


def asdict_super(x):
    return dict((k, getattr(x, k)) for k in dir(x) if not k.startswith("_"))


# ==============================================================================


@dataclass
class AgentParams:
    project_issue: ProjectIssue

    model: ChatModel
    gas: int

    using_stackoverflow: bool
    max_questions: int

    def ident(self):
        return " ".join(
            itertools.chain.from_iterable(
                [
                    [f"project_issue=({self.project_issue.ident()})"],
                    [f"model={self.model}"],
                    [f"gas={self.gas}"],
                    ["debug"] if _DEBUG_MODE else [],
                ]
            )
        )


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

    @abstractmethod
    def toJSON(self):
        pass


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


# ==============================================================================


class AgentAction:
    _tag: str

    @abstractmethod
    def toJSON(self):
        pass


@dataclass
class DoneAgentAction(AgentAction):
    _tag = "Done"
    remaining_gas: int

    def __str__(self):
        return f"Done(remaining_gas={self.remaining_gas})"

    def toJSON(self):
        return {"_tag": self._tag, "remaining_gas": self.remaining_gas}


@dataclass
class ToolUseAgentAction(AgentAction):
    _tag = "ToolUse"
    tool_call_id: str
    _tool_use: ToolUse

    def __str__(self):
        return f"ToolUseAgentAction({self._tool_use.template.name}, ...)"

    def toJSON(self):
        return {
            "_tag": self._tag,
            "tool_call_id": self.tool_call_id,
            "_tool_use": self._tool_use.toJSON(),
        }


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
                "description": "Array of potential bugs. Each item is a concise description of a single potential bug, and explicitly references the points of the source code that the bug manifests in. DO NOT include numberings or bullet points at the beginning of each bug description.",
                "items": {"type": "string"},
            }
        },
    )

    potential_bugs: list[str]

    def __str__(self):
        return f"EnumerateBugs(...)"

    def toJSON(self):
        return {"name": self.template.name, "potential_bugs": self.potential_bugs}


@dataclass
@toolclass
class QueryStackOverflow(ToolUse):
    template = ToolTemplate(
        name="query_stackoverflow",
        description="Query StackOverflow for related questions and answers.",
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

    def toJSON(self):
        return {"name": self.template.name, "query_string": self.query_string}


@dataclass
class NextMainMessage(AgentAction):
    _tag = "NextMainMessage"

    def __str__(self):
        return "NextMainMessage(...)"

    def toJSON(self):
        return {"_tag": self._tag}


@dataclass
class AppendMainMessage(AgentAction):
    _tag = "AppendMainMessage"
    msg: ChatCompletionMessageParam

    def __str__(self):
        return "AppendMainMessage(...)"

    def toJSON(self):
        return {
            "_tag": self._tag,
            "msg": {
                "role": self.msg["role"] if "role" in self.msg else self.msg.role,
                "content": (
                    self.msg["content"] if "content" in self.msg else self.msg.content
                ),
                "tool_call_id": (
                    (self.msg["tool_call_id"] if "tool_call_id" in self.msg else "None")
                    if isinstance(self.msg, dict)
                    else (
                        self.msg.tool_call_id
                        if "tool_call_id" in self.msg.__annotations__
                        else "None"
                    )
                ),
            },
        }


@dataclass
class EnumeratePotentialBugsMessage(AgentAction):
    _tag = "AppendEnumeratorMessage"
    msg: ChatCompletionUserMessageParam

    def __str__(self):
        return "EnumeratePotentialBugsMessage(...)"

    def toJSON(self):
        return {
            "_tag": self._tag,
            "msg": {
                "role": self.msg["role"] if "role" in self.msg else self.msg.role,
                "content": (
                    self.msg["content"] if "content" in self.msg else self.msg.content
                ),
            },
        }


@dataclass
class RecordBug(AgentAction):
    _tag = "RecordBug"
    bug_description: str

    def __str__(self):
        return f"RecordBug({self.bug_description})"

    def toJSON(self):
        return {"_tag": self._tag, "bug_description": self.bug_description}


@dataclass
class MainToolMessage(AgentAction):
    _tag = "MainToolMessage"
    msg: ChatCompletionToolMessageParam

    def __str__(self):
        return "MainToolMessage"

    def toJSON(self):
        return {
            "_tag": self._tag,
            "msg": {
                "role": self.msg["role"] if "role" in self.msg else self.msg.role,
                "content": (
                    self.msg["content"] if "content" in self.msg else self.msg.content
                ),
                "tool_call_id": (
                    self.msg["tool_call_id"]
                    if "tool_call_id" in self.msg
                    else self.msg.tool_call_id
                ),
            },
        }


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
        # print(f"BEGIN messages")
        # for msg in self.messages:
        #     print(msg)
        # print(f"END messages")

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
                ChatCompletionAssistantMessageParam(
                    content=choice.message.content,
                    refusal=choice.message.refusal,
                    role=choice.message.role,
                    tool_calls=choice.message.tool_calls,
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
        self.main_convo = Conversation(
            model=self.params.model,
            messages=[
                ChatCompletionSystemMessageParam(
                    role="system",
                    content="""
You are an expert assistant for identifying bugs in Python programs. You have access to a few tools to help you gather extra contextual information about programming topics or questions about the program at hand. You are very conscious of minimizing your false positive rate in identifying bugs.
""".strip(),
                ),
            ],
            tools=[
                QueryStackOverflow.template.to_chat_completion_tool_param(),
            ],
            cumulative=True,
        )
        self.enumerator_convo = Conversation(
            model=self.params.model,
            messages=[
                ChatCompletionSystemMessageParam(
                    role="system",
                    content="""
You are an expert assistant for enumerating potential bugs in Python programs. You are ALWAYS careful to ONLY identify bugs that you are relatively sure about, in order to minimize your false positive rate.
""".strip(),
                )
            ],
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
                    lambda fn: fn.startswith(self.params.ident()),
                    os.listdir(f"{pybughive.repository_path}transcripts/"),
                ),
            )
        )
        transcript_filename = f"transcripts/{self.params.ident()} #{str(n)}.json"
        debug_log(f"writing transcript to file: {transcript_filename}")
        # for action in self.transcript:
        #     print(asdict_super(action))
        self.transcript.append(DoneAgentAction(remaining_gas=self.gas))
        json.dump(
            [action.toJSON() for action in self.transcript],
            open(f"{pybughive.repository_path}{transcript_filename}", "w+"),
        )

    def enumerate_potential_bugs(
        self, action: EnumeratePotentialBugsMessage
    ) -> list[str]:
        self.transcribe_action(action)
        self.enumerator_convo.messages.append(action.msg)
        message = self.enumerator_convo.next_message()
        for tool_call in message.tool_calls if message.tool_calls is not None else []:
            tooluse_action = ToolUseAgentAction(
                tool_call_id=tool_call.id,
                _tool_use=from_ChatCompletionMessageToolCall_to_ToolUse(tool_call),
            )
            self.transcribe_action(tooluse_action)
            tool_use = tooluse_action._tool_use
            if isinstance(tool_use, EnumerateBugs):
                return tool_use.potential_bugs
            else:
                raise Exception(
                    f"enumerator_convo should use only tool EnumerateBugs: {tooluse_action._tool_use}"
                )

    def next_main_message(self, action: NextMainMessage) -> ChatCompletionMessage:
        self.transcribe_action(action)
        message = self.main_convo.next_message()
        self.transcribe_action(AppendMainMessage(message))
        return message

    def append_main_message(self, action: AppendMainMessage):
        self.transcribe_action(action)
        self.main_convo.messages.append(action.msg)

    def append_main_tool_message(self, msg: ChatCompletionToolMessageParam):
        self.transcribe_action(MainToolMessage(msg))
        self.main_convo.messages.append(msg)

    def record_bug(self, bug_description: str):
        self.transcribe_action(RecordBug(bug_description))

    def handle_tool_call_agent_action(self, action: ToolUseAgentAction):
        self.transcribe_action(action)
        if isinstance(action._tool_use, QueryStackOverflow):
            query_string = action._tool_use.query_string
            question_list = stackexchange.query_questions(query_string)["items"]
            if len(question_list) == 0:
                tool_call_message = ChatCompletionToolMessageParam(
                    role="tool",
                    content="No questions matched that query.",
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
                    role="tool",
                    content=content,
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

        potential_bugs = self.enumerate_potential_bugs(
            EnumeratePotentialBugsMessage(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=(
                        f"""
Consider the following Python code from various files in a project.
{diff_block_pp_s}
These files exist in a larger project that you don't have access to, but try to infer what that context would probably be in order to make sense of this code that you do have access to. You should assume that things defined in the snippets included above are used elsewhere in the project. You should also assume that things referenced but not defined in the snippets are imported from dependencies or other parts of the project.
Now, use the "enumerate_potential_bugs" tool to enumerate the most likely potential bugs that could be present in the sections of code you've been presented with above.
Note that there may be NO bugs, or at most a few (around 1-3) bugs.
                    """.strip()
                        if not _DEBUG_MODE
                        else f"""
Use the "enumerate_potential_bugs" tool to give a give a single example bug.
Make your description of the bug as concise as possible.
""".strip()
                    ),
                )
            )
        )
        potential_bugs_descs = "\n\n".join(
            [
                f"Potential Bug #{i + 1}: {potential_bug.strip()}"
                for i, potential_bug in enumerate(potential_bugs)
            ]
        )
        self.append_main_message(
            AppendMainMessage(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"""
The following potential bugs were identified:

{potential_bugs_descs}
""".strip(),
                )
            )
        )

        if self.params.using_stackoverflow:
            for i, potential_bug in enumerate(potential_bugs):
                self.append_main_message(
                    AppendMainMessage(
                        ChatCompletionUserMessageParam(
                            role="user",
                            content=f"""
Now, for Potential Bug #{i + 1}, make a relevant query to StackOverflow, using the `query_stackoverflow` tool, in order to get extra information.
""".strip(),
                        )
                    )
                )

                search_for_help = self.next_main_message(NextMainMessage())

                for tool_call in (
                    search_for_help.tool_calls
                    if search_for_help.tool_calls is not None
                    else []
                ):
                    self.handle_tool_call_agent_action(
                        ToolUseAgentAction(
                            tool_call_id=tool_call.id,
                            _tool_use=from_ChatCompletionMessageToolCall_to_ToolUse(
                                tool_call
                            ),
                        )
                    )

                self.append_main_message(
                    AppendMainMessage(
                        ChatCompletionUserMessageParam(
                            role="user",
                            content=f"""
    Given this new additional context, now respond with one of the following:
    - if you now think the potential bug is probably NOT a bug, then respond with: "NO BUG"
    - if you STILL think the potential bug probably IS a bug, then respond with "YES BUG" followed up an updated description of the potential bug
    
    You MUST repond with exactly one of those options.
    """.strip(),
                        )
                    )
                )

                judgment: ChatCompletionMessage = self.next_main_message(
                    NextMainMessage()
                )
                if "YES BUG" in judgment.content.upper():
                    self.record_bug(judgment.content)
                elif "NO BUG" in judgment.content.upper():
                    pass
                else:
                    debug_log(f"failed to judge potential bug: {potential_bug}")

        self.save_transcript()


project_issues = [
    [
        ProjectIssue(username="psf", repository="black", issue_index=i)
        for i in range(38)
    ],
    ProjectIssue(username="cookiecutter", repository="cookiecutter", issue_index=0),
]


if __name__ == "__main__":
    agent = Agent(
        AgentParams(
            project_issue=project_issues[1],
            model="gpt-3.5-turbo",
            # model="gpt-4-turbo",
            gas=100,
            max_questions=1,
            using_stackoverflow=True,
        ),
    )
    agent.run()
    pass
