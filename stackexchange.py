from typing import Any
import requests
import urllib.parse
import json
import html


class StackexchangeException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def get_json(response: requests.Response) -> Any:
    if response.ok:
        data = response.json()
        return data
    else:
        raise StackexchangeException(f"status code: {response.status_code}")


def query_questions(query: str) -> dict:
    query = urllib.parse.quote(query)
    url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q={query}&accepted=True&tagged=python&site=stackoverflow&filter=!CdjJRv-anzUSbmwG0Uj7B"
    response = requests.get(url)
    return get_json(response)


def get_answers_by_ids(ids: list[int]):
    ids_str = ";".join([str(id) for id in ids])
    url = f"https://api.stackexchange.com/2.3/answers/{ids_str}?order=desc&sort=activity&site=stackoverflow&filter=!.f9fwfet97zlpL2SIgQI"
    response = requests.get(url)
    return get_json(response)


if __name__ == "__main__":
    query_string = "python function syntax"
    max_questions = 2
    question_list = query_questions(query_string)["items"]
    if len(question_list) == 0:
        print("no related questions")
    else:
        diagnostics: list[str] = []

        question_list = question_list[:max_questions]
        print(question_list)

        answer_id_list = []
        for question in question_list:
            answer_id_list.append(question["accepted_answer_id"])

        print(f"answer_id_list: {answer_id_list}")
        answer_list = get_answers_by_ids(answer_id_list)["items"]
        print(f"answer_list: {answer_list}")

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
