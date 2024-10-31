from typing import Any
import requests
import urllib.parse
import json


class StackexchangeException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def get_json(response: requests.Response) -> Any:
    if response.ok:
        data = response.json()
        return data
    else:
        raise StackexchangeException(f"status code: {response.status_code}")


def query_questions(query: str) -> Any:
    query = urllib.parse.quote(query)
    url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q={query}&accepted=True&tagged=python&site=stackoverflow&filter=!CdjJRv-anzUSbmwG0Uj7B"
    response = requests.get(url)
    return get_json(response)


def get_answers_by_ids(ids: list[int]):
    ids_str = ";".join([str(id) for id in ids])
    url = f"https://api.stackexchange.com/2.3/answers/{";".join(ids_str)}?order=desc&sort=activity&site=stackoverflow&filter=!.f9fwfet97zlpL2SIgQI"
    response = requests.get(url)
    return get_json(response)


if __name__ == "__main__":
    # response = query_questions("hello world")
    # print(response)
    # response = get_answers_by_ids([419185])
    # print(response)
    pass
