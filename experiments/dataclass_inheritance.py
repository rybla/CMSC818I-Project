from dataclasses import asdict, dataclass
from typing import Any
import typing
import inspect

from collections import ChainMap


def all_annotations(cls) -> ChainMap:
    """Returns a dictionary-like ChainMap that includes annotations for all
    attributes defined in cls or inherited from superclasses."""
    return ChainMap(
        *(c.__annotations__ for c in cls.__mro__ if "__annotations__" in c.__dict__)
    )


@dataclass
class A:
    y: int
    x: int = 2


@dataclass
class B:
    z: int
    y = 1
    x: int = 2


b = B(10)
# print(asdict(b))


class Animal:
    is_animal = True
    name: str


@dataclass
class Cat(Animal):
    name = "Cat"
    meow: bool


def asdict_all(x):
    return dict((k, getattr(x, k)) for k in dir(x) if not k.startswith("__"))


cat = Cat(meow=True)
# print(asdict(cat))
# print(cat.__annotations__)
# print(get_type_hints(Cat, include_extras=True))
# print([k for k in dir(cat.__class__) if not k.startswith("__")])
# print([k for k in dir(cat) if not k.startswith("__")])
print(asdict_all(cat))
# print(all_annotations(Cat))
# print(inspect.get_annotations(Cat))
# print(Cat.__dict__.keys())
