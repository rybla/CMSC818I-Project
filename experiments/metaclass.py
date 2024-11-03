from dataclasses import asdict, dataclass


class Meta(type):
    def __new__(cls, name, bases, dict):
        # del dict["tmp"]
        print(bases)
        return super().__new__(cls, name, bases, dict)


@dataclass
class A(object, metaclass=Meta):
    tmp = "hello world"
    x: int
    y: str


a = A(1, "hello")
print(asdict(a))
