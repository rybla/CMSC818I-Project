from dataclasses import asdict, dataclass


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
print(asdict(b))
