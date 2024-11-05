from dataclasses import dataclass


@dataclass
class A:
    x: int

    def add(self, y):
        self.x = self.x + y

    @classmethod
    def zero(cls):
        return A(0)


a0 = A.zero()
print(a0)
