def deco(cls: type):
    # print(cls.__doc__)
    # print(cls.__dict__)
    print(cls.__annotations__)
    # print(cls.__getattribute__("x"))
    print(cls.x)
    # print(cls)


@deco
class A:
    """
    The class called "A"
    """

    x: int
    "this is x"

    y: str
    z: bool

    pass
