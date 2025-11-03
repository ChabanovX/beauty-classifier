from typing import TypeVar, Generic, Callable

T = TypeVar("T")


class classproperty(Generic[T]):
    def __init__(self, f: Callable[[type], T]):
        self.f = f

    def __get__(self, obj: None, owner: type) -> T:
        return self.f(owner)
