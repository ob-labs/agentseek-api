from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict


class User(BaseModel):
    model_config = ConfigDict(extra="allow")

    identity: str
    is_authenticated: bool = True
    display_name: str = ""
    permissions: Sequence[str] = ()

    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __iter__(self):
        yield from self.model_fields_set | set(self.model_fields.keys())
        if self.__pydantic_extra__:
            yield from self.__pydantic_extra__
