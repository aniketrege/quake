from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class CleanupRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CleanupResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class InstanceRequest(_message.Message):
    __slots__ = ("name", "payload", "ts")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    TS_FIELD_NUMBER: _ClassVar[int]
    name: str
    payload: bytes
    ts: float
    def __init__(self, name: _Optional[str] = ..., payload: _Optional[bytes] = ..., ts: _Optional[float] = ...) -> None: ...

class InstanceResponse(_message.Message):
    __slots__ = ("uuid", "ts")
    UUID_FIELD_NUMBER: _ClassVar[int]
    TS_FIELD_NUMBER: _ClassVar[int]
    uuid: int
    ts: float
    def __init__(self, uuid: _Optional[int] = ..., ts: _Optional[float] = ...) -> None: ...

class CommandRequest(_message.Message):
    __slots__ = ("uuid", "method", "payload", "ts")
    UUID_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    TS_FIELD_NUMBER: _ClassVar[int]
    uuid: int
    method: str
    payload: bytes
    ts: float
    def __init__(self, uuid: _Optional[int] = ..., method: _Optional[str] = ..., payload: _Optional[bytes] = ..., ts: _Optional[float] = ...) -> None: ...

class CommandResponse(_message.Message):
    __slots__ = ("result", "direct", "ts")
    RESULT_FIELD_NUMBER: _ClassVar[int]
    DIRECT_FIELD_NUMBER: _ClassVar[int]
    TS_FIELD_NUMBER: _ClassVar[int]
    result: bytes
    direct: bool
    ts: float
    def __init__(self, result: _Optional[bytes] = ..., direct: bool = ..., ts: _Optional[float] = ...) -> None: ...

class ImportRequest(_message.Message):
    __slots__ = ("package", "as_name", "item", "ts")
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    AS_NAME_FIELD_NUMBER: _ClassVar[int]
    ITEM_FIELD_NUMBER: _ClassVar[int]
    TS_FIELD_NUMBER: _ClassVar[int]
    package: str
    as_name: str
    item: str
    ts: float
    def __init__(self, package: _Optional[str] = ..., as_name: _Optional[str] = ..., item: _Optional[str] = ..., ts: _Optional[float] = ...) -> None: ...

class ImportResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
