"""src/genomic_variant_classifier/inert_pickle.py

Author: Monzia Moodie

An INERT reader of joblib/pickle files: it recovers plain data (numbers, strings, containers and plain-dtype NumPy
arrays) from a historical artifact WITHOUT importing, constructing or running any class the file names.

Why it exists (package 1, set 1b, gate G2.3). Run 15's out-of-fold identity must be derived from the artifact itself
(`ensemble.joblib`, sha256 71f87e32...): the orchestrator stores `oof_fit_indices_`, the fit-subset positions of the
out-of-fold rows. The only admission route for models (model_admission.admit_artifact) denies every artifact (C10), and
it should: deserializing a classifier runs its classes' code. This module is NOT a model loader and can never yield a
usable model. Every global the stream names becomes an `InertObject` recording (module, name, arguments, state); no
module is imported, no `__reduce__` callable is invoked, no `__setstate__` of a real class runs and no persistent id is
resolved. The ONLY values it materialises are:

  * the pickle machine's own primitives (None, bool, int, float, str, bytes, tuple, list, dict, set, frozenset);
  * NumPy arrays stored by joblib's NumpyArrayWrapper whose dtype is a plain boolean, integer, unsigned or floating
    scalar type (no fields, no subarray), read from the raw bytes that follow the wrapper exactly as joblib 1.x lays
    them out (one padding-length byte and 0xff padding when the wrapper records an alignment).

An object-dtype array (which joblib writes as a nested pickle stream) is read by a second inert machine and returned
as its inert record; a NumPy array pickled through `numpy.core.multiarray._reconstruct` stays an InertObject; a
structured dtype is refused with InertPickleError. Nothing is coerced. The reader works on the pure-Python
`pickle._Unpickler`, so every opcode goes through code that this module controls.

What it is evidence of: the bytes of the file, interpreted by the pickle opcode grammar. It is not evidence that the
recorded objects were valid, nor that the run used them.
"""
from __future__ import annotations

import io
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["InertObject", "InertPickleError", "read_inert", "read_inert_bytes"]

_WRAPPER = ("joblib.numpy_pickle", "NumpyArrayWrapper")
_NDARRAY = ("numpy", "ndarray")
_DTYPE_GLOBALS = frozenset({("numpy", "dtype"), ("numpy.core.multiarray", "dtype"),
                            ("numpy._core.multiarray", "dtype")})
_PLAIN_KINDS = frozenset("biuf")


class InertPickleError(ValueError):
    """The stream needs something an inert reader refuses to provide (never repaired or guessed)."""


class InertObject:
    """A record of what the pickle stream asked for. Never a real instance of the named class."""

    __slots__ = ("module", "name", "args", "state", "items", "setitems")

    def __init__(self, module: str, name: str, args: tuple = ()) -> None:
        self.module = module
        self.name = name
        self.args = args
        self.state: Any = None
        self.items: list = []           # APPEND/APPENDS onto an inert object
        self.setitems: list = []        # SETITEM/SETITEMS onto an inert object

    @property
    def qualname(self) -> str:
        return f"{self.module}.{self.name}"

    def append(self, value: Any) -> None:
        self.items.append(value)

    def extend(self, values: Any) -> None:
        self.items.extend(values)

    def add(self, value: Any) -> None:            # ADDITEMS onto an inert object
        self.items.append(value)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.setitems.append((key, value))

    def __repr__(self) -> str:
        return f"InertObject({self.qualname})"


class _InertGlobal:
    """What find_class returns: a callable recorder. Calling it (REDUCE) or `__new__`-ing it (NEWOBJ) records the
    arguments in an InertObject; nothing named by the stream is ever imported."""

    __slots__ = ("module", "name")

    def __init__(self, module: str, name: str) -> None:
        self.module = module
        self.name = name

    def __call__(self, *args: Any) -> InertObject:
        return InertObject(self.module, self.name, tuple(args))

    def __repr__(self) -> str:
        return f"InertGlobal({self.module}.{self.name})"


class _StreamView:
    """The parent machine's byte stream, offered to a nested inert machine (joblib object arrays)."""

    def __init__(self, parent: "_InertUnpickler") -> None:
        self._parent = parent

    def read(self, n: int = -1) -> bytes:
        return self._parent.read(n)

    def readline(self) -> bytes:
        return self._parent.readline()

    def readinto(self, buf: Any) -> int:
        data = self._parent.read(len(buf))
        buf[:len(data)] = data
        return len(data)


def _plain_dtype(node: Any) -> np.dtype:
    """Rebuild a plain scalar dtype from its inert REDUCE record, or refuse."""
    if type(node) is not InertObject or (node.module, node.name) not in _DTYPE_GLOBALS:
        raise InertPickleError(f"array dtype is not a recorded numpy.dtype: {node!r}")
    if not node.args or type(node.args[0]) is not str:
        raise InertPickleError("dtype record without a type code")
    base = np.dtype(node.args[0])
    if base.kind not in _PLAIN_KINDS or base.fields is not None or base.subdtype is not None:
        raise InertPickleError(f"only plain bool/int/uint/float dtypes are materialised, not {base!r}")
    state = node.state
    if type(state) is not tuple or len(state) < 2:
        raise InertPickleError(f"unexpected dtype state {state!r}")
    order = state[1]
    if order not in ("<", ">", "|", "="):
        raise InertPickleError(f"unexpected dtype byte order {order!r}")
    if len(state) >= 5 and (state[2] is not None or state[3] is not None or state[4] is not None):
        raise InertPickleError("subarray, field names or fields present: not a plain dtype")
    return base.newbyteorder(order) if order in ("<", ">") else base


class _InertUnpickler(pickle._Unpickler):  # noqa: SLF001 -- the pure-Python machine, so every opcode is ours
    dispatch = dict(pickle._Unpickler.dispatch)  # noqa: SLF001

    def find_class(self, module: str, name: str) -> _InertGlobal:  # never imports
        if type(module) is not str or type(name) is not str:
            raise InertPickleError("non-string global reference")
        return _InertGlobal(module, name)

    def persistent_load(self, pid: Any) -> Any:
        raise InertPickleError(f"persistent id {pid!r} refused: an inert reader resolves no external reference")

    def load_newobj(self) -> None:
        args = self.stack.pop()
        cls = self.stack.pop()
        self._push_new(cls, args)

    def load_newobj_ex(self) -> None:
        kwargs = self.stack.pop()
        args = self.stack.pop()
        cls = self.stack.pop()
        if kwargs:
            raise InertPickleError("NEWOBJ_EX keyword arguments are not supported by the inert reader")
        self._push_new(cls, args)

    def _push_new(self, cls: Any, args: Any) -> None:
        if type(cls) is not _InertGlobal:
            raise InertPickleError(f"NEWOBJ on a non-global {cls!r}")
        self.append(InertObject(cls.module, cls.name, tuple(args)))

    def load_reduce(self) -> None:
        args = self.stack.pop()
        func = self.stack[-1]
        if type(func) is not _InertGlobal:
            raise InertPickleError(f"REDUCE on a non-global callable {func!r}")
        self.stack[-1] = func(*args)

    def load_build(self) -> None:
        state = self.stack.pop()
        inst = self.stack[-1]
        if type(inst) is not InertObject:
            raise InertPickleError(f"BUILD on a materialised value of type {type(inst).__name__}")
        inst.state = state
        if (inst.module, inst.name) == _WRAPPER:
            self.stack[-1] = self._read_wrapped_array(inst)

    def _read_wrapped_array(self, wrapper: InertObject) -> Any:
        state = wrapper.state
        if type(state) is not dict:
            raise InertPickleError("NumpyArrayWrapper state is not a dictionary")
        subclass = state.get("subclass")
        if type(subclass) is not _InertGlobal or (subclass.module, subclass.name) != _NDARRAY:
            raise InertPickleError(f"only numpy.ndarray is materialised, not {subclass!r}")
        shape = state.get("shape")
        if type(shape) is not tuple or any(type(d) is not int or d < 0 for d in shape):
            raise InertPickleError(f"invalid recorded shape {shape!r}")
        order = state.get("order")
        if order not in ("C", "F"):
            raise InertPickleError(f"invalid recorded order {order!r}")
        dtype_node = state.get("dtype")
        if type(dtype_node) is InertObject and dtype_node.args and dtype_node.args[0] == "O8":
            # joblib writes an object array as a NESTED protocol-5 pickle stream; it is read by a second inert
            # machine over the same bytes and returned as its inert record, never as an array of real objects.
            return _InertUnpickler(_StreamView(self)).load()
        dtype = _plain_dtype(dtype_node)
        alignment = state.get("numpy_array_alignment_bytes")
        if alignment is not None:
            pad = self.read(1)
            if len(pad) != 1:
                raise InertPickleError("truncated array padding byte")
            n_pad = pad[0]
            if n_pad and self.read(n_pad) != b"\xff" * n_pad:
                raise InertPickleError("array padding is not the joblib 0xff fill")
        count = 1
        for d in shape:
            count *= d
        nbytes = count * dtype.itemsize
        data = self.read(nbytes)
        if len(data) != nbytes:
            raise InertPickleError(f"truncated array data: expected {nbytes} bytes, got {len(data)}")
        array = np.frombuffer(data, dtype=dtype, count=count).copy()
        if order == "F":
            array = array.reshape(shape[::-1]).transpose()
        else:
            array = array.reshape(shape)
        array.flags.writeable = False
        return array

    dispatch[pickle.NEWOBJ[0]] = load_newobj
    dispatch[pickle.NEWOBJ_EX[0]] = load_newobj_ex
    dispatch[pickle.REDUCE[0]] = load_reduce
    dispatch[pickle.BUILD[0]] = load_build


def read_inert_bytes(payload: bytes) -> Any:
    """Inertly interpret one pickle stream; the whole payload must be consumed (no trailing bytes)."""
    if type(payload) is not bytes:
        raise InertPickleError("bytes required")
    stream = io.BytesIO(payload)
    value = _InertUnpickler(stream).load()
    if stream.tell() != len(payload):
        raise InertPickleError(f"{len(payload) - stream.tell()} trailing bytes after the pickle stream")
    return value


def read_inert(path: str | Path) -> Any:
    """Inertly interpret an UNCOMPRESSED joblib/pickle file (a compressed file is refused, not decompressed)."""
    data = Path(path).read_bytes()
    if not data.startswith(b"\x80"):
        raise InertPickleError("not an uncompressed protocol-2+ pickle stream (compressed joblib files are refused)")
    return read_inert_bytes(data)
