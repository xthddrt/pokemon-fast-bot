"""A drop-in `numpy` proxy that COUNTS calls and output elements.

Used to turn "0.083 ms/state in numpy" into a machine-independent work figure:
how many array operations, over how many elements, the dynamic half actually
performs. That is what converts to a Rust estimate -- the call count is the
dispatch overhead Rust deletes outright, the element count is the arithmetic
Rust still has to do (fused, in registers, instead of through temporaries).

Correctness is asserted by the caller: the counted run must be bit-identical to
the uncounted one.
"""
import numpy as _np

STATS = {"calls": 0, "elems": 0}
BY_KIND = {}
TRACE = []          # per-call output snapshots, when RECORD is on
RECORD = [False]


def reset(record=False):
    STATS["calls"] = 0
    STATS["elems"] = 0
    BY_KIND.clear()
    TRACE.clear()
    RECORD[0] = record


def _note(name, r):
    STATS["calls"] += 1
    n = 0
    if isinstance(r, _np.ndarray):
        n = r.size
    elif isinstance(r, tuple):
        n = sum(x.size for x in r if isinstance(x, _np.ndarray))
    STATS["elems"] += n
    c, e = BY_KIND.get(name, (0, 0))
    BY_KIND[name] = (c + 1, e + n)
    if RECORD[0]:
        if isinstance(r, _np.ndarray):
            snap = _np.array(r, copy=True).view(_np.ndarray)
        elif isinstance(r, tuple):
            snap = tuple(_np.array(x, copy=True).view(_np.ndarray)
                         if isinstance(x, _np.ndarray) else None for x in r)
        else:
            snap = None
        TRACE.append((name, snap))


def _bare(x):
    return x.view(_np.ndarray) if isinstance(x, CA) else x


def _wrap(x):
    if isinstance(x, _np.ndarray) and not isinstance(x, CA):
        return x.view(CA)
    if isinstance(x, tuple):
        return tuple(_wrap(i) for i in x)
    return x


class CA(_np.ndarray):
    """ndarray that counts every ufunc / array-function it takes part in."""

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        ins = tuple(_bare(x) for x in inputs)
        if "out" in kw and kw["out"] is not None:
            kw["out"] = tuple(_bare(o) for o in kw["out"])
        r = getattr(ufunc, method)(*ins, **kw)
        _note(ufunc.__name__, r)
        return _wrap(r)

    def __array_function__(self, func, types, args, kwargs):
        def u(x):
            if isinstance(x, CA):
                return _bare(x)
            if isinstance(x, (list, tuple)):
                return type(x)(u(i) for i in x)
            if isinstance(x, dict):
                return {k: u(v) for k, v in x.items()}
            return x
        r = func(*u(args), **u(kwargs))
        _note(func.__name__, r)
        return _wrap(r)


_CREATORS = ("zeros", "ones", "empty", "full", "arange", "zeros_like",
             "ones_like", "empty_like", "asarray", "array")


def __getattr__(name):
    v = getattr(_np, name)
    if name in _CREATORS and callable(v):
        def mk(*a, __f=v, __n=name, **k):
            r = __f(*a, **k)
            if __n in ("zeros", "ones", "empty", "full", "arange"):
                _note("alloc:" + __n, r)
            return _wrap(r)
        return mk
    return v
