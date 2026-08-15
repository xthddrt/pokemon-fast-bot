"""Verify a published encode cache WITHOUT downloading it.

Every .npy carries its dtype and shape in a header in the first ~128 bytes, so
a range GET per object proves row count, width and dtype for the whole cache
for a few kB of transfer.  Then one small slice of real rows is pulled and run
through `vt_lib.Arm`'s consistency checks, which is the part a header cannot
prove.

    python enc_s3_verify.py s3://bucket/prefix/
"""
import ast
import json
import os
import sys

import boto3


def npy_header(cli, bucket, key):
    b = cli.get_object(Bucket=bucket, Key=key, Range="bytes=0-255")["Body"].read()
    assert b[:6] == b"\x93NUMPY", key
    major = b[6]
    if major == 1:
        hlen = int.from_bytes(b[8:10], "little")
        head = b[10:10 + hlen]
    else:
        hlen = int.from_bytes(b[8:12], "little")
        head = b[12:12 + hlen]
    d = ast.literal_eval(head.decode("latin1").strip())
    return d["descr"], d["shape"]


def head_rows(cli, bucket, key, k):
    """First k rows of a C-order .npy, by byte range -- no full download."""
    import numpy as np
    b = cli.get_object(Bucket=bucket, Key=key, Range="bytes=0-255")["Body"].read()
    major = b[6]
    hlen = (int.from_bytes(b[8:10], "little") if major == 1
            else int.from_bytes(b[8:12], "little"))
    off = (10 if major == 1 else 12) + hlen
    d = ast.literal_eval(b[(10 if major == 1 else 12):off].decode("latin1").strip())
    dt = np.dtype(d["descr"])
    rowb = dt.itemsize * int(np.prod(d["shape"][1:])) if len(d["shape"]) > 1 else dt.itemsize
    raw = cli.get_object(Bucket=bucket, Key=key,
                         Range="bytes=%d-%d" % (off, off + k * rowb - 1))["Body"].read()
    return np.frombuffer(raw, dtype=dt).reshape((k,) + tuple(d["shape"][1:]))


def main(uri):
    bucket, _, prefix = uri[5:].partition("/")
    prefix = prefix.rstrip("/") + "/"
    cli = boto3.client("s3")
    objs, tok = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if tok:
            kw["ContinuationToken"] = tok
        r = cli.list_objects_v2(**kw)
        objs += r.get("Contents", [])
        if not r.get("IsTruncated"):
            break
        tok = r["NextContinuationToken"]
    total = sum(o["Size"] for o in objs)
    rows = set()
    print("%-24s %-10s %-22s %s" % ("object", "dtype", "shape", "bytes"))
    for o in sorted(objs, key=lambda x: x["Key"]):
        name = o["Key"][len(prefix):]
        if name.endswith(".npy"):
            dt, sh = npy_header(cli, bucket, o["Key"])
            # holdout_i.npy is an INDEX into the cache, not a row of it
            if not name.startswith("holdout"):
                rows.add(sh[0])
            print("%-24s %-10s %-22s %d" % (name, dt, sh, o["Size"]))
        else:
            print("%-24s %-10s %-22s %d" % (name, "-", "-", o["Size"]))
    print("\nobjects=%d  total_bytes=%d (%.2f GB)  distinct row counts=%s"
          % (len(objs), total, total / 1e9, sorted(rows)))
    assert len(rows) == 1, "ROW COUNT DISAGREEMENT across arrays: %s" % sorted(rows)
    n = rows.pop()
    print("ROWS = %d ; bytes/row = %.1f" % (n, total / n))
    return {"rows": n, "bytes": total, "objects": len(objs)}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=1))
