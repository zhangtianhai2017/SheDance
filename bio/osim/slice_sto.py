#!/usr/bin/env python3
"""Slice the first N data frames of an .sto (keep header). Args: in_sto out_sto N"""
import sys
inp, outp, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = open(inp).read().splitlines()
hi = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader")
out = lines[:hi + 2] + lines[hi + 2:hi + 2 + n]
open(outp, "w").write("\n".join(out) + "\n")
print(f"sliced {min(n, len(lines) - (hi + 2))} frames -> {outp}")
