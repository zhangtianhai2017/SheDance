#!/usr/bin/env python3
"""Horizontally montage labeled PNGs. Args: out.png  label=path  label=path ..."""
import sys
from PIL import Image, ImageDraw
out = sys.argv[1]; items = sys.argv[2:]
imgs = []
for it in items:
    label, path = it.split("=", 1)
    im = Image.open(path).convert("RGB")
    ImageDraw.Draw(im).text((10, 10), label, fill=(255, 230, 60))
    imgs.append(im)
Wtot = sum(i.width for i in imgs); Hmax = max(i.height for i in imgs)
sheet = Image.new("RGB", (Wtot, Hmax), (20, 20, 28)); x = 0
for i in imgs:
    sheet.paste(i, (x, 0)); x += i.width
sheet.save(out); print("montage ->", out)
