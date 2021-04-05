"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from fontTools.ttLib import TTFont
from itertools import chain
from PIL import Image, ImageFont, ImageDraw


def get_available_chars(fontfile):
    ttf = TTFont(fontfile)
    chars = sorted(set(chain.from_iterable([chr(y) for y in x.cmap.keys()] for x in ttf["cmap"].tables)))
    return chars


def read_font(fontfile, size=150):
    font = ImageFont.truetype(str(fontfile), size=size)
    return font


def render(font, char, size=(128, 128), pad=20):
    width, height = font.getsize(char)
    max_size = max(width, height)

    if width < height:
        start_w = (height - width) // 2 + pad
        start_h = pad
    else:
        start_w = pad
        start_h = (width - height) // 2 + pad

    img = Image.new("L", (max_size+(pad*2), max_size+(pad*2)), 255)
    draw = ImageDraw.Draw(img)
    draw.text((start_w, start_h), char, font=font)
    img = img.resize(size, 2)
    return img
