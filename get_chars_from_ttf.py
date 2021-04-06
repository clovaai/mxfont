import sys
from tqdm import tqdm
from pathlib import Path
import numpy as np

from datasets import read_font, render, get_filtered_chars


def main():
    root_dir = sys.argv[1]
    print(root_dir)
    ttffiles = list(Path(root_dir).rglob("*.ttf"))
    
    for ttffile in tqdm(ttffiles):
        filename = ttffile.stem
        dirname = ttffile.parent
        avail_chars = get_filtered_chars(ttffile)
        with open((dirname / (filename+".txt")), "w") as f:
            f.write("".join(avail_chars))
            

if __name__ == "__main__":
    main()