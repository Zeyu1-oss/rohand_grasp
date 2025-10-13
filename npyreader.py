#!/usr/bin/env python3
import numpy as np
import argparse
import pprint

def main():
    parser = argparse.ArgumentParser(description="Print full contents of a .npy file")
    parser.add_argument("npy_file", help="Path to the .npy file")
    args = parser.parse_args()

    data = np.load(args.npy_file, allow_pickle=True)
    print(f"Loaded npy file: {args.npy_file}")
    print(f"Type: {type(data)}")
    print(f"Shape: {getattr(data, 'shape', 'N/A')}\n")

    # 用 pprint 打印更清晰
    pprint.pprint(data)

if __name__ == "__main__":
    main()
