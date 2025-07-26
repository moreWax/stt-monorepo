#!/usr/bin/env python3
"""
verify_extension_files.py

Checks that required Chrome extension files exist in the correct locations and are not empty.
"""
import os
import sys


REQUIRED_FILES = [
    "manifest.json",
    "popup.js",
    "background.js",
    "whisperWorker.js"
]

def main():
    missing = []
    empty = []
    for path in REQUIRED_FILES:
        if not os.path.isfile(path):
            missing.append(path)
        else:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        empty.append(path)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                empty.append(path)
    if missing:
        print("Missing files:")
        for f in missing:
            print(f"  - {f}")
    if empty:
        print("Empty files:")
        for f in empty:
            print(f"  - {f}")
    if not missing and not empty:
        print("All required files are present and non-empty.")
    if missing or empty:
        sys.exit(1)

if __name__ == "__main__":
    main()
