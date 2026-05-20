"""
Compatibility wrapper for the old eval entrypoint.

The streaming inference implementation now lives in inference/infer.py. Metric
calculation is handled by scripts under evaluate/.
"""
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)

from inference.infer import main


if __name__ == "__main__":
    main()
