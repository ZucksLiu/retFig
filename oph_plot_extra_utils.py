
import os
from heapq import merge
import pandas as pd


def capitalize_first_letter(s: str) -> str:
    return s[0].upper() + s[1:]