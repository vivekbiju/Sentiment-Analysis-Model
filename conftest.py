import sys
import os

# This tells pytest to look at the root folder for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))