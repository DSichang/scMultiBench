"""Stand-in for the logger module that MIRA's main_MIRA.py imports.

The script runs ``from logger import *`` and uses nothing from it, and the
public scMultiBench repository does not include the file. multibench copies
this empty module next to main_MIRA.py before the first run.
"""
