# -*- coding: utf-8 -*-
"""
Created on Sat May  2 02:45:29 2020

@author: aaaambition
"""

def make_debug_print(flags_set):
    def debug_print(*args, flags=set()):
        if not isinstance(flags, list):
            flags = [flags]
        if flags_set & set(flags):
            print(*args)
    return debug_print
