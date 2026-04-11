import numpy as np

def mapping(x, in_min, in_max):
    out_min = 0
    out_max = 1
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def unmapping(x, out_min, out_max):
    in_min = 0
    in_max = 1
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min