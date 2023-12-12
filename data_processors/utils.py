""" Data Processing utils functions"""
# Libraries
import numpy as np


# Gaussian Function
def gauss(x, mean, std_dev):
    c = 1 / np.sqrt(2 * np.pi * std_dev)
    num = (x - mean) ** 2
    den = 2 * (std_dev ** 2)
    return c * np.exp(-num / den)
