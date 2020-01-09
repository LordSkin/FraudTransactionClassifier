import numpy as np

file_name = "creditcard.csv"


def read_data():
    file = open(file_name)
    file.readline()  # skip the header
    data = np.loadtxt(file, delimiter=',')
    return data
