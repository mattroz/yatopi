import os
import numpy as np
import pandas as pd


def retrieve_class_names(filename):
    assert os.path.exists(filename), "Cannot find file"
    contents = pd.read_csv(filename, header=None)
    return contents.to_numpy()