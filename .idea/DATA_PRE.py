import pandas as pd
from urllib.request import urlretrieve

def load_data(download=True):
    # download data from : http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
    if download:
        data_path, _ = urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", "car.csv")
        print("Downloaded to car.csv")

    # use pandas to view the data structure
    col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    data = pd.read_csv("car.csv", names=col_names)
    return data