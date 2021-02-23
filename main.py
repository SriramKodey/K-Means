import pandas as pd
import numpy as np
from kmean import kmeans_cluster

def solve(data):

    for i in range(2, 11):    
        kmeans = kmeans_cluster(data, i)

        kmeans.cluster()
        kmeans.plot()
        kmeans.metrics()



if __name__ == "__main__":
    np.set_printoptions(precision = 1)
    df = pd.read_csv("C:/Users/kodey/Documents/Python Scripts/Data Mining/data/1.csv", sep = ',' , header = None)
    data = np.delete(df.values, 20, 1)

    solve(data)
