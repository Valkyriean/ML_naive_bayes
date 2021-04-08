import pandas as pd
import numpy as np

class Pose:
    def __init__(self, pose, mean, std, prior):
        self.pose = pose
        self.mean = mean
        self.std = std
        self.prior = prior


def preprocess(filename):
    return pd.read_csv(filename, header = None).replace(9999, np.NaN)


def train(data):
    Model = []
    labels = sorted(list(set(data[data.columns[0]])))
    for label in labels:
        pose_data = data.loc[data[0] == label]
        pose_data = pose_data[pose_data.columns[1:]]
        pose = Pose(label, 
                    np.nanmean(pose_data, axis=0),
                    np.nanstd(pose_data, axis=0),
                    len(pose_data)/len(data)
                    )


filename = "train.csv"
# x, y = preprocess(filename)
# train(x, y)

a = pd.read_csv(filename, header = None)
a = a.replace(9999, np.NaN)
b = sorted(list(set(a[a.columns[0]])))
c = a.loc[a[0] == b[0]]
c = c[c.columns[1:]]
# c = c.transpose()
d = np.nanmean(c, axis=0)
e = np.nanstd(c,axis= 0)
print(len(c))

