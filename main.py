import pandas as pd
import numpy as np

class Pose:
    def __init__(self, pose, x):
        self.pose = pose
        self.x = data
        self.prior = 0


def preprocess(filename):
    y = []
    x = []
    f = open("train.csv",'r')
    for line in f.readlines():
        atts = line.strip().split(',')
        y.append(atts[0])
        floatX = []
        for att in atts[1:]:
            floatX.append(float(att))
        x.append(floatX)
    f.close()
    return x,y

# def train(x, y):
#     model = []
#     data = []
#     cur_pose = y[0]
#     for i in range(len(x)):
#         if y[i] != cur_pose:
            
#             model.append(Pose(cur_pose, data))
#             data = []
#         for j in range(len(x[i])):
#             data[j].append(x[i][j])
#     return model


filename = "train.csv"
# x, y = preprocess(filename)
# train(x, y)

a = pd.read_csv(filename, header = None)
b = sorted(list(set(a[a.columns[0]])))
c = a.loc[a[0] == b[0]]
c = c[c.columns[1:]]
# c = c.transpose()
c = c.replace(9999, np.NaN)
d = np.nanmean(c, axis=0)
e = np.nanstd(c,axis= 0)
print(e)

