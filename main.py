import pandas as pd


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

def train(x, y):
    model = []
    data = [len(x[0])][0]
    cur_pose = y[0]
    for i in range(len(x)):
        if y[i] != cur_pose:
            
            model.append(Pose(cur_pose, data))
            data = [len(x[0])][0]
        for j in range(len(x[i])):
            data[j].append(x[i][j])
    return model


filename = "train.csv"
# print(preprocess(filename))

a = pd.read_csv(filename)
print(a)