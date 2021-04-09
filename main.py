import pandas as pd
import numpy as np
from scipy.stats import norm

def preprocess(filename):
    return pd.read_csv(filename, header = None).replace(9999, np.NaN)


def train(data):
    model = []
    labels = sorted(list(set(data[data.columns[0]])))
    for label in labels:
        pose_data = data.loc[data[0] == label]
        pose_data = pose_data[pose_data.columns[1:]]
        norm_list = []
        mean_list = np.nanmean(pose_data, axis=0)
        std_list = np.nanstd(pose_data, axis=0)
        for i in range(len(mean_list)):
            norm_list.append(norm(mean_list[i], std_list[i]))
        prior = len(pose_data)/len(data)
        pose = [label, 
                prior,
                norm_list
                ]

        model.append(pose)
    return model

def predict(test, model):
    predictions = []
    data = test[test.columns[1:]]
    for index, instance in data.iterrows():
        prediction = ""
        best_score = 0
        first = True
        for pose in model:
            score = np.log(pose[1])
            for i in range(len(instance)):
                if np.isnan(instance.iloc[i]): 
                    continue
                score += pose[2][i].logpdf(instance.iloc[i])
            if first is True:
                best_score = score
                prediction = pose[0]
                first = False
            elif score > best_score:
                best_score = score
                prediction = pose[0]
        predictions.append(prediction)
    return predictions

def evaluate(predictions, test):
    label = data[data.columns[0]]
    total = len(predictions)
    correct = 0
    for i in range(len(predictions)):
        if label[i] == predictions[i]:
            correct+=1
    return correct/total

filename = "train.csv"
data = preprocess(filename)
model = train(data)
test = preprocess("test.csv")
predictions = predict(test, model)
acc = evaluate(predictions, test)
print(acc)
