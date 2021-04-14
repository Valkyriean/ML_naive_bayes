import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import shapiro
from collections import Counter
import statsmodels.api as sm

# Reading data from file and convert to dataframe while replacing 9999 to NaN
def preprocess(filename):
    return pd.read_csv(filename, header = None).replace(9999, np.NaN)

# Calculate prior probabilities and normal distribution of each attribute for each class
def train(data):
    model = []
    # Non-repeated class label list
    labels = sorted(list(set(data[data.columns[0]])))
    for label in labels:
        # Split to only take data from given class
        pose_data = data.loc[data[0] == label]
        pose_data = pose_data[pose_data.columns[1:]]
        mean_list = np.nanmean(pose_data, axis=0)
        std_list = np.nanstd(pose_data, axis=0)
        norm_list = []
        for i in range(len(mean_list)):
            norm_list.append(norm(mean_list[i], std_list[i]))
        prior = len(pose_data)/len(data)
        pose = [label, 
                prior,
                norm_list]
        model.append(pose)
    return model

# Predict labels of test set based on training model
def predict(test, model):
    predictions = []
    # drop labels from test set
    data = test[test.columns[1:]]
    for index, instance in data.iterrows():
        prediction = ""
        best_score = 0
        first = True
        # Calculate probabilities of each class
        for pose in model:
            # log prior probabilities
            score = np.log(pose[1])
            # sum up log of likelihood of features
            for i in range(len(instance)):
                # Ignore the missing features
                if np.isnan(instance.iloc[i]): 
                    continue
                else: 
                    score += pose[2][i].logpdf(instance.iloc[i])
            if first is True:
                best_score = score
                prediction = pose[0]
                first = False
            # Predict the highest socre class
            elif score > best_score:
                best_score = score
                prediction = pose[0]
        predictions.append(prediction)
    return predictions


# Evaluateing the predictions
def evaluate(data, predictions, test):
    truth = test[test.columns[0]]
    labels = sorted(list(set(data[data.columns[0]])))
    count = Counter(list(test[test.columns[0]]))
    total = len(predictions)
    # initialize  the error table
    error_table = {}
    for label in labels:
        # TP FN FP TN
        error_table[label] = [0,0,0,0]
    correct = 0
    for i in range(len(predictions)):
        if truth[i] == predictions[i]:
            # TP +1 for truth class and TN +1 for rest classes
            error_table[truth[i]][0] += 1
            for key, value in error_table.items():
                if key != truth[i]:
                    error_table[key][3] += 1
            correct+=1
        else:
            # FN +1 for truth class and FP +1 for predicted class
            error_table[truth[i]][1] +=1
            error_table[predictions[i]][2] +=1
    accuracy = correct/total
    macro_p = 0
    macro_r = 0
    micro_p_n = 0
    micro_p_d = 0
    micro_r_n = 0
    micro_r_d = 0
    weight_p = 0
    weight_r = 0
    f_score = {}
    # Calculate performance measurements based on error table
    for label in labels:
        tp = error_table[label][0]
        fn = error_table[label][1]
        fp = error_table[label][2]
        tn = error_table[label][3]
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        macro_p += precision
        macro_r += recall
        f_score[label] = 2*precision*recall /(precision+recall)
        micro_p_n += tp
        micro_p_d += tp+fp
        micro_r_n += tp
        micro_r_d += tp+fn
        weight_p += count[label]/total*precision
        weight_r += count[label]/total*recall
    # printing out the performance measurements results
    print("Overall Accuracy: " + str(accuracy))
    macro_p = macro_p/len(labels)
    macro_r = macro_r/len(labels)
    macro_f = 2*macro_p*macro_r / (macro_p+macro_r)
    print("\nMacro-averaging:")
    print("Precision: " + str(macro_p))
    print("Recall: " + str(macro_r))
    print("F-score: " + str(macro_f))
    micro_p = micro_p_n/micro_p_d
    micro_r = micro_r_n/micro_r_d
    micro_f = 2*micro_p*micro_r / (micro_p+micro_r)
    print("\nMicro-averaging:")
    print("Precision: " + str(micro_p))
    print('Recall: ' + str(micro_r))
    print("F-score: " + str(micro_f))
    weight_f = 2*weight_p*weight_r / (weight_p+weight_r)
    print("\nWeighted averaging:")
    print("Precision: " + str(weight_p))
    print('Recall: ' + str(weight_r))
    print("F-score: " + str(weight_f))
    x = []
    y = []
    # correspond count to f-score
    for label in labels:
        x.append(count[label])
        y.append(f_score[label])
    # printing out graph for question 1
    plt.title("F-score vs Count")
    plt.xlabel("Count of class in test set")
    plt.ylabel("F-socre of class")
    plt.scatter(x,y)
    plt.show()
    return

# Question 1
data = preprocess("train.csv")
model = train(data)
test = preprocess("test.csv")
predictions = predict(test, model)
evaluate(data, predictions, test)

# Question 2
labels = sorted(list(set(data[data.columns[0]])))
count = 0
example = True
for label in labels:
    # Split to only take data from given class
    pose_data = data.loc[data[0] == label]
    pose_data = pose_data[pose_data.columns[1:]]
    for index, content in pose_data.items():
        content = content.dropna()
        stat, p = shapiro(content)
        # Reject H0 at alpha = 0.05
        if(p < 0.05):
            # Showing first non-Gaussian distribution as example
            if example:
                if index >= 11:
                    point = "Y" + str(index-10)
                else:
                    point = "X" + str(index+1)
                print("For example: point " + point + " column of "+ label + " class has violated the Gaussian assumption with p-value of " + str(p))
                plt.hist(content)
                plt.show()
                sm.qqplot(content, line ='s')
                plt.show()
                example = False
            count+=1
print("There have been " + str(count) + " out of " + str(len(labels)*22) + " attributes of classes that violated the Gaussian assumption")