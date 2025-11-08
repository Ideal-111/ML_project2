from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from NN import *
from nn_train import nn_train
from nn_test import nn_test
import numpy as np
import os
import matplotlib.image as mpimg
import pickle
from nn_forward import nn_forward
from nn_predict import nn_predict
from nn_backward import nn_backward
from nn_applygradient import nn_applygradient
from function import sigmoid, softmax
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variable(filename):
    f=open(filename,'rb')
    r=pickle.load(f)
    f.close()
    return r

def load_data(dir_path, total_count):
    file_ls = os.listdir(dir_path)
    data = np.zeros((total_count, 784), dtype=float)
    label = np.zeros((total_count, 10), dtype=float)
    flag = 0
    
    # 使用tqdm创建进度条
    with tqdm(total=total_count, desc=f"加载{dir_path.split(os.sep)[-1]}数据") as pbar:
        for dir in file_ls:
            files = os.listdir(os.path.join(dir_path, dir))
            for file in files:
                filename = os.path.join(dir_path, dir, file)
                img = mpimg.imread(filename)
                data[flag,:] = np.reshape(img, -1)/255
                label[flag, int(dir)] = 1.0
                flag += 1
                pbar.update(1)
    return data, label

# # Training
# dir_path = 'D:\\dataset\\MNIST\\train'
# file_ls = os.listdir(dir_path)
# data = np.zeros((60000, 784), dtype=float)
# label = np.zeros((60000, 10), dtype=float)
# flag = 0
# for dir in file_ls:
#     files = os.listdir(dir_path+'\\'+dir)
#     for file in files:
#         filename = dir_path+'\\'+dir+'\\'+file
#         img = mpimg.imread(filename)
#         data[flag,:] = np.reshape(img, -1)/255
#         label[flag, int(dir)] = 1.0
#         flag+=1
train_dir = 'D:\\dataset\\MNIST\\train'
data, label = load_data(train_dir, 60000)
ratioTraining = 0.95
xTraining, xValidation, yTraining, yValidation = train_test_split(data, label, test_size=1 - ratioTraining, random_state=0)  # 随机分配数据集


if os.path.exists('storedNN_MNIST.npz'):
    nn = load_variable('storedNN_MNIST.npz')
else:
    nn = NN(layer=[784,400,169,49,10], 
            batch_normalization = 1, 
            active_function='relu', 
            batch_size = 50, 
            learning_rate=0.01, 
            optimization_method='Adam', 
            objective_function='Cross Entropy')

epoch = 0
maxAccuracy = 0
totalAccuracy = []
totalCost = []
maxEpoch = 100
with tqdm(total=maxEpoch, desc="Train Process") as pbar_epoch:
    while epoch < maxEpoch:
        epoch += 1
        nn = nn_train(nn, xTraining, yTraining)
        totalCost.append(sum(nn.cost.values()) / len(nn.cost.values()))
        wrongs, predictedLabel, accuracy, y_output = nn_test(nn, xValidation, yValidation)
        totalAccuracy.append(accuracy)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            storedNN = nn
            save_variable(nn, 'storedNN_MNIST.npz')
        cost = totalCost[epoch - 1]
        # print('Epoch:', epoch)
        # print('Accuracy:',accuracy)
        # print('Cost:',cost)

        tqdm.write(f'Epoch: {epoch}, Accuracy: {accuracy:.4f}, Cost: {cost:.6f}')
        
        pbar_epoch.update(1)

test_dir = 'D:\\dataset\\MNIST\\test'
xTesting, yTesting = load_data(test_dir, 10000)

# Testing
# dir_path = 'D:\\dataset\\MNIST\\test'
# file_ls = os.listdir(dir_path)
# xTesting = np.zeros((10000, 784), dtype=float)
# yTesting = np.zeros((10000, 10), dtype=float)
# flag = 0
# for dir in file_ls:
#     files = os.listdir(dir_path+'\\'+dir)
#     for file in files:
#         filename = dir_path+'\\'+dir+'\\'+file
#         img = mpimg.imread(filename)
#         xTesting[flag,:] = np.reshape(img, -1)/255
#         yTesting[flag, int(dir)] = 1.0
#         flag+=1

if os.path.exists('storedNN_MNIST.npz'):
    storedNN = load_variable('storedNN_MNIST.npz')
    wrongs, predictedLabel, accuracy, y_output = nn_test(storedNN, xTesting, yTesting)
    print('Accuracy on Test set:', accuracy)
    confusionMatrix = np.zeros((10,10),dtype=int)
    for i in range(len(predictedLabel)):
        trueLabel = np.argmax(yTesting[i,:])
        confusionMatrix[trueLabel,predictedLabel[i]]+=1
    print('The Confusion Matrix is:\n', confusionMatrix)

    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10),
                yticklabels=range(10))
    plt.xlabel('Predicted Label', fontsize=12) 
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix for MNIST Classification', fontsize=14)
    plt.savefig('confusion matrix_MNIST.png', dpi=300)