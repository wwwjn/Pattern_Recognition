import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn import *
from sklearn.svm import LinearSVC

trainfile ="train_data_E3E5_10genes.txt"  # 每次进行测试时请先修改对应的文件使其正确
testfile="test_data_E3E5_10genes.txt"

#打开训练集
TrainSet =pd.read_table(trainfile, sep=' ') #读入数据
TrainLabel = (TrainSet.values).T[TrainSet.shape[1]-1] # 筛选出每个样本的label作为Y向量
TrainFeature = np.log(np.asmatrix((TrainSet.values).T[0:TrainSet.shape[1]-1].T).astype(float)+1e-10) # 筛选出每个样本的Feature（每个样本有两个Feature），应该是样本数*2的矩阵
TrainFeature=preprocessing.scale(TrainFeature)

TestSet =pd.read_table(testfile, sep=' ') #读入数据
TestLabel = (TestSet.values).T[TestSet.shape[1]-1] # 筛选出每个样本的label作为Y向量
TestFeature = np.log(np.asmatrix((TestSet.values).T[0:TestSet.shape[1]-1].T).astype(float)+1e-10)
TestFeature=preprocessing.scale(TestFeature)

clf = LinearSVC(C=0.01,penalty="l1",dual=False)
clf.fit(TrainFeature, TrainLabel)
print(clf.coef_)

# Set a minimum threshold of 0.25
sfm = SelectFromModel(clf, prefit=True,threshold=0.05)

X_transform=sfm.transform(TrainFeature)
n_features = sfm.transform(TrainFeature).shape[1]
print(X_transform.shape)

# Reset the threshold till the number of features equals two.
# Note that the attribute can be set directly instead of repeatedly
# fitting the metatransformer.

while n_features > 3:
    sfm.threshold += 0.0001
    X_transform = sfm.transform(TrainFeature)
    n_features = X_transform.shape[1]

print(X_transform[:,0]==TrainFeature[:,0])
print(X_transform[:,1]==TrainFeature[:,1])
print(X_transform[:,2]==TrainFeature[:,9])


'''
# Plot the selected two features from X.
plt.title(
    "Features selected from Boston using SelectFromModel with "
    "threshold %0.3f." % sfm.threshold)
feature1 = X_transform[:, 0]
feature2 = X_transform[:, 1]
feature
plt.plot(feature1, feature2, 'r.')
plt.xlabel("Feature number 1")
plt.ylabel("Feature number 2")
plt.ylim([np.min(feature2), np.max(feature2)])
plt.show()
'''