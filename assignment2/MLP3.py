import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import *
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV
trainfile = "train_data_E3E5_10genes.txt"  # 每次进行测试时请先修改对应的文件使其正确
testfile = "test_data_E3E5_10genes.txt"

#打开训练集
TrainSet =pd.read_table(trainfile, sep=' ') #读入数据
TrainLabel = (TrainSet.values).T[TrainSet.shape[1]-1] # 筛选出每个样本的label作为Y向量
TrainFeature = np.log(np.asmatrix((TrainSet.values).T[0:TrainSet.shape[1]-1].T).astype(float)+1e-6) # 筛选出每个样本的Feature（每个样本有两个Feature），应该是样本数*2的矩阵

TestSet =pd.read_table(testfile, sep=' ') #读入数据
TestLabel = (TestSet.values).T[TestSet.shape[1]-1] # 筛选出每个样本的label作为Y向量
TestFeature = np.log(np.asmatrix((TestSet.values).T[0:TestSet.shape[1]-1].T).astype(float)+1e-6) # 筛选出每个测试集样本的Feature（每个样本有两个Feature），应该是样本数*2的矩阵

X_train=TrainFeature
X_test=TestFeature
y_train=TrainLabel
y_test=TestLabel
X_trainscaled = preprocessing.scale(X_train)
X_testscaled = preprocessing.scale(X_test)

y_trainscaled=[]
y_testscaled=[]

for i in range(0,y_train.shape[0]):
    if y_train[i]=='E3':
        y_trainscaled.append(1)  # "E3"的时候赋值为1
    else:
        y_trainscaled.append(0)
for i in range(0,y_test.shape[0]):
    if y_test[i]=='E3':
        y_testscaled.append(1)  # "E3"的时候赋值为1
    else:
        y_testscaled.append(0)
'''
h = .02  # step size in the mesh

alphas = np.logspace(-4, 2, 4)
names = []
for i in alphas:
    names.append('alpha ' + str(i))

classifiers = []
for i in alphas:
    classifiers.append(MLPClassifier(alpha=i, random_state=1,max_iter=500))

figure = plt.figure(figsize=(20, 3))
i = 1

x_min, x_max = X_trainscaled[:, 0].min() - .5, X_trainscaled[:, 0].max() + .5
y_min, y_max = X_trainscaled[:, 1].min() - .5, X_trainscaled[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF']) # 红色和蓝色

# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(1, len(classifiers) + 1, i)
    clf.fit(X_trainscaled, y_trainscaled)
    score = clf.score(X_testscaled, y_testscaled)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.5)

    # Plot also the training points
    ax.scatter(X_trainscaled[:, 0], X_trainscaled[:, 1], c=y_trainscaled, cmap=cm_bright,edgecolors='w', s=25)
    # and testing points
    #ax.scatter(X_testscaled[:, 0], X_testscaled[:, 1], c=y_testscaled, cmap=cm_bright,alpha=0.6, edgecolors='w', s=25)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),size=15, horizontalalignment='right')
    i += 1

figure.subplots_adjust(left=.02, right=.98)
plt.show()
i = 1
# 第二部分，测试adam和lbfgs,和sgd两种分类结果
names2 = ['adam','lbfgs','sgd']
classifiers2 = []
for i in [0,1,2]:
    classifiers2.append(MLPClassifier(solver=names2[i],alpha=1e-4, random_state=1,max_iter=500))

figure = plt.figure(figsize=(15, 3))

x_min, x_max = X_trainscaled[:, 0].min() - .5, X_trainscaled[:, 0].max() + .5
y_min, y_max = X_trainscaled[:, 1].min() - .5, X_trainscaled[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF']) # 红色和蓝色

# iterate over classifiers
for name, clf in zip(names2, classifiers2):
    ax = plt.subplot(1, len(classifiers2) + 1, i)
    clf.fit(X_trainscaled, y_trainscaled)
    score = clf.score(X_testscaled, y_testscaled)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.5)

    # Plot also the training points
    ax.scatter(X_trainscaled[:, 0], X_trainscaled[:, 1], c=y_trainscaled, cmap=cm_bright,edgecolors='w', s=25)
    # and testing points
    #ax.scatter(X_testscaled[:, 0], X_testscaled[:, 1], c=y_testscaled, cmap=cm_bright,alpha=0.6, edgecolors='w', s=25)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),size=15, horizontalalignment='right')
    i += 1
figure.subplots_adjust(left=.02, right=.98)
plt.show()
'''
# 还差一个十折交叉验证，然后输出一下训练正确率和测试正确率
'''
# 用GridSearchCV网格法找出最好的C
param_grid = [{'hidden_layer_sizes':[(5, 5), (10, 10), (30, 30),(100, 100),(5,5,5),(10,10,10),(100,100,100)], 'alpha':[0.0001, 0.001, 0.01, 0.1, 0.5]}]

def print_best_score(gsearch,param_test):
     # 输出best score
    print("Best score: %0.8f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(dict(param_test).keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        print(best_parameters['alpha'])

gsearch = GridSearchCV(MLPClassifier(max_iter= 2000,random_state=0), param_grid = param_grid, scoring='roc_auc', cv=10 )  # cv为整数k时为k折交叉验证
gsearch.fit(X_trainscaled, y_train)
print_best_score(gsearch, param_grid)
'''
cls=MLPClassifier(alpha=0.0001,hidden_layer_sizes=(30,30),max_iter=2000,random_state=0)
cls.fit(X_trainscaled,y_train)
print(cls.score(X_testscaled,y_test))
