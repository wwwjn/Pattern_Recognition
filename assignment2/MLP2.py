import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd

trainfile = "train_data_E3E5_2genes.txt"  # 每次进行测试时请先修改对应的文件使其正确
testfile = "test_data_E3E5_2genes.txt"

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

for i in y_train.shape[0]:
    if y_train[i]=='E3':
        y_trainscaled.append(1)  # "E3"的时候赋值为1
    else:
        y_trainscaled.append(0)
for i in y_test.shape[0]:
    if y_test[i]=='E3':
        y_testscaled.append(1)  # "E3"的时候赋值为1
    else:
        y_testscaled.append(0)

print(y_testscaled)


h = .02  # step size in the mesh

alphas = np.logspace(-5, 3, 5)
names = []
for i in alphas:
    names.append('alpha ' + str(i))

classifiers = []
for i in alphas:
    classifiers.append(MLPClassifier(alpha=i, random_state=1))
'''
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=0, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]
'''
figure = plt.figure(figsize=(17, 9))
i = 1
'''
# iterate over datasets
for X, y in datasets:
    # preprocess dataset, split into training and test part
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
    '''
x_min, x_max = X_trainscaled[:, 0].min() - .5, X_trainscaled[:, 0].max() + .5
y_min, y_max = X_trainscaled[:, 1].min() - .5, X_trainscaled[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.subplot(1,len(classifiers) + 1,1)
# Plot the training points
ax.scatter(X_trainscaled[:, 0], X_trainscaled[:, 1], c=y_trainscaled, cmap=cm_bright)
# and testing points
ax.scatter(X_testscaled[:, 0], X_testscaled[:, 1], c=y_testscaled, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(1, len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,edgecolors='black', s=25)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,alpha=0.6, edgecolors='black', s=25)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),size=15, horizontalalignment='right')
    i += 1

figure.subplots_adjust(left=.02, right=.98)
plt.show()