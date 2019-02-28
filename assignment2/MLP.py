import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn import datasets,linear_model,svm,preprocessing
from sklearn.neural_network import MLPClassifier
import pandas as pd
trainfile ="train_data_E3E5_2genes.txt"  # 每次进行测试时请先修改对应的文件使其正确
testfile="test_data_E3E5_2genes.txt"

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

clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,5),random_state=0)
clf.fit(X_trainscaled,y_train)
print(clf.score(X_testscaled,y_test))

figure = plt.figure(figsize=(17, 9))
h=.02
x_min, x_max = X_trainscaled[:, 0].min() - .5, X_trainscaled[:, 0].max() + .5
y_min, y_max = X_trainscaled[:, 1].min() - .5, X_trainscaled[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
'''
ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
 # and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
'''
# iterate over classifiers
ax = plt.subplot(1,1,1)
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
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='black', s=25)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6, edgecolors='black', s=25)

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_title('MLP')
ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')

figure.subplots_adjust(left=.02, right=.98)
plt.show()


E3Mat = np.mat([0.0,0.0])
E5Mat = np.mat([0.0,0.0])
for i in range(0,TrainFeature.shape[0]):  # 将E3和E5分别存入两个矩阵
    if TrainLabel[i] == 'E3': # 剔除掉偏差过大的点
        E3Mat = np.row_stack((E3Mat, X_trainscaled[i].astype(float))) # 加上一个很小的数，防止有0存在
    elif TrainLabel[i] == 'E5'and TrainFeature[i,0]<10000 and 0<TrainFeature[i,1]<20000:
        E5Mat = np.row_stack((E5Mat, X_trainscaled[i].astype(float)))

E3Mat= E3Mat[1:]  # 此时E3Mat和E5Mat第一行都是[0,0]，使用时应注意删掉
E5Mat= E5Mat[1:]

plt.figure(1,figsize=(4,3))
X1 = np.asarray(E3Mat.T[0])
Y1 = np.asarray(E3Mat.T[1])
X2 = np.asarray(E5Mat.T[0])
Y2 = np.asarray(E5Mat.T[1])
p1=plt.scatter(X1,Y1,marker = 'o', color = 'r',label='1',s=10)
p2=plt.scatter(X2,Y2,marker = 'o',color ='b',label='2',s=10)

plt.legend((p1, p2), ('E3', 'E5'), loc=2)
plt.xlabel("Gene1" )
plt.ylabel("Gene2")

plt.xticks(())
plt.yticks(())


plt.show()
