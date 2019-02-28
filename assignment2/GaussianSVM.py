import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model,svm,preprocessing
from sklearn.model_selection import GridSearchCV
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
# 将数据进行preprocessing处理
X_train=TrainFeature
X_test=TestFeature
y_train=TrainLabel
y_test=TestLabel
X_trainscaled = preprocessing.scale(X_train)
X_testscaled = preprocessing.scale(X_test)

#测试gamma对SVM正确率的影响
def test_SVC_rbf(*data):
    X_train, X_test, y_train, y_test = data
    ###测试gamm###
    gamms = range(1, 50)
    train_scores = []
    test_scores = []
    for gamm in gamms:
        #gamm = gamm/10
        cls = svm.SVC(kernel='rbf', gamma=gamm)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(gamms, train_scores, label='Training score', marker='+')
    ax.plot(gamms, test_scores, label='Testing score', marker='o')
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'score')
    ax.set_ylim(0.9, 1.05)
    ax.set_title('SVC_rbf')
    ax.legend(loc='best')
    plt.show()

test_SVC_rbf(X_trainscaled,X_testscaled,y_train,y_test)

#测试参数C对高斯SVM正确率的影响
def test_rbfSVC_C(X_train,X_test,y_train,y_test ):
    Cs = np.logspace(-2,4)
    train_scores = []
    test_scores = []
    for C in Cs:
        cls = svm.SVC(kernel='rbf',C=C)
        cls.fit(X_train,y_train)
        train_scores.append(cls.score(X_train,y_train))
        test_scores.append(cls.score(X_test,y_test))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Cs,train_scores,label = 'Training score')
    ax.plot(Cs,test_scores,label = 'Testing score')
    ax.set_xlabel(r'C')
    ax.set_xscale('log')
    ax.set_ylabel(r'score')
    ax.set_title('SVC_rbf')
    ax.legend(loc='best')
    plt.show()

test_rbfSVC_C(X_trainscaled,X_testscaled,y_train,y_test)

E3Mat = np.zeros((1,2))
E5Mat = np.zeros((1,2))
for i in range(0,TrainFeature.shape[0]):  # 将E3和E5分别存入两个矩阵
    if TrainLabel[i] == 'E3': # 剔除掉偏差过大的点
        E3Mat = np.row_stack((E3Mat, X_trainscaled[i].astype(float))) # 加上一个很小的数，防止有0存在
    elif TrainLabel[i] == 'E5'and TrainFeature[i,0]<10000 and 0<TrainFeature[i,1]<20000:
        E5Mat = np.row_stack((E5Mat, X_trainscaled[i].astype(float)))

E3Mat= E3Mat[1:]  # 此时E3Mat和E5Mat第一行都是[0,0]，使用时应注意删掉
E5Mat= E5Mat[1:]

clf = svm.SVC(kernel ='rbf', gamma=2)
clf.fit(X_trainscaled,y_train)
plt.figure(1,figsize=(4,3))
plt.clf()
plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1],s=30,facecolors ='none', zorder = 10, edgecolors = 'w')
plt.axis('tight')
x_min = -3
x_max = 3
y_min = -3
y_max = 3
XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
#同样也是用来生成网格的与meshgrid类似，x_min:x_max:200j用于生成array，好像只能和mgrid连用,这点需要注意
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

Z = Z.reshape(XX.shape)
plt.figure(1,figsize=(4,3))
plt.pcolormesh(XX, YY, Z>0, cmap = plt.cm.Paired)
#pcolormesh:Plot a quadrilateral mesh.参数C may be a masked array
plt.contour(XX, YY,Z, colors = ['k','k','k'], linestyles=['--','-','--'], levels = [-.5,0,.5])
#这里画出了三条线，分别是wx+b等于-0.5，0，0.5三种
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

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
'''
param_grid = [{'C': [0.05,0.1,0.3,0.5,1,10,20,50,100], 'gamma': [0.0001,0.05,0.1,1,2.5,5,10,100]}]

def print_best_score(gsearch,param_test):
     # 输出best score
    print("Best score: %0.8f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(dict(param_test).keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print(best_parameters['gamma'])

gsearch = GridSearchCV(svm.SVC(kernel='rbf',max_iter=2000), param_grid = param_grid, scoring='roc_auc', cv=10 )  # cv为整数k时为k折交叉验证
gsearch.fit(X_trainscaled, y_train)
print_best_score(gsearch, param_grid)

#测试最优参数正确率
clf=svm.SVC(kernel='rbf',max_iter=2000)
clf.fit(X_trainscaled, y_train)  # 训练过程

print('Best Parameter CorrectRatio:%.8f' % clf.score(X_trainscaled, y_train))
'''