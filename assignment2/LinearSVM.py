'''
一个包含SVM线性分类器和SVM高斯核函数的分类器，使用前需要先修改trainfile和testfile的路径
其中还包含了使用pyplot画出分界面和样本的代码
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import *
from sklearn.model_selection import GridSearchCV
import pandas as pd

trainfile ="train_data_E3E5_10genes.txt"  # 每次进行测试时请先修改对应的文件使其正确
testfile="test_data_E3E5_10genes.txt"

def LinearSVM(X_train,X_test,y_train,y_test):
    cls = svm.SVC(kernel='linear',class_weight=None)  # 此时惩罚系数C默认为1
    cls.fit(X_train,y_train)  # 训练过程
    print('Coefficients:%s'%(cls.coef_))
    print('Intercepr:%s'%cls.intercept_)
    print('CorrectRatio:%.8f'%cls.score(X_test,y_test))
    return cls.support_vectors_

def LinearSVM2(X_train,X_test,y_train,y_test):
    cls =svm.LinearSVC(class_weight=None)  # 此时惩罚系数C默认为1
    cls.fit(X_train,y_train)  # 训练过程
    print('Coefficients:%s'%(cls.coef_))
    print('Intercepr:%s'%cls.intercept_)
    print('CorrectRatio:%.8f'%cls.score(X_test,y_test))

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
# 用默认的C=1训练的结果
print("SVC using testset to test")
support_vectors=LinearSVM(X_trainscaled,X_testscaled,y_train,y_test)
print("SVC using trainset to test")
LinearSVM(X_trainscaled,X_trainscaled,y_train,y_train)
print("LinearSVC using trainset to test")
LinearSVM2(X_trainscaled,X_trainscaled,y_train,y_train)
print('LinearSVC using testset to test')
LinearSVM2(X_trainscaled,X_testscaled,y_train,y_test)


# 用GridSearchCV网格法找出最好的C
param_grid = [{'C': [0.05,0.1,0.3,0.5,1,10,20,50,100,1000,10000], 'kernel': ['linear']}]

def print_best_score(gsearch,param_test):
     # 输出best score
    print("Best score: %0.8f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(dict(param_test).keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

gsearch = GridSearchCV(svm.SVC(kernel='linear',max_iter=2000), param_grid = param_grid, scoring='roc_auc', cv=10 )  # cv为整数k时为k折交叉验证
gsearch.fit(X_trainscaled, y_train)
print_best_score(gsearch, param_grid)


# 探究损失函数的形式
def test_LinearSVC_loss(*data):
    X_train,X_test,y_train,y_test = data
    losses = ['hinge','squared_hinge']
    for loss in losses:
        cls = svm.LinearSVC(loss=loss)
        cls.fit(X_train,y_train)
        print('loss:%s'%loss)
        print('Coefficients:%s,intercept%s'%(cls.coef_,cls.intercept_))
        print('Score:%.8f'%cls.score(X_test,y_test))

test_LinearSVC_loss(X_trainscaled,X_testscaled,y_train,y_test)

def test_LinearSVC_L12(*data):
    X_train,X_test,y_train,y_test = data
    L12 = ['l1','l2']
    for p in L12:
        cls = svm.LinearSVC(penalty=p,dual=False)
        cls.fit(X_train,y_train)
        print('penalty:%s'%p)
        print('Coefficients:%s,intercept%s'%(cls.coef_,cls.intercept_))
        print('Score:%.8f'%cls.score(X_test,y_test))

test_LinearSVC_L12(X_trainscaled,X_testscaled,y_train,y_test)

# 考察罚项系数C的影响，绘制罚项系数C对正确率的影响
def test_LinearSVC_C(*data):
    X_train,X_test,y_train,y_test = data
    Cs = np.logspace(-1,5)
    train_scores = []
    test_scores = []
    for C in Cs:
        cls = svm.LinearSVC(C=C)
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
    ax.set_title('LinearSVC')
    ax.legend(loc='best')
    plt.show()

test_LinearSVC_C(X_trainscaled,X_testscaled,y_train,y_test)

'''
# 以下为画分类面的部分
E3Mat = np.mat([0.0,0.0])
E5Mat = np.mat([0.0,0.0])
for i in range(0,TrainFeature.shape[0]):  # 将E3和E5分别存入两个矩阵
    if TrainLabel[i] == 'E3': # 剔除掉偏差过大的点
        E3Mat = np.row_stack((E3Mat, X_trainscaled[i].astype(float))) # 加上一个很小的数，防止有0存在
    elif TrainLabel[i] == 'E5'and TrainFeature[i,0]<10000 and 0<TrainFeature[i,1]<20000:
        E5Mat = np.row_stack((E5Mat, X_trainscaled[i].astype(float)))

E3Mat= E3Mat[1:]  # 此时E3Mat和E5Mat第一行都是[0,0]，使用时应注意删掉
E5Mat= E5Mat[1:]

fig = plt.figure()
ax1=fig.add_subplot(1,1,1)
X1 = np.asarray(E3Mat.T[0])
Y1 = np.asarray(E3Mat.T[1])
X2 = np.asarray(E5Mat.T[0])
Y2 = np.asarray(E5Mat.T[1])
p1 = ax1.scatter(X1,Y1,marker = 'o', color = 'r',label='1',s=10)
p2 = ax1.scatter(X2,Y2,marker = 'o',color ='b',label='2',s=10)
plt.scatter(support_vectors[:,0],support_vectors[:,1],s=20,
               facecolors ='none', zorder = 10, edgecolors = 'k')

ax1.legend((p1, p2), ('E3', 'E5'), loc=2)
plt.xlabel("Gene1" )
plt.ylabel("Gene2")
 using trainset to 
def h(x):
    return (0.33457998*(x)+2.40369385)/3.15291934
def f(x):
    return (0.23132579*(x)+1.5602567)/2.0760136
plt.plot([-10,10], [h(-10), h(10)],label='Linear_kernel')
plt.plot([-10,10], [f(-10), f(10)],label='LinearSVC')
plt.show()
'''