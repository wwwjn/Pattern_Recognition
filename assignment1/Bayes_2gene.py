'''
2个基因的贝叶斯分类器训练，使用时需先修改开头处的先验概率，trainfile和testfile。请将testfile和trainfile放置在与本py文件同一目录下。
'''
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

priorE3 = 0.5 #先验概率
priorE5 = 0.5
LikelihoodRatio = priorE5/priorE3  # E3和E5先验概率均为0.5时，似然比为1
trainfile="train_data_E3E5_2genes.txt"
testfile="test_data_E3E5_2genes.txt"

#打开训练集
TwoGenesSet =pd.read_table(trainfile, sep=' ') #读入数据
TwoGenesLabel = (TwoGenesSet.values).T[2] # 筛选出每个样本的label作为Y向量
TwoGenesFeature = np.asmatrix((TwoGenesSet.values).T[0:2].T) # 筛选出每个样本的Feature（每个样本有两个Feature），应该是样本数*2的矩阵

# 训练样本集在假设正态分布下估计概率密度函数
# 用最大似然分别估计E3和E5的均值和方差
E3Mat = np.mat([0.0,0.0])
E5Mat = np.mat([0.0,0.0])

for i in range(0,TwoGenesFeature.shape[0]):  # 将E3和E5分别存入两个矩阵
    if TwoGenesLabel[i] == 'E3'and TwoGenesFeature[i,0]<10000 and 0<TwoGenesFeature[i,1]<20000: # 剔除掉偏差过大的点
        E3Mat = np.row_stack((E3Mat, np.log((TwoGenesFeature[i]+1e-10).astype(float)))) # 加上一个很小的数，防止有0存在
    elif TwoGenesLabel[i] == 'E5'and TwoGenesFeature[i,0]<10000 and 0<TwoGenesFeature[i,1]<20000:
        E5Mat = np.row_stack((E5Mat, np.log((TwoGenesFeature[i]+1e-10).astype(float))))

E3Mat= E3Mat[1:]  # 此时E3Mat和E5Mat第一行都是[0,0]，使用时应注意删掉
E5Mat= E5Mat[1:]

E3Mean = np.mean(E3Mat, axis=0)
E5Mean = np.mean(E5Mat, axis=0)

E3Sum = np.mat([[0.0, 0.0], [0.0, 0.0]])
E5Sum = np.mat([[0.0, 0.0], [0.0, 0.0]])
for i in range(0,E3Mat.shape[0]):
    c = ((E3Mat[i]- E3Mean).T) *(E3Mat[i]-E3Mean)
    E3Sum = E3Sum + c
E3Var = E3Sum/(E3Mat.shape[0])
print(E3Var)
print(E3Mean)

for i in range(0, E5Mat.shape[0]):
    c = ((E5Mat[i]- E5Mean).T) *(E5Mat[i]-E5Mean)
    E5Sum = E5Sum + c
E5Var = E5Sum/(E5Mat.shape[0])
print(E5Var)
print(E5Mean)

# 打开测试集
TestSet =pd.read_table(testfile, sep=' ')
TrueLabel = (TestSet.values).T[2]  # 测试集的真实类别标签
TestFeature = np.asmatrix((TestSet.values).T[0:2].T)


for i in range(0,TestFeature.shape[0]):
    TestFeature[i] = np.log((TestFeature[i]+1e-10).astype(float)) # 给每一项都加上一个很小的数，防止其中有0存在不能取log

E5VarI = E5Var.astype(float).I
E3VarI = E3Var.astype(float).I
E3VarDet = np.linalg.det(E3Var.astype(float))
E5VarDet = np.linalg.det(E5Var.astype(float))
pi = 3.1415926
def probE3 (x): # 计算某样本x属于3的类条件概率，返回a为实数
    a = math.exp(-0.5*(x-E3Mean)*E3VarI*(x-E3Mean).T)/(2*pi*(E3VarDet**0.5))
    return a

def probE5 (x): # 计算某样本x属于E5的类条件概率,返回a为实数
    b = math.exp(-0.5*(x-E5Mean)*E5VarI*(x-E5Mean).T)/(2*pi*(E5VarDet**0.5))
    return b

def posteriorE3(x): # 计算样本x的属于E3类的后验概率
    c = probE3(x)*priorE3/(probE3(x)*priorE3+probE5(x)*priorE5)
    return c

def posteriorE5(x): # 计算样本x的属于E5类的后验概率
    d = probE5(x)*priorE5/(probE3(x)*priorE3+probE5(x)*priorE5)
    return d

Predict = []
CorrectNum=0
for i in range(0, TestFeature.shape[0]): # 对测试集运用贝叶斯分类，得到预测结果
    l = probE3(TestFeature[i])/probE5(TestFeature[i])
    if l > LikelihoodRatio:
        Predict.append('E3')
    else :
        Predict.append('E5')
for i in range(0, TestFeature.shape[0]):
    if Predict[i]==TrueLabel[i]:
        CorrectNum= CorrectNum+1

CorrectRatio = CorrectNum/TestFeature.shape[0]  # 计算正确率
print(CorrectRatio)


'''
# 画图部分
fig = plt.figure()
ax1=fig.add_subplot(1,1,1)
plt.axis([-2, 10, -2, 12])
X1 = np.asarray(E3Mat.T[0])
Y1 = np.asarray(E3Mat.T[1])
X2 = np.asarray(E5Mat.T[0])
Y2 = np.asarray(E5Mat.T[1])
p1 = ax1.scatter(X1,Y1,marker = 'o', color = 'r',label='1',s=10)
p2 = ax1.scatter(X2,Y2,marker = 'o',color ='b',label='2',s=10)
ax1.legend((p1, p2), ('E3', 'E5'), loc=2)
plt.xlabel("Gene1" )
plt.ylabel("Gene2")

def f(x,y):  # 先验概率分别为1/6、5/6时的贝叶斯分类图线
    return np.exp(-0.5*(((x-5.52905751)*1.11419979+(y-8.58936641)*(-0.76240309))*(x-5.52905751)+(y-8.58936641)*((x-5.52905751)*(-0.76240309)+(y-8.58936641)*1.75739784)))/(2*pi*(E3VarDet**0.5))-5*np.exp(-0.5*(((y-5.58025939)* 0.13958308+(y-4.81857411)*(-0.09365703))*(x-5.58025939)+(y-4.81857411)*((x-5.58025939)*(-0.09365703)+(y-4.81857411)*0.51199497)))/(2*pi*(E3VarDet**0.5))
x = np.linspace(-2, 10, 256)
y = np.linspace(-2, 12, 256)
X,Y = np.meshgrid(x, y)
plt.contourf(X, Y, f(X, Y), 0, alpha = 0,cmap = plt.cm.hot)
C = plt.contour(X, Y, f(X,Y), 0, colors = 'green', linewidth = 0.5)

def f2(x,y):  # 先验概率分别为0.5的贝叶斯分类图线
    return np.exp(-0.5*(((x-5.52905751)*1.11419979+(y-8.58936641)*(-0.76240309))*(x-5.52905751)+(y-8.58936641)*((x-5.52905751)*(-0.76240309)+(y-8.58936641)*1.75739784)))/(2*pi*(E3VarDet**0.5))-np.exp(-0.5*(((y-5.58025939)* 0.13958308+(y-4.81857411)*(-0.09365703))*(x-5.58025939)+(y-4.81857411)*((x-5.58025939)*(-0.09365703)+(y-4.81857411)*0.51199497)))/(2*pi*(E3VarDet**0.5))

plt.contourf(X, Y, f2(X, Y), 0, alpha = 0,cmap = plt.cm.hot)
C2 = plt.contour(X, Y, f2(X,Y), 0, colors = 'yellow', linewidth = 0.5)

def g(x):
    return (0.00120384*(x)+0.0195)/0.00349472
plt.plot([-2,15], [g(-2), g(15)])

def h(x):
    return (1.05967488*(x)+9.5)/2.09698244
plt.plot([-2,15], [h(-2), h(15)])
plt.show()
'''




