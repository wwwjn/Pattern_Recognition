# 穷举和遗传,从十个基因中选择出3个影响最大的基因。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import *
import random
import copy
import math
trainfile ="train_data_E3E5_10genes.txt"  # 每次进行测试时请先修改对应的文件使其正确
testfile="test_data_E3E5_10genes.txt"
random.seed(0)

#打开训练集
TrainSet =pd.read_table(trainfile, sep=' ') #读入数据
TrainLabel = (TrainSet.values).T[TrainSet.shape[1]-1] # 筛选出每个样本的label作为Y向量
TrainFeature = np.log(np.asmatrix((TrainSet.values).T[0:TrainSet.shape[1]-1].T).astype(float)+1e-6) # 筛选出每个样本的Feature（每个样本有两个Feature），应该是样本数*2的矩阵


TestSet =pd.read_table(testfile, sep=' ') #读入数据
TestLabel = (TestSet.values).T[TestSet.shape[1]-1] # 筛选出每个样本的label作为Y向量
TestFeature = np.log(np.asmatrix((TestSet.values).T[0:TestSet.shape[1]-1].T).astype(float)+1e-6)

E3Mat = np.zeros([1, TrainFeature.shape[1]]) # 将E3类和E5类的基因分开
E5Mat = np.zeros([1, TrainFeature.shape[1]])

for i in range(0, TrainFeature.shape[0]):  # 将E3和E5分别存入两个矩阵
    if TrainLabel[i] == 'E3':
        E3Mat = np.row_stack((E3Mat, TrainFeature[i]))  # 加上一个很小的数，防止有0存在
    elif TrainLabel[i] == 'E5':
        E5Mat = np.row_stack((E5Mat, TrainFeature[i]))

E3Mat = E3Mat[1:]  # 此时E3Mat和E5Mat第一行都是[0,0]，使用时应注意删掉
E5Mat = E5Mat[1:]

PE3=E3Mat.shape[0]/TrainFeature.shape[0]  #两类的先验概率
PE5=E5Mat.shape[0]/TrainFeature.shape[0]


def SVMTest(b):  # b是一个数组，对选择得到的结果进行测试,b应该是一个3维数组
    TestMat = np.zeros([TestFeature.shape[0], 1])
    TrainMat = np.zeros([TrainFeature.shape[0], 1])
    for i in b:
        TestMat = np.column_stack((TestMat, TestFeature[:,i]))
        TrainMat = np.column_stack((TrainMat, TrainFeature[:,i]))
    TestMat = TestMat[:, 1:]
    TrainMat = TrainMat[:, 1:]
    cls = svm.SVC(kernel="linear")
    cls.fit(TrainMat, TrainLabel)
    print("Score of SVM: " + str(cls.score(TestMat, TestLabel)))
    print("Coef of SVM："+str(cls.coef_))
    print("Intercept of SVM " + str(cls.intercept_))

# 方法1：枚举+过滤法
def Discrim(x1,x2): #x1和x2分别是E3细胞和E5细胞对应的矩阵(行数为元素个数，列数为特征个数),计算类内离散度和类间离散度
    x1mean=np.mean(x1,axis=0)
    x2mean=np.mean(x2,axis=0)
    x1Sum = np.zeros([x1.shape[1],x1.shape[1]])
    x2Sum = np.zeros([x2.shape[1],x2.shape[1]])
    xmean=np.mean(np.row_stack((x1,x2)))
    for i in range(0,x1.shape[0]):  # 计算E3的类内离散度矩阵
        c = (x1[i] - x1mean).T * (x1[i]-x1mean)
        x1Sum = x1Sum + c
    for i in range(0,x2.shape[0]):  # 计算E5的类内离散度矩阵
        c = (x2[i] - x2mean).T * (x2[i]-x2mean)
        x2Sum = x2Sum + c
    S_w = PE3*x1Sum/x1.shape[0] + PE5*x2Sum/x2.shape[0] # 总类内离散度矩阵

    S_b = PE3*(x1mean - xmean).T * (x1mean - xmean) + PE5*(x2mean - xmean).T * (x2mean - xmean)  # 类间离散度矩阵

    return np.trace(S_b)/np.trace(S_w)


# 穷举法,选出第i，j，k个基因，过滤法，求判据最优的组合
maxDiscrim=Discrim(E3Mat.T[[0,1,2]].T,E5Mat.T[[0,1,2]].T) #注意此时的切片操作，需要传入一个数组，而不是传入3个参数
b=[0,1,2]
for i in range(0,10):
    for j in range(i+1,10):
        for k in range(j+1,10):
            a=Discrim(E3Mat.T[[i,j,k]].T,E5Mat.T[[i,j,k]].T)
            if a>maxDiscrim:
                maxDiscrim=a
                b=[i,j,k]
print("Result of criterion: "+str(b))
SVMTest(b)

# 方法2：包裹法枚举
def LinearSVM(X_train,y_train,X_test,y_test):
    #cls = svm.SVC(kernel='linear',class_weight={'E3':PE3,'E5':PE5})  # 此时惩罚系数C默认为1
    cls = svm.SVC(kernel='linear')
    cls.fit(X_train,y_train)  # 训练过程
    return cls.score(X_test,y_test)

X_train=TrainFeature
X_test=TestFeature
y_train=TrainLabel
y_test=TestLabel
X_trainscaled = X_train
X_testscaled = X_test
#X_trainscaled = preprocessing.scale(X_train)
#X_testscaled = preprocessing.scale(X_test)

maxC=LinearSVM(X_trainscaled.T[[0,1,2]].T,y_train,X_testscaled.T[[0,1,2]].T,y_test)
c=[0,1,2]
for i in range(0,10):
    for j in range(i+1,10):
        for k in range(j+1,10):
            a=LinearSVM(X_trainscaled.T[[i,j,k]].T , y_train ,X_trainscaled.T[[i,j,k]].T,y_train)
            if a>maxC:
                maxC=a
                c=[i,j,k]
print("Result of wrapper: "+str(c))
SVMTest(c)


# 方法3：遗传算法

lst=[0,0,0,0,0,0,0,1,1,1]
PopulationNum=20  #初始种群中染色体个数
ExchangeRate=0.3
MutationRate=0.3
MaxOperation=50
#M=[]
Population=[]  # Population维护的是当前种群的所有个体
MaxIteration=300

for i in range(0,PopulationNum):  #创建初始的群落,具有10个染色体
    random.shuffle(lst)
    flag=True
    for j in range(0,len(Population)):
        if(Population[j]==lst):
            flag=False
    if(flag):
        Population.append(copy.deepcopy(lst))
#M.append(copy.deepcopy(a))

def fit(list1):  #适应度函数,使用可分性判据
    E3TrainMat=np.zeros((E3Mat.shape[0])).T
    E5TrainMat=np.zeros((E5Mat.shape[0])).T
    for i in range(0,len(list1)):
        if list1[i]==1:
            E3TrainMat=np.column_stack((E3TrainMat,E3Mat.T[i].T))
            E5TrainMat=np.column_stack((E5TrainMat,E5Mat.T[i].T))
    E3TrainMat=E3TrainMat.T[1:].T
    E5TrainMat = E5TrainMat.T[1:].T
    return Discrim(E3TrainMat,E5TrainMat)

#对选中的种群进行操作
def exchange(x1,x2): #对x1和x2进行随机交换
    b=random.randrange(0,10)
    c=copy.deepcopy(x1[b:10])
    d = copy.deepcopy(x2[b:10])
    e = copy.deepcopy(x1[0:b])+d

    f=copy.deepcopy(x2[0:b])+c
    count1=0
    for i in range(0, 10):
        if(e[i]==1):
            count1=count1+1
    count2=0
    for i in range(0, 10):
        if(f[i]==1):
            count2=count1+1
    if (count1==3):
        choosen.append(e)
    if (count2==3):
        choosen.append(f)

def mutation(x):  # 对x的某个基因进行随机突变,为了保证突变后还是选中3个基因，同时对两个基因进行突变
    e=random.randrange(0,10)
    f=random.randrange(0,10)
    g=copy.deepcopy(x)
    if(x[e]==1 and x[f]==0):
        g[e]=0
        g[f]=1
    elif (x[e] == 0 and x[f] == 1):
        g[e] = 1
        g[f] = 0
    choosen.append(g)

Iteration=0

while(Iteration<=MaxIteration and len(Population)>=2):
    fitness=[]
    prob=[]
    for i in range(0,len(Population)):
        fitness.append(fit(Population[i]))
    Sumfit=sum(fitness)
    # 以一定概率选中某一个？转盘法
    choosen=[] # 被选中的集合
    for i in range(0,len(Population)):  # 每一个个体被选中的概率
        r=random.random() # 产生0-1之间随机数，
        if(r<=(len(Population)*0.8*fitness[i]/Sumfit)):
            choosen.append(Population[i])
    a=random.randint(0,MaxOperation)
    for i in range(0,a):
        type=random.randint(0,1)
        right=random.random()
        if(type==0):
            if(right<=MutationRate):
                g = random.randint(0, len(choosen)-1)
                mutation(choosen[g])
        if(type==1):
            if(right<=ExchangeRate):
                g=random.randint(0,len(choosen)-1)
                h = random.randint(0, len(choosen)-1)
                exchange(choosen[g],choosen[h])
    Population=choosen
    Iteration=Iteration+1

maxf=1
Result=[]
f=[]
for i in range(0, len(Population)):
    j=fit(Population[i])
    if(j>maxf):
        maxf=j
        Result=Population[i]
for i in range(0,10):
    if Result[i]!=0:
        f.append(i)
print("Result of Genetic Algorithem: "+str(f))
SVMTest(f)

#用其他线性分类器来测试正确率


