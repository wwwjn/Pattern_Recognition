import numpy as np
import pandas as pd
import math

priorE3=1/6 #先验概率
priorE5=5/6

#打开训练集
TenGenesSet =pd.read_table("train_data_E3E5_10genes.txt", sep=' ') #读入数据
TenGenesLabel = (TenGenesSet.values).T[10] # 筛选出每个样本的label作为Y向量
TenGenesFeature = np.asmatrix((TenGenesSet.values).T[0:10].T) # 筛选出每个样本的Feature（每个样本有两个Feature），应该是样本数*2的矩阵

# 训练样本集在假设正态分布下估计概率密度函数
# 用最大似然分别估计E3和E5的均值和方差
E3Mat = np.mat([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
E5Mat = np.mat([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

for i in range(0, TenGenesFeature.shape[0]):  # 将E3和E5分别存入两个矩阵
    if TenGenesLabel[i] == 'E3':
        E3Mat = np.row_stack((E3Mat, np.log((TenGenesFeature[i]+1e-10).astype(float)))) # 加上一个很小的数，防止有0存在
    elif TenGenesLabel[i] == 'E5':
        E5Mat = np.row_stack((E5Mat, np.log((TenGenesFeature[i]+1e-10).astype(float))))


E3Mean = np.mean(E3Mat, axis=0) #此时E3Mat和E5Mat第一行都是[0,0]，使用时应注意删掉
E5Mean = np.mean(E5Mat, axis=0)

E3Sum= np.zeros([10,10])
E5Sum= np.zeros([10,10])
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
TestSet =pd.read_table("test_data_E3E5_10genes.txt", sep=' ')
TrueLable = (TestSet.values).T[10]  # 测试集的真实类别标签
TestFeature = np.asmatrix((TestSet.values).T[0:10].T)
LikelihoodRatio = priorE5/priorE3  # E3和E5先验概率均为0.5时，似然比为1

for i in range(0,TestFeature.shape[0]):
    TestFeature[i] = np.log((TestFeature[i]+1e-10).astype(float)) # 给每一项都加上一个很小的数，防止其中有0存在不能取log

E5VarI = E5Var.astype(float).I
E3VarI = E3Var.astype(float).I
E3VarDet = np.linalg.det(E3Var.astype(float))
E5VarDet = np.linalg.det(E5Var.astype(float))
pi = 3.1415926
def probE3 (x): # 计算某样本x属于3的类条件概率，返回a为实数
    a = math.exp(-0.5*(x-E3Mean)*E3VarI*(x-E3Mean).T)/((2*pi)**5*(E3VarDet**0.5))
    return a

def probE5 (x): # 计算某样本x属于E5的类条件概率,返回a为实数
    b = math.exp(-0.5*(x-E5Mean)*E5VarI*(x-E5Mean).T)/((2*pi)**5*(E5VarDet**0.5))
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
    if Predict[i]==TrueLable[i]:
        CorrectNum= CorrectNum+1

CorrectRatio = CorrectNum/TestFeature.shape[0]  # 计算正确率
print(TestFeature.shape[0])
print(CorrectNum)
print(CorrectRatio)