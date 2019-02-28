'''
2个基因和10个基因的Fisher分类器实验，假设样本近似正态分布，用样本的算术平均作为均值的估计，把样本协方差矩阵当做是真实协方差矩阵的估计，确定阈值Wo
两种测试方法：1.用样本集或训练集测试  2.10折交叉验证
'''
import numpy as np
import pandas as pd
import math
import random

wo=-0.0195  # 阈值通过尝试得出

def OpenFile(file):
    Set = pd.read_table(file, sep=' ')  # 读入数据
    Label = (Set.values).T[Set.values.shape[1] - 1] # 筛选出每个样本的label作为Y向量
    Feature = np.asmatrix((Set.values).T[0:Set.values.shape[1] - 1].T).astype(float)  # 筛选出每个样本的Feature（每个样本有两个Feature），应该是样本数*2的矩阵
    return Feature,Label

def Train(TrainFeature,TrainLabel):
    for i in range(0, TrainFeature.shape[0]):
        TrainFeature[i] = np.log((TrainFeature[i] + 1e-6).astype(float))  # 给每一项都加上一个很小的数，防止其中有0存在不能取log
    E3Mat = np.zeros([1,TrainFeature.shape[1]])
    E5Mat = np.zeros([1,TrainFeature.shape[1]])

    for i in range(0,TrainFeature.shape[0]):  # 将E3和E5分别存入两个矩阵
        if TrainLabel[i] == 'E3':
            E3Mat = np.row_stack((E3Mat, TrainFeature[i]))  # 加上一个很小的数，防止有0存在
        elif TrainLabel[i] == 'E5':
            E5Mat = np.row_stack((E5Mat, TrainFeature[i]))

    E3Mat= E3Mat[1:]  # 此时E3Mat和E5Mat第一行都是[0,0]，使用时应注意删掉
    E5Mat= E5Mat[1:]

    E3Mean = np.asmatrix(np.mean(E3Mat, axis=0))  # 计算类均值向量
    E5Mean = np.asmatrix(np.mean(E5Mat, axis=0))

    E3Sum = np.zeros([TrainFeature.shape[1],TrainFeature.shape[1]])
    E5Sum = np.zeros([TrainFeature.shape[1],TrainFeature.shape[1]])
    for i in range(0,E3Mat.shape[0]):  # 计算E3的类内离散度矩阵
        c = (E3Mat[i] - E3Mean).T * (E3Mat[i]-E3Mean)
        E3Sum = E3Sum + c
    for i in range(0,E5Mat.shape[0]):  # 计算E5的类内离散度矩阵
        c = (E5Mat[i] - E5Mean).T * (E5Mat[i]-E5Mean)
        E5Sum = E5Sum + c
    S_w = np.asmatrix(E3Sum + E5Sum).astype(float) # 总类内离散度矩阵
    S_b = (E3Mean - E5Mean).T * (E3Mean - E5Mean)  # 类间离散度矩阵
    w = S_w.I * (E3Mean - E5Mean).T
    return w  # 返回一个元组

def Test(TestFeature, TestLabel, w):
    TestFeature2 = np.zeros([TestFeature.shape[0],TestFeature.shape[1]])
    for i in range(0, TestFeature.shape[0]):
        TestFeature2[i] = np.log(np.asmatrix((TestFeature[i] + 1e-6)).astype(float))  # 给每一项都加上一个很小的数，防止其中有0存在不能取log
    def discrim(x):  # 判别函数
        if (w.T * x.T) + wo > 0:
            return 'E3'
        elif (w.T * x.T) + wo <= 0:
            return 'E5'

    Predict=[]
    for i in range(0, TestFeature.shape[0]):
        Predict.append(discrim(np.asmatrix(TestFeature2[i])))
    CorrectNum = 0
    for i in range(0, TestLabel.shape[0]):
        if Predict[i] == TestLabel[i]:
            CorrectNum = CorrectNum+1
        #else:
             #print(i)
    CorrectRatio = CorrectNum/TestFeature.shape[0]  # 计算正确率
    return CorrectRatio

# 在2个基因的训练集上做训练，并在训练集上测试
TrainSet1 = OpenFile('train_data_E3E5_2genes.txt')
TestSet1 = OpenFile('train_data_E3E5_2genes.txt')
TrainResult1 = Train(TrainSet1[0],TrainSet1[1])
print(Test(TestSet1[0],TestSet1[1],TrainResult1))

# 在2个基因训练集上做训练，并在测试集上测试
TrainSet2 = OpenFile('train_data_E3E5_2genes.txt')
TestSet2 = OpenFile('test_data_E3E5_2genes.txt')
TrainResult2 = Train(TrainSet2[0],TrainSet2[1])
print(Test(TestSet2[0],TestSet2[1],TrainResult2))

# 在10个基因的训练集上做训练，并在训练集上测试
TrainSet3 = OpenFile('train_data_E3E5_10genes.txt')
TestSet3 = OpenFile('train_data_E3E5_10genes.txt')
TrainResult3 = Train(TrainSet3[0],TrainSet3[1])
print(Test(TestSet3[0],TestSet3[1],TrainResult3))

# 在10个基因训练集上做训练，并在测试集上测试
TrainSet4 = OpenFile('train_data_E3E5_10genes.txt')
TestSet4 = OpenFile('test_data_E3E5_10genes.txt')
TrainResult4 = Train(TrainSet4[0],TrainSet4[1])
print(Test(TestSet4[0],TestSet4[1],TrainResult4))

def TenFoldValidation(file):
    Set = pd.read_table(file, sep=' ')  # 读入数据
    Label = ((Set.values).T[Set.values.shape[1] - 1] ) # 筛选出每个样本的label作为Y向量 n行1列
    Feature =np.asmatrix((Set.values).T[0:Set.values.shape[1] - 1].T) # 筛选出每个样本的Feature（每个样本有两个Feature），应该是样本数*2的矩阵
    list=[] # 十折交叉验证
    for i in range(Feature.shape[0]):
        list.append(i)
    random.shuffle(list)  # 生成一波随机数，作为10折交叉验证的下标
    k = math.ceil(Feature.shape[0]/10)-1  # 此处使用向下取整,最后一组的数量稍微大些
    MiniSet = [] # 应该此处设置合适大小
    MiniFeature = []
    MiniLabel = []  # 把Set划分为10个MiniSet
    for j in range(0,9):
        MiniSet.append((Set.values)[list[j*k:j*k+k]])
        MiniLabel.append((MiniSet[j].T)[(Set.values).shape[1] -1].T)
        MiniFeature.append(np.asmatrix(MiniSet[j].T[0:Set.values.shape[1] - 1].T))
    MiniSet.append(Set.values[list[9*k:Feature.shape[0]]])   # 最后一个MiniSet
    MiniLabel.append(np.asmatrix(MiniSet[9].T[Set.values.shape[1] - 1]) .T ) # MiniLabel 应该是一个数组,里面的元素都是
    MiniFeature.append(np.asmatrix(MiniSet[9].T[0:Set.values.shape[1] - 1]).T)
    Ratio = [0,0,0,0,0,0,0,0,0,0]
    for m in range(0,10):
        TrainFeature = np.asmatrix(np.zeros([1,Feature.shape[1]]))
        TrainLabel = np.zeros([1,1])
        for n in range(0,10):
            if n != m:
                TrainFeature = np.row_stack((TrainFeature, MiniFeature[n]))  # 注意深复制和浅复制的区别
                TrainLabel = np.append(TrainLabel, MiniLabel[n])
        TrainFeature= TrainFeature[1:]
        a=TrainFeature
        TrainLabel= TrainLabel[1:]
        b=TrainLabel
        TrainResult = Train(a, b)
        c=MiniFeature[m]
        d=MiniLabel[m]
        Ratio[m] = Test(c, d, TrainResult)   # 在TenFold中不应该再取对数，因为Test已经取了对数
    print(Ratio)
    return np.mean(Ratio)

# 分别对2个基因和10个基因进行Fisher分类，并用10折交叉验证法验证
TenFoldValidation('train_data_E3E5_2genes.txt')
TenFoldValidation('train_data_E3E5_10genes.txt')

TenFold=[]
for i in range(0,10):
    TenFold.append(TenFoldValidation('train_data_E3E5_2genes.txt'))
print(np.mean(np.asarray(TenFold)))

TenFold2=[]
for i in range(0,10):
    TenFold2.append(TenFoldValidation('train_data_E3E5_10genes.txt'))
print(np.mean(np.asarray(TenFold2)))

