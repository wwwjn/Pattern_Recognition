'''
10个基因的感知器实验
'''
import numpy as np
import pandas as pd
import math
import random

def OpenFile(file):
    Set = pd.read_table(file, sep=' ')  # 读入数据
    Label = (Set.values).T[Set.values.shape[1] - 1]  # 筛选出每个样本的label作为向量
    Feature = np.asmatrix((Set.values).T[0:Set.values.shape[1] - 1]).T  # 筛选出每个样本的Feature（每个样本有两个Feature），应该是样本数*2的矩阵
    return Feature,Label

def Test(Feature, Label, a):
    b = np.zeros([Feature.shape[0], 1])
    for i in range(0, Feature.shape[0]):
        b[i] = 1

    Feature2 = np.column_stack((b,np.log((Feature+1e-6).astype(float))))

    Predict=[]
    for i in range(0, Feature.shape[0]):
        if a * np.asmatrix(Feature2[i]).T > 0:
            Predict.append('E3')
        elif a * np.asmatrix(Feature2[i]).T <= 0:
            Predict.append('E5')
    CorrectNum = 0
    for i in range(0, Feature.shape[0]):
        if Predict[i] == Label[i]:
            CorrectNum = CorrectNum + 1
        #else:
            #print(i)

    CorrectRatio = CorrectNum/Feature.shape[0]  # 计算正确率
    #print(CorrectRatio)
    return CorrectRatio

def Perceptron(Feature, Label):
    E3Mat = np.zeros([1,Feature.shape[1]])
    E5Mat = np.zeros([1,Feature.shape[1]])

    for i in range(0,Feature.shape[0]):  # 将E3和E5分别存入两个矩阵
        if Label[i] == 'E3':
            E3Mat = np.row_stack((E3Mat, np.log((Feature[i]+1e-6).astype(float))))  # 加上一个很小的数，防止有0存在
        elif Label[i] == 'E5':
            E5Mat = np.row_stack((E5Mat, np.log((Feature[i]+1e-6).astype(float))))  # E

    E3Mat= E3Mat[1:]  # 此时E3Mat和E5Mat第一行都是[0,0]，使用时应注意删掉
    E5Mat= E5Mat[1:]


    Add3 = np.zeros([E3Mat.shape[0],1])
    for i in range(0, E3Mat.shape[0]):
        Add3[i] = 1
    Add5 = np.zeros([E5Mat.shape[0],1])
    for i in range(0, E5Mat.shape[0]):
        Add5[i] = 1

    yE3 = np.column_stack((Add3, E3Mat))  # 增广之后的样本向量
    yE5 = np.column_stack((Add5,  E5Mat))
    y = np.row_stack((yE3, - yE5))  # 把E5类的Feature变成负的
    #MaxIterationTime = 1000 # 设置最大迭代次数上限
    #learningRate = 0.1
    a = np.zeros([1, Feature.shape[1] + 1])  # 増广后的权重矩阵a,初始时候a的每个元素都是0
    #print(a)
    #print(y.shape[0])
    #for k in range(0, MaxIterationTime):  #  单步更新
    k=0

    while True:
        h = k % y.shape[0]
        learningRate = 0.8
        while a * y[h].T <= 0:
            b = y[h] * learningRate
            #print(b)
            a = a + b
            #print("change a to" , a)
            learningRate = learningRate *0.95 # 步长缩减

        tag = 0  # 检查此时的a是否对于每个yi都有a.T*yi>0
        for j in range(0, Feature.shape[0]):
            if a*y[j].T > 0:
                tag = tag + 1
            else:
                tag = 0
            j+=1

        if tag == y.shape[0]:
            Test(Feature,Label,a)  # 观察a迭代过程中的正确率变化
            print(1)
            break  # 如果对所有的y[i]，a*yi>0都成立，就退出循环

        if b.all() < 0.001 :  # 退出条件1，如果两次迭代之间，每个元素变化很小，也退出
            #print("Condition 2")
            break
        elif Test(Feature,Label,a)>0.95:  #退出条件2，预测正确率大于一个值
            #print('OK')
            break
        else:
            k += 1
            continue
    return a


def TenFoldValidation(file):
    Set = pd.read_table(file, sep=' ')  # 读入数据
    Label = np.asmatrix((Set.values).T[Set.values.shape[1] - 1] ) # 筛选出每个样本的label作为Y向量 n行1列
    Feature =np.asmatrix((Set.values).T[0:Set.values.shape[1] - 1].T)  # 筛选出每个样本的Feature（每个样本有两个Feature），应该是样本数*2的矩阵
    list=[] # 十折交叉验证
    for i in range(Feature.shape[0]):
        list.append(i)
    random.shuffle(list)  # 生成一波随机数，作为10折交叉验证的下标
    k = math.ceil(Feature.shape[0]/10)-1  # 此处使用向下取整,最后一组的数量稍微大些
    MiniSet = [0,0,0,0,0,0,0,0,0,0] # 应该此处设置合适大小
    MiniFeature = [0,0,0,0,0,0,0,0,0,0]
    MiniLabel = [0,0,0,0,0,0,0,0,0,0]  # 把Set划分为10个MiniSet
    for j in range(0,9):
        MiniSet[j] = Set.values[j*k:j*k+k]
        MiniLabel[j] = np.asmatrix(MiniSet[j].T[Set.values.shape[1] - 1]).T
        MiniFeature[j] = np.asmatrix(MiniSet[j].T[0:Set.values.shape[1] - 1]).T
    MiniSet[9] = Set.values[9*k:Feature.shape[0]]  # 最后一个MiniSet
    MiniLabel[9] = np.asmatrix(MiniSet[9].T[Set.values.shape[1] - 1]) .T # MiniLabel 应该是一个数组,里面的元素都是
    MiniFeature[9] = np.asmatrix(MiniSet[9].T[0:Set.values.shape[1] - 1]).T
    Ratio=[0,0,0,0,0,0,0,0,0,0]

    for m in range(0,10):
        TrainFeature = np.asmatrix(np.zeros([1,Feature.shape[1]]))
        TrainLabel = np.mat([0])
        for n in range(0,10):
            if n != m:
                TrainFeature = np.vstack((TrainFeature, MiniFeature[n]))
                TrainLabel = np.vstack((TrainLabel, MiniLabel[n]))
        TrainFeature= TrainFeature[1:]  # 去掉第一行的0
        TrainLabel= TrainLabel[1:]
        a = Perceptron(TrainFeature, TrainLabel)
        Ratio[m] = Test(MiniFeature[m], MiniLabel[m], a)
    print(Ratio)
    print(np.mean(Ratio))

TrainSet3 = OpenFile('train_data_E3E5_10genes.txt')
TestSet3 = OpenFile('train_data_E3E5_10genes.txt')
Train3 = Perceptron(TrainSet3[0], TrainSet3[1])
print(Test(TestSet3[0], TestSet3[1], Train3))

TrainSet4 = OpenFile('train_data_E3E5_10genes.txt')
TestSet4 = OpenFile('test_data_E3E5_10genes.txt')
Train4 = Perceptron(TrainSet4[0], TrainSet4[1])
print(Test(TestSet4[0], TestSet4[1], Train4))

TenFoldValidation("train_data_E3E5_10genes.txt")

TenFold=[]
for i in range(0,10):
    TenFold.append(TenFoldValidation('train_data_E3E5_10genes.txt'))
print(np.mean(np.asarray(TenFold)))









