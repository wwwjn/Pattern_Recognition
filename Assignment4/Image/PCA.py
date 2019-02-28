
import matplotlib.pyplot as plt
import numpy as np
from sklearn import *
from sklearn.model_selection import GridSearchCV
import pandas as pd
import copy

np.random.seed(0)

trainfile ="train_data_E3E5_10genes.txt"  # 每次进行测试时请先修改对应的文件使其正确
testfile="test_data_E3E5_10genes.txt"

#打开训练集
TrainSet =pd.read_table(trainfile, sep=' ') #读入数据
TrainLabel = (TrainSet.values).T[TrainSet.shape[1]-1] # 筛选出每个样本的label作为Y向量
TrainFeature = np.log(np.asmatrix((TrainSet.values).T[0:TrainSet.shape[1]-1].T).astype(float)+1e-6) # 筛选出每个样本的Feature（每个样本有两个Feature），应该是样本数*2的矩阵

TestSet =pd.read_table(testfile, sep=' ') #读入数据
TestLabel = (TestSet.values).T[TestSet.shape[1]-1] # 筛选出每个样本的label作为Y向量
TestFeature = np.log(np.asmatrix((TestSet.values).T[0:TestSet.shape[1]-1].T).astype(float)+1e-6) # 筛选出每个测试集样本的Feature（每个样本有两个Feature），应该是样本数*2的矩阵



# 计算均值向量
meanTrain= np.mean(TrainFeature,axis=0)
newTrainFeature=TrainFeature-meanTrain

assert meanTrain.shape!=(10,1) ,"Mean caclulate error!"

#计算协方差矩阵
covmat=np.cov(newTrainFeature,rowvar=0)
eigVals,eigVects=np.linalg.eig(np.mat(covmat))
eigValIndice=np.argsort(-eigVals) #对序号进行排序
sumAll=np.sum(eigVals)
eigValsSorted=eigVals[eigValIndice]
print(eigValsSorted)
# 主成分方差占总方差的比例
Ratio=[0]
All=np.sum(eigValsSorted)
for i in range(0,eigValsSorted.shape[0]):
    sum=np.sum(eigValsSorted[0:i+1])
    Ratio.append(sum/All)

'''
fig = plt.figure()
ax1=fig.add_subplot(1,1,1)
X1 = np.asarray(range(1,11))
Y1 = np.asarray(eigValsSorted)
p1 = ax1.scatter(X1,Y1,marker = 'o', color = 'r',label='1',s=10)
p1 = plt.plot(X1,Y1, color = 'r',label='1')

plt.xlabel("i" )
plt.ylabel("Lamda i")
plt.show()

fig = plt.figure()
ax1=fig.add_subplot(1,1,1)
X1 = np.asarray(range(0,11))
Y1 = np.asarray(Ratio)
p1 = ax1.scatter(X1,Y1,marker = 'o', color = 'r',label='1',s=10)
p1 = plt.plot(X1,Y1, color = 'r',label='1')
plt.ylim(0, 1)
plt.xlabel("i" )
plt.ylabel("The first n Principal Component/Total Variance")
plt.show()

'''
i=3 # 选前三个主成分
selected=eigValIndice[0:3]
selected2=eigValIndice[-1:-(i+1):-1]
print(selected)
print(selected2)
selectedVect=eigVects[:,selected]
selectedVals=eigVals[selected]
sum1=np.sum(selectedVals)
lowDFeature=newTrainFeature*selectedVect

w=np.matrix.tolist(lowDFeature)

#计算分为1类的时候Je为多少
Jemat=[]
Je1=0
mean1=np.matrix.tolist(np.mean(lowDFeature,axis=0))
y=np.matrix.tolist(lowDFeature)
for i in range(0, lowDFeature.shape[0]):
    for j in range(0,3):
        Je1 = Je1 + (y[i][j] - mean1[0][j]) ** 2  # 计算初始的Je
print(Je1)
Jemat.append(Je1)


Label2=[]
Label3=[]
Label4=[]
mean2=[]
mean3=[]
mean4=[]
for c in range(2,round(lowDFeature.shape[0]/3.0)): # 最多每3个点分一类
    Divided = []
    mean = []
    Label = []
    Je=0
    n = int(lowDFeature.shape[0] / c)  # 首先均匀分类
    #print(n)
    for i in range(0,c-1):
        a = np.matrix.tolist(lowDFeature[i*n:(i+1)*n,:]) #把每一类分出来
        Divided.append(a)
        for j in range(i*n,(i+1)*n):
            Label.append(i)
    Divided.append(np.matrix.tolist(lowDFeature[(c-1)*n:,:])) # 最后一类,每一类是一个矩阵
    for i in range((c-1)*n,lowDFeature.shape[0]):
        Label.append(c-1)
    for i in range(0,c):
        mean.append(np.matrix.tolist(np.mean(Divided[i],axis=0)))  #按照列均值，算出每一列均值
    for i in range(0,c):
        for j in range(0,len(Divided[i])):
            for k in range(0,3):
                Je=Je+(Divided[i][j][k]-mean[i][k])**2 #计算初始按照默认顺序的Je
    count=0
    while(count<=50):
        Jecopy=Je
        k = np.random.randint(0,lowDFeature.shape[0])  # 随机选择一个
        #print("k= "+str(k))
        if(len(Divided[Label[k]])==1):
            continue
        elif(len(Divided[Label[k]])>1):
            P=[]
            y=np.matrix.tolist(lowDFeature[k])
            for i in range(0,c): # 计算第i类的pi
                if(i!=Label[k]):
                    pi=len(Divided[i])/(len(Divided[i])+1)*((y[0][0]-mean[i][0])**2+(y[0][1]-mean[i][1])**2+(y[0][2]-mean[i][2])**2)
                    P.append(pi)
                else:
                    pj=len(Divided[i])/(len(Divided[i])-1)*((y[0][0]-mean[i][0])**2+(y[0][1]-mean[i][1])**2+(y[0][2]-mean[i][2])**2)
                    P.append(pj)
            if(min(P)==pj): #不用移动k
                count = count + 1
                continue
            else:  # 需要移动k
                z=np.matrix.tolist(lowDFeature[k])[0]
                Divided[Label[k]].remove(z) #改变k属于的类
                Label[k]=P.index(min(P)) # 修改标签
                #print("label "+str(Label[k]))
                Divided[Label[k]].append(z)
                Je=0
                #更新mean
                mean=[]
                for i in range(0, c):
                    mean.append(np.matrix.tolist(np.mean(Divided[i], axis=0)))  # 按照列均值，算出每一列均值
                for i in range(0, c):
                    for j in range(0, len(Divided[i])):
                        for k in range(0, 3):
                            Je = Je + (Divided[i][j][k] - mean[i][k]) ** 2  # 计算Je
                #print(Je)
                if(np.abs(Je-Jecopy).all()<=0.00001):
                    count=count+1
                else:
                    count=0
                    continue
    if(c==2):
        Label2=Label
        mean2=mean
    elif(c==3):
        Label3=Label
        mean3=mean
    elif(c==4):
        Label4=Label
        mean4=mean
    print(c)
    print(Je)
    Jemat.append(Je)
#print(Label2)
#print(Label3)
#print(Label4)

fig = plt.figure()
ax1=fig.add_subplot(1,1,1)
X1 = np.asarray(range(1,len(Jemat)+1))
Y1 = np.asarray(Jemat)
p1 = ax1.scatter(X1,Y1,marker = 'o', color = 'r',label='1',s=10)
p1 = plt.plot(X1,Y1, color = 'r',label='1')
plt.xlabel("c" )
plt.ylabel("Je")
plt.show()
'''
# 以下为分别画出E3和E5
E3Mat = np.zeros((1,lowDFeature.shape[1]))
E5Mat = np.zeros((1,lowDFeature.shape[1]))
trueNum=0
falseNum=0
for i in range(0,TrainFeature.shape[0]):  # 将E3和E5分别存入两个矩阵
    if (TrainLabel[i] == 'E3'):
        E3Mat = np.row_stack((E3Mat, lowDFeature[i].astype(float))) # 加上一个很小的数，防止有0存在
        if(Label2[i]==1):
            trueNum=trueNum+1
        else:
            falseNum=falseNum+1
    elif (TrainLabel[i] == 'E5'):
        E5Mat = np.row_stack((E5Mat, lowDFeature[i].astype(float)))
        if (Label2[i] == 0):
            trueNum = trueNum + 1
        else:
            falseNum = falseNum + 1

print("Correct Ratio= "+ str(trueNum/TrainFeature.shape[0]))
E3Mat= E3Mat[1:]  # 此时E3Mat和E5Mat第一行都是[0,0]，使用时应注意删掉
E5Mat= E5Mat[1:]

fig = plt.figure()
ax1=fig.add_subplot(1,1,1)
E3X1 = np.asarray(E3Mat.T[0])
E3Y1 = np.asarray(E3Mat.T[1])
E5X2 = np.asarray(E5Mat.T[0])
E5Y2 = np.asarray(E5Mat.T[1])
p1 = ax1.scatter(E3X1,E3Y1,marker = 'o', color = '#F08080',label='GroundTruth E3',s=10)
p2 = ax1.scatter(E5X2,E5Y2,marker = 'o',color ='#836FFF',label='GroundTruth E5',s=10)
plt.legend()
plt.show()


#Label2=[0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]

fig = plt.figure()
ax1=fig.add_subplot(1,1,1)
X1=[]
Y1=[]
X2=[]
Y2=[] #用来储存两个类的横纵坐标
for i in range(0,len(Label2)):
    if(Label2[i]==0):
        X1.append(w[i][0])
        Y1.append(w[i][1])
    elif(Label2[i]==1):
        X2.append(w[i][0])
        Y2.append(w[i][1])

p1=ax1.scatter(mean2[0][0],mean2[0][1],color = '#F08080',label='Class 1 Center',s=50)
p2=ax1.scatter(mean2[1][0],mean2[1][1],color = '#836FFF',label='Class 2 Center',s=50)
p3 = ax1.scatter(X1,Y1, color = 'r',label='Class 1',s=10)
p4 = ax1.scatter(X2,Y2, color = 'b',label='Class 2',s=10)
plt.legend()
plt.show()

#Label3=[0, 0, 1, 2, 0, 2, 2, 0, 0, 0, 1, 1, 0, 2, 1, 0, 1, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 2, 0, 2, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 2, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 2, 1, 1, 1, 0, 2, 0, 0, 1, 2, 1, 2, 2, 1, 2, 1, 0, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 2, 1, 1, 2, 1, 0, 2, 2, 2, 2, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 1, 1, 1, 0, 1, 1, 2, 1, 2, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 2, 2, 0, 0, 1, 0, 2, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 0, 0, 2, 2, 2, 1, 1, 1, 0, 1, 0, 1, 1, 2, 2, 1, 0, 2, 2, 1, 1, 2, 1, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 1, 1, 1, 0, 1, 1, 2, 0, 0, 1, 0, 0, 0, 0, 1, 2, 2, 1, 1, 0, 0, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 2, 0, 1, 0, 1, 0, 2, 1, 1, 0, 1, 1, 1, 2, 1, 0, 2, 0, 0, 2, 2, 0, 1, 1, 0, 2, 1, 1, 0, 1, 1, 0, 0, 2, 2, 1, 2, 2, 0, 1, 1, 2, 0, 0, 2, 1, 1, 1, 0, 1, 2, 1, 2, 0, 1, 2, 1, 1, 1, 0, 1, 1, 0, 1, 2, 0, 1, 0, 2, 2, 0, 2, 1, 0, 1, 2, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 1, 0, 1, 0, 1, 1, 1, 2, 1, 0, 2]
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
X1=[]
Y1=[]
X2=[]
Y2=[] #用来储存两个类的横纵坐标
X3=[]
Y3=[]
for i in range(0,len(Label3)):
    if(Label3[i]==0):
        X1.append(w[i][0])
        Y1.append(w[i][1])
    elif(Label3[i]==1):
        X2.append(w[i][0])
        Y2.append(w[i][1])
    elif(Label3[i]==2):
        X3.append(w[i][0])
        Y3.append(w[i][1])
p1=ax1.scatter(mean3[0][0],mean3[0][1],color = '#F08080',label='Class 1 Center',s=50)
p2=ax1.scatter(mean3[1][0],mean3[1][1],color = '#836FFF',label='Class 2 Center',s=50)
p6=ax1.scatter(mean3[2][0],mean3[2][1],color = '#9ACD32',label='Class 3 Center',s=50)
p3 = ax1.scatter(X1,Y1, color = 'r',label='Class 1',s=10)
p4 = ax1.scatter(X2,Y2, color = 'b',label='Class 2',s=10)
p5 = ax1.scatter(X3,Y3, color = 'g',label='Class 3',s=10)
plt.legend()
plt.show()

#Label4=[1, 1, 2, 3, 1, 3, 3, 1, 1, 1, 1, 0, 1, 3, 1, 1, 1, 3, 1, 3, 2, 3, 2, 3, 0, 3, 1, 2, 0, 2, 0, 0, 0, 0, 2, 2, 1, 1, 2, 1, 2, 1, 1, 0, 1, 2, 1, 0, 1, 3, 0, 3, 0, 0, 2, 1, 1, 0, 1, 1, 1, 0, 1, 2, 3, 0, 1, 2, 2, 1, 0, 2, 2, 1, 0, 3, 1, 1, 2, 1, 3, 1, 1, 1, 3, 0, 3, 0, 0, 3, 1, 1, 2, 1, 3, 1, 1, 1, 1, 0, 0, 2, 0, 0, 1, 1, 1, 3, 1, 0, 2, 1, 0, 1, 3, 1, 2, 1, 1, 1, 3, 3, 3, 3, 1, 1, 0, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 0, 2, 1, 2, 0, 2, 0, 3, 2, 3, 1, 1, 1, 1, 0, 1, 2, 2, 2, 0, 1, 2, 3, 3, 1, 1, 0, 1, 3, 1, 0, 0, 2, 1, 0, 1, 2, 2, 1, 2, 2, 1, 3, 1, 1, 2, 0, 1, 3, 3, 3, 2, 2, 1, 0, 0, 0, 0, 1, 3, 3, 1, 1, 3, 3, 2, 1, 2, 1, 2, 2, 3, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 2, 3, 3, 2, 2, 2, 0, 3, 1, 0, 1, 3, 0, 1, 2, 0, 1, 1, 0, 0, 3, 3, 0, 1, 0, 1, 3, 0, 3, 1, 2, 2, 2, 0, 1, 0, 1, 3, 0, 1, 1, 2, 1, 1, 0, 2, 1, 0, 1, 1, 2, 3, 1, 2, 1, 0, 1, 3, 1, 3, 1, 2, 2, 1, 3, 1, 1, 3, 1, 2, 2, 2, 0, 1, 2, 1, 3, 1, 1, 1, 0, 0, 1, 0, 3, 3, 2, 3, 3, 1, 1, 1, 3, 1, 0, 2, 2, 1, 2, 1, 0, 3, 2, 3, 1, 0, 2, 0, 1, 2, 1, 1, 2, 0, 1, 3, 1, 1, 1, 2, 3, 1, 3, 0, 1, 0, 3, 2, 0, 3, 2, 0, 0, 2, 3, 2, 2, 2, 1, 1, 1, 1, 2, 1, 3, 2, 0, 3]

fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
X1=[]
Y1=[]
X2=[]
Y2=[] #用来储存两个类的横纵坐标
X3=[]
Y3=[]
X4=[]
Y4=[]
for i in range(0,len(Label4)):
    if(Label4[i]==0):
        X1.append(w[i][0])
        Y1.append(w[i][1])
    elif(Label4[i]==1):
        X2.append(w[i][0])
        Y2.append(w[i][1])
    elif(Label4[i]==2):
        X3.append(w[i][0])
        Y3.append(w[i][1])
    elif (Label4[i] == 3):
        X4.append(w[i][0])
        Y4.append(w[i][1])
p1=ax1.scatter(mean4[0][0],mean4[0][1],color = '#F08080',label='Class 1 Center',s=50)
p2=ax1.scatter(mean4[1][0],mean4[1][1],color = '#836FFF',label='Class 2 Center',s=50)
p7=ax1.scatter(mean4[2][0],mean4[2][1],color = '#9ACD32',label='Class 3 Center',s=50)
p8=ax1.scatter(mean4[3][0],mean4[3][1],color = '#FFD700',label='Class 4 Center',s=50)
p3 = ax1.scatter(X1,Y1, color = 'r',label='Class 1',s=10)
p4 = ax1.scatter(X2,Y2, color = 'b',label='Class 2',s=10)
p5 = ax1.scatter(X3,Y3, color = 'g',label='Class 3',s=10)
p6 = ax1.scatter(X4,Y4, color = 'y',label='Class 4',s=10)
plt.legend()
plt.show()

#每个聚类的中心是mean
#用相同的矩阵对测试集做变换
meanTest= np.mean(TestFeature,axis=0)
newTestFeature=TestFeature-meanTest
lowDFeature2=np.matrix.tolist(newTestFeature*selectedVect)

LabelTest=[]
X1Test=[]
Y1Test=[]
X2Test=[]
Y2Test=[]

# 2分类时
for i in range(0,TestFeature.shape[0]): #欧几里得距离,采用前三个主成分
    dist1=(lowDFeature2[i][0]-mean2[0][0])**2+(lowDFeature2[i][1]-mean2[0][1])**2+(lowDFeature2[i][2]-mean2[0][2])**2
    dist2=(lowDFeature2[i][0]-mean2[1][0])**2+(lowDFeature2[i][1]-mean2[1][1])**2+(lowDFeature2[i][2]-mean2[1][2])**2
    if(dist1>dist2): #归为第2类
        LabelTest.append(1)
        X2Test.append(lowDFeature2[i][0])
        Y2Test.append(lowDFeature2[i][1])
    else:
        LabelTest.append(0)
        X1Test.append(lowDFeature2[i][0])
        Y1Test.append(lowDFeature2[i][1])

# 在2分类的图上画出结果
fig = plt.figure()
ax1=fig.add_subplot(1,1,1)
X1=[]
Y1=[]
X2=[]
Y2=[] #用来储存两个类的横纵坐标
for i in range(0,len(Label2)):
    if(Label2[i]==0):
        X1.append(w[i][0])
        Y1.append(w[i][1])
    elif(Label2[i]==1):
        X2.append(w[i][0])
        Y2.append(w[i][1])

p1=ax1.scatter(mean2[0][0],mean2[0][1],color = '#F08080',label='Class 1 Center',s=50)
p2=ax1.scatter(mean2[1][0],mean2[1][1],color = '#836FFF',label='Class 2 Center',s=50)
#p3 = ax1.scatter(X1, Y1, color = 'r',label='Class 1',s=10)
#p4 = ax1.scatter(X2, Y2, color = 'b',label='Class 2',s=10)
p5 = ax1.scatter(X1Test, Y1Test, color = '#FF00FF',label='Class 1 Test',s=10)
p6 = ax1.scatter(X2Test, Y2Test, color = '#1E90FF',label='Class 2 Test',s=10)
plt.legend()
plt.show()


# 以下为分别画出测试类的E3和E5,计算正确率

E3Mat = np.zeros((1,3))
E5Mat = np.zeros((1,3))
trueNum=0
falseNum=0
for i in range(0,TestFeature.shape[0]):  # 将E3和E5分别存入两个矩阵
    if (TestLabel[i] == 'E3'):
        E3Mat = np.row_stack((E3Mat, lowDFeature2[i])) # 加上一个很小的数，防止有0存在
        if(LabelTest[i]==1):
            trueNum=trueNum+1
        else:
            falseNum=falseNum+1
    elif (TestLabel[i] == 'E5'):
        E5Mat = np.row_stack((E5Mat, lowDFeature2[i]))
        if (LabelTest[i] == 0):
            trueNum = trueNum + 1
        else:
            falseNum = falseNum + 1
E3Mat= E3Mat[1:]  # 此时E3Mat和E5Mat第一行都是[0,0]，使用时应注意删掉
E5Mat= E5Mat[1:]

fig = plt.figure()
ax1=fig.add_subplot(1,1,1)
E3X1 = np.asarray(E3Mat.T[0])
E3Y1 = np.asarray(E3Mat.T[1])
E5X2 = np.asarray(E5Mat.T[0])
E5Y2 = np.asarray(E5Mat.T[1])
p1 = ax1.scatter(E3X1,E3Y1,marker = 'o', color = '#F08080',label='GroundTruth E3',s=10)#F08080
p2 = ax1.scatter(E5X2,E5Y2,marker = 'o',color ='#836FFF',label='GroundTruth E5',s=10)#836FFF
plt.title("Test Set")
plt.legend()
plt.show()
print("Test Correct Ratio= "+ str(trueNum/TestFeature.shape[0]))

# 三分类的结果：
LabelTest=[]
X1Test=[]
Y1Test=[]
X2Test=[]
Y2Test=[]
X3Test=[]
Y3Test=[]
for i in range(0,TestFeature.shape[0]): #欧几里得距离,采用前三个主成分
    dist1=(lowDFeature2[i][0]-mean3[0][0])**2+(lowDFeature2[i][1]-mean3[0][1])**2+(lowDFeature2[i][2]-mean3[0][2])**2
    dist2=(lowDFeature2[i][0]-mean3[1][0])**2+(lowDFeature2[i][1]-mean3[1][1])**2+(lowDFeature2[i][2]-mean3[1][2])**2
    dist3 = (lowDFeature2[i][0] - mean3[2][0]) ** 2 + (lowDFeature2[i][1] - mean3[2][1]) ** 2 + (
                lowDFeature2[i][2] - mean3[2][2]) ** 2
    if(dist1<dist2 and dist1<dist3): #归为第1类
        LabelTest.append(0)
        X1Test.append(lowDFeature2[i][0])
        Y1Test.append(lowDFeature2[i][1])
    elif(dist2<dist1 and dist2<dist3):
        LabelTest.append(1)
        X2Test.append(lowDFeature2[i][0])
        Y2Test.append(lowDFeature2[i][1])
    elif(dist3<dist1 and dist3<dist2):
        LabelTest.append(2)
        X3Test.append(lowDFeature2[i][0])
        Y3Test.append(lowDFeature2[i][1])
#Label3=[0, 0, 1, 2, 0, 2, 2, 0, 0, 0, 1, 1, 0, 2, 1, 0, 1, 2, 1, 2, 0, 2, 1, 2, 0, 2, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 2, 0, 2, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 2, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 2, 1, 1, 1, 0, 2, 0, 0, 1, 2, 1, 2, 2, 1, 2, 1, 0, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 2, 1, 1, 2, 1, 0, 2, 2, 2, 2, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 1, 1, 1, 0, 1, 1, 2, 1, 2, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 2, 2, 0, 0, 1, 0, 2, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 0, 0, 2, 2, 2, 1, 1, 1, 0, 1, 0, 1, 1, 2, 2, 1, 0, 2, 2, 1, 1, 2, 1, 1, 1, 2, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 1, 1, 1, 0, 1, 1, 2, 0, 0, 1, 0, 0, 0, 0, 1, 2, 2, 1, 1, 0, 0, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 2, 0, 1, 0, 1, 0, 2, 1, 1, 0, 1, 1, 1, 2, 1, 0, 2, 0, 0, 2, 2, 0, 1, 1, 0, 2, 1, 1, 0, 1, 1, 0, 0, 2, 2, 1, 2, 2, 0, 1, 1, 2, 0, 0, 2, 1, 1, 1, 0, 1, 2, 1, 2, 0, 1, 2, 1, 1, 1, 0, 1, 1, 0, 1, 2, 0, 1, 0, 2, 2, 0, 2, 1, 0, 1, 2, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 1, 0, 1, 0, 1, 1, 1, 2, 1, 0, 2]
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
X1=[]
Y1=[]
X2=[]
Y2=[] #用来储存两个类的横纵坐标
X3=[]
Y3=[]
for i in range(0,len(Label3)):
    if(Label3[i]==0):
        X1.append(w[i][0])
        Y1.append(w[i][1])
    elif(Label3[i]==1):
        X2.append(w[i][0])
        Y2.append(w[i][1])
    elif(Label3[i]==2):
        X3.append(w[i][0])
        Y3.append(w[i][1])
p1=ax1.scatter(mean3[0][0],mean3[0][1],color = '#F08080',label='Class 1 Center',s=50)
p2=ax1.scatter(mean3[1][0],mean3[1][1],color = '#836FFF',label='Class 2 Center',s=50)
p6=ax1.scatter(mean3[2][0],mean3[2][1],color = '#9ACD32',label='Class 3 Center',s=50)
#p3 = ax1.scatter(X1,Y1, color = 'r',label='Class 1',s=10)
#p4 = ax1.scatter(X2,Y2, color = 'b',label='Class 2',s=10)
#p5 = ax1.scatter(X3,Y3, color = 'g',label='Class 3',s=10)
p7 = ax1.scatter(X1Test,Y1Test, color = '#FF00FF',label='Class 1 Test',s=10)
p8 = ax1.scatter(X2Test,Y2Test, color = '#1E90FF',label='Class 2 Test',s=10)
p9 = ax1.scatter(X3Test,Y3Test, color = '#00FF00',label='Class 3 Test',s=10)
plt.legend()
plt.show()
'''

