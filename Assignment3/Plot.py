# 穷举和遗传,从十个基因中选择出3个影响最大的基因。

import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import *

trainfile ="train_data_E3E5_10genes.txt"  # 每次进行测试时请先修改对应的文件使其正确
testfile="test_data_E3E5_10genes.txt"


#打开训练集
TrainSet =pd.read_table(trainfile, sep=' ') #读入数据
TrainLabel = (TrainSet.values).T[TrainSet.shape[1]-1] # 筛选出每个样本的label作为Y向量
TrainFeature = np.asmatrix((TrainSet.values).T[0:TrainSet.shape[1]-1].T).astype(float) # 筛选出每个样本的Feature（每个样本有两个Feature），应该是样本数*2的矩阵


TestSet =pd.read_table(testfile, sep=' ') #读入数据
TestLabel = (TestSet.values).T[TestSet.shape[1]-1] # 筛选出每个样本的label作为Y向量
TestFeature = np.asmatrix((TestSet.values).T[0:TestSet.shape[1]-1].T).astype(float)


E3Mat = np.zeros([1, TrainFeature.shape[1]]) # 将E3类和E5类的基因分开
E5Mat = np.zeros([1, TrainFeature.shape[1]])

for i in range(0, TrainFeature.shape[0]):  # 将E3和E5分别存入两个矩阵
    if TrainLabel[i] == 'E3':
        E3Mat = np.row_stack((E3Mat, TrainFeature[i]))  # 加上一个很小的数，防止有0存在
    elif TrainLabel[i] == 'E5':
        E5Mat = np.row_stack((E5Mat, TrainFeature[i]))

E3Mat = E3Mat[1:]  # 此时E3Mat和E5Mat第一行都是[0,0]，使用时应注意删掉
E5Mat = E5Mat[1:]
'''
E3Mat = np.zeros([1, TestFeature.shape[1]]) # 将E3类和E5类的基因分开
E5Mat = np.zeros([1, TestFeature.shape[1]])

for i in range(0, TestFeature.shape[0]):  # 将E3和E5分别存入两个矩阵
    if TestLabel[i] == 'E3':
        E3Mat = np.row_stack((E3Mat, TestFeature[i]))  # 加上一个很小的数，防止有0存在
    elif TestLabel[i] == 'E5':
        E5Mat = np.row_stack((E5Mat, TestFeature[i]))

E3Mat = E3Mat[1:]  # 此时E3Mat和E5Mat第一行都是[0,0]，使用时应注意删掉
E5Mat = E5Mat[1:]

#画图部分
countE3=np.array([0,0,0,0,0,0,0,0,0,0])
for i in range(0,E3Mat1.shape[1]):
    for h in range(0,E3Mat1.shape[0]):
        if(E3Mat1[h,i]!=[0]):
            countE3[i]=countE3[i]+1
print(countE3)
countE5=np.array([0,0,0,0,0,0,0,0,0,0])
for i in range(0,E5Mat1.shape[1]):
    for h in range(0,E5Mat1.shape[0]):
        if(E5Mat1[h,i]!=[0]):
            countE5[i]=countE5[i]+1
print(countE5)


# 必须配置中文字体，否则会显示成方块
# 注意所有希望图表显示的中文必须为unicode格式


font_size = 10  # 字体大小
fig_size = (8, 6)  # 图表大小

names = [u'E3', u'E5']  # 姓名
subjects = [1,2,3,4,5,6,7,8,9,10]  # 科目
scores = [countE3, countE5]  # 成绩

# 更新字体大小
mpl.rcParams['font.size'] = font_size
# 更新图表大小
mpl.rcParams['figure.figsize'] = fig_size
# 设置柱形图宽度
bar_width = 0.3

index = np.arange(len(scores[0]))
rects1 = plt.bar(index, scores[0], bar_width, color='#0072BC', label=names[0])
rects2 = plt.bar(index + bar_width, scores[1], bar_width, color='#ED1C24', label=names[1])
# X轴标题
plt.xticks(index + bar_width, subjects)
# Y轴范围
#plt.ylim(top=100, bottom=0)
plt.ylim()
# 图表标题
plt.title(u'Gene Distribution')
# 图例显示在图表下方
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)

# 添加数据标签
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
        # 柱形图边缘用白色填充，纯粹为了美观
        rect.set_edgecolor('white')

add_labels(rects1)
add_labels(rects2)
#plt.xticks([])
#plt.yticks([])
plt.show()

x1, y1, z1 = np.log(E3Mat[:,1]+1e-6), np.log(E3Mat[:,2]+1e-6), np.log(E3Mat[:,4]+1e-6)
x2, y2, z2 = np.log(E5Mat[:,1]+1e-6), np.log(E5Mat[:,2]+1e-6), np.log(E5Mat[:,4]+1e-6)
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(x1,y1,z1, c='b')  # 绘制数据点
ax.scatter(x2,y2,z2, c='r')
ax.set_zlabel('Gene 5')  # 坐标轴
ax.set_ylabel('Gene 3')
ax.set_xlabel('Gene 2')
plt.show()

x1, y1, z1 = np.log(E3Mat[:,0]+1e-6),np.log(E3Mat[:,1]+1e-6),np.log(E3Mat[:,9]+1e-6)
x2, y2, z2 = np.log(E5Mat[:,0]+1e-6),np.log(E5Mat[:,1]+1e-6),np.log(E5Mat[:,9]+1e-6)
x3, y3, z3 = np.log(E3Mat1[:,0]+1e-6),np.log(E3Mat1[:,1]+1e-6),np.log(E3Mat1[:,9]+1e-6)
x4, y4, z4 = np.log(E5Mat1[:,0]+1e-6),np.log(E5Mat1[:,1]+1e-6),np.log(E5Mat1[:,9]+1e-6)
fig = plt.figure()
ax1 = Axes3D(fig)
X = np.arange(-10, 10, 0.5)
Y = np.arange(-10, 10, 0.5)
X, Y = np.meshgrid(X, Y)
Z = (-0.10582355*X-0.10340082*Y-1.37407842)/0.09172775
ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
#ax=plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#ax1.scatter(x1,y1,z1, c='b')  # 绘制数据点
#ax1.scatter(x2,y2,z2, c='r')
ax1.scatter(x3,y3,z3, c='g')  # 绘制数据点
ax1.scatter(x4,y4,z4, c='orange')
ax1.set_zlabel('Gene 10')  # 坐标轴
ax1.set_ylabel('Gene 2')
ax1.set_xlabel('Gene 1')
plt.show()

#画出包裹法结果


x1, y1, z1 = np.log(E3Mat[:,1]+1e-6),np.log(E3Mat[:,4]+1e-6),np.log(E3Mat[:,6]+1e-6)
x2, y2, z2 = np.log(E5Mat[:,1]+1e-6),np.log(E5Mat[:,4]+1e-6),np.log(E5Mat[:,6]+1e-6)
x3, y3, z3 = np.log(E3Mat1[:,1]+1e-6),np.log(E3Mat1[:,4]+1e-6),np.log(E3Mat1[:,6]+1e-6)
x4, y4, z4 = np.log(E5Mat1[:,1]+1e-6),np.log(E5Mat1[:,4]+1e-6),np.log(E5Mat1[:,6]+1e-6)
# 画出包裹法的结果
fig = plt.figure()
ax1 = Axes3D(fig)
X = np.arange(-10, 10, 0.5)
Y = np.arange(-10, 10, 0.5)
X, Y = np.meshgrid(X, Y)
Z = (-0.08017311*X-0.45458353*Y+2.49312608)/(-0.08808377)
ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
#ax=plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
ax1.scatter(x1,y1,z1, c='b')  # 绘制数据点
ax1.scatter(x2,y2,z2, c='r')
#ax1.scatter(x3,y3,z3, c='g')  # 绘制数据点
#ax1.scatter(x4,y4,z4, c='orange')
ax1.set_zlabel('Gene 7')  # 坐标轴
ax1.set_ylabel('Gene 5')
ax1.set_xlabel('Gene 2')
plt.show()

fig=plt.figure()
ax=plt.subplot(1,1,1)
for i in range(0,TrainFeature.shape[0]):
    for j in range(0,TrainFeature.shape[1]):
        if(TrainLabel[i]=="E3"):
            ax.scatter(j,TrainFeature[i,j],c="g")
        else:
            ax.scatter(j,TrainFeature[i,j],c='orange')
ax.set_ylabel('Value')
ax.set_xlabel('Gene')
plt.show()
'''


#画图部分
countE3=np.zeros([7,10])
for i in range(0,E3Mat.shape[1]):
    for h in range(0,E3Mat.shape[0]):
        if(E3Mat[h,i]==0):
            countE3[0,i]=countE3[0,i]+1
        if(E3Mat[h,i]>0 and E3Mat[h,i]<=100):
            countE3[1, i] = countE3[1, i] + 1
        if (E3Mat[h, i] > 100 and E3Mat[h, i] <= 300):
            countE3[2, i] = countE3[2, i] + 1
        if (E3Mat[h, i] > 300 and E3Mat[h, i] <= 500):
            countE3[3, i] = countE3[3, i] + 1
        if (E3Mat[h, i] > 500 and E3Mat[h, i] <= 1000):
            countE3[4, i] = countE3[4, i] + 1
        if (E3Mat[h, i] > 1000 and E3Mat[h, i] <= 5000):
            countE3[5, i] = countE3[5, i] + 1
        if (E3Mat[h, i] > 5000 ):
            countE3[6, i] = countE3[6, i] + 1
countE5=np.zeros([7,10])
for i in range(0,E5Mat.shape[1]):
    for h in range(0,E5Mat.shape[0]):
        if(E5Mat[h,i]==0):
            countE5[0,i]=countE5[0,i]+1
        if(E5Mat[h,i]>0 and E5Mat[h,i]<=100):
            countE5[1, i] = countE5[1, i] + 1
        if (E5Mat[h, i] > 100 and E5Mat[h, i] <= 300):
            countE5[2, i] = countE5[2, i] + 1
        if (E5Mat[h, i] > 300 and E5Mat[h, i] <= 500):
            countE5[3, i] = countE5[3, i] + 1
        if (E5Mat[h, i] > 500 and E5Mat[h, i] <= 1000):
            countE5[4, i] = countE5[4, i] + 1
        if (E5Mat[h, i] > 1000 and E5Mat[h, i] <= 5000):
            countE5[5, i] = countE5[5, i] + 1
        if (E5Mat[h, i] > 5000 ):
            countE5[6, i] = countE5[6, i] + 1
print(countE5)


font_size = 10  # 字体大小
fig_size = (8, 6)  # 图表大小

names = ["Gene1","Gene2","Gene3","Gene4","Gene5","Gene6","Gene7","Gene8","Gene9","Gene10"]  # 姓名
subjects = ["=0","0-100","100-300","300-500","500-1000","1000-5000",">5000"]  # 科目
scores = [countE3.T[0],countE3.T[1],countE3.T[2],countE3.T[3],countE3.T[4],countE3.T[5],countE3.T[6],countE3.T[7],countE3.T[8],countE3.T[9]]  # 成绩
#scores = [countE5.T[0],countE5.T[1],countE5.T[2],countE5.T[3],countE5.T[4],countE5.T[5],countE5.T[6],countE5.T[7],countE5.T[8],countE5.T[9]]  # 成绩

# 更新字体大小
mpl.rcParams['font.size'] = font_size
# 更新图表大小
mpl.rcParams['figure.figsize'] = fig_size
# 设置柱形图宽度
bar_width = 0.05

index = np.arange(7)
rects1 = plt.bar(index, scores[0], bar_width, color='#DB9019', label=names[0])
rects2 = plt.bar(index + bar_width, scores[1], bar_width, color='#5ED5D1', label=names[1])
rects3 = plt.bar(index + 2*bar_width, scores[2], bar_width, color='#1A2D27', label=names[2])
rects4 = plt.bar(index + 3*bar_width, scores[3], bar_width, color='#FF6E97', label=names[3])
rects5 = plt.bar(index + 4*bar_width, scores[4], bar_width, color='#F1AAA6', label=names[4])
rects6 = plt.bar(index + 5*bar_width, scores[5], bar_width, color='#82A6F5', label=names[5])
rects7 = plt.bar(index + 6*bar_width, scores[6], bar_width, color='#EAF048', label=names[6])
rects8 = plt.bar(index + 7*bar_width, scores[7], bar_width, color='#2A5200', label=names[7])
rects9 = plt.bar(index + 8*bar_width, scores[8], bar_width, color='#F6D6FF', label=names[8])
rects10 = plt.bar(index + 9*bar_width, scores[9], bar_width, color='#9FF048', label=names[9])

plt.xticks(index + bar_width, subjects)
# Y轴范围
#plt.ylim(top=100, bottom=0)
plt.ylim()
# 图表标题
plt.title(u'Gene Distribution')
# 图例显示在图表下方
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)

# 添加数据标签
'''
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
        # 柱形图边缘用白色填充，纯粹为了美观
        rect.set_edgecolor('white')

#add_labels(rects1)
add_labels(rects2)
add_labels(rects3)
add_labels(rects4)
add_labels(rects5)
add_labels(rects6)
add_labels(rects7)
add_labels(rects8)
add_labels(rects9)
add_labels(rects10)
#plt.xticks([])
#plt.yticks([])
'''
plt.show()

