import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

##################
#数据处理
##################
path_csv = 'D:\\pythonProject_al\\data\\predict_data.csv'
data_1 = pd.read_csv(path_csv)

dic_time = {'10年以上':6,'5-10年':5,'3-5年':4,'1-3年':3,'1年以内':2,'在校/应届':1,'经验不限':0}
dic_edu = {'硕士':3,'本科':2,'大专':1,'学历不限':0}
dic_com = {'电子商务':4,'移动互联网':3,'其他行业':2,'计算机软件':1,'互联网':0}

X = []
y_high = [i for i in data_1['月薪上限']]
y_low = [i for i in data_1['月薪下限']]
#地区特征
v1 = []
#工作经历
v2_mid = []
v2 = []
#学历
v3_mid = []
v3 = []
#公司行业
v4 = []
#工作能力
v5 = []

#将不同特征转成onehot向量
for i in data_1['工作经验']:
    v2_mid = dic_time[i]
    y = np.array(v2_mid)
    v2.append(np.eye(len(dic_time))[y.reshape(-1)].tolist())

for i in data_1['学历']:
    v3_mid = dic_edu[i]
    y = np.array(v3_mid)
    v3.append(np.eye(len(dic_edu))[y.reshape(-1)].tolist())

for i in data_1['公司行业']:
    if i in dic_com.keys():
        v4_mid = dic_com[i]
    else:
        v4_mid = dic_com['其他行业']
    y = np.array(v4_mid)
    v4.append(np.eye(len(dic_com))[y.reshape(-1)].tolist())

dic_city = {}
for i in range(len(data_1)):
    city = data_1["市"][i]
    if city in dic_city.keys():
        dic_city[city] = dic_city[city]+1
    else:
        dic_city[city] = 1

index = 0
for i in dic_city.keys():
    dic_city[i] = index
    index = index+1
print(dic_city)

for i in data_1['市']:
    v1_mid = dic_city[i]
    y = np.array(v1_mid)
    v1.append(np.eye(len(dic_city))[y.reshape(-1)].tolist())

dic_cap = {}
for i in range(len(data_1)):
    for word in eval(data_1['工作能力'][i]):
        if word in dic_cap.keys():
            dic_cap[word] = dic_cap[word]+1
        else:
            for word_x in dic_cap.keys():
                if word_x.casefold() == word.casefold():
                    dic_cap[word_x] = dic_cap[word_x]+1
                    continue
            dic_cap[word] = 1

dic_cap_mid = [[k,v] for k,v in dic_cap.items()]
dic_cap1 = sorted(dic_cap_mid,key=lambda x:x[1],reverse=True)

dic_cap={}
index = 0
for i in range(len(dic_cap1)):
    if dic_cap1[i][1]<50:
        continue
    dic_cap[dic_cap1[i][0]] = index
    index = index+1
# dic_cap1 = sorted(dic_cap,key=lambda x:x[0])
print(dic_cap)
print(dic_cap1)

v5_mid2 = np.zeros(len(dic_cap))
v5_mid3 = np.zeros(len(dic_cap))
for i in data_1['工作能力']:
    for word in eval(i):
        for word_x in dic_cap.keys():
            if word.casefold() == word_x.casefold():
                v5_mid = dic_cap[word_x]
                y = np.array(v5_mid)
                v5_mid2 = np.eye(len(dic_cap))[y.reshape(-1)].tolist()
        v5_mid3 = v5_mid3+v5_mid2
    v5.append(v5_mid3)

# print(v5[0])
# print(type(v1[0]))
# print(type(v2[0]))
# print(type(v3[0]))
# print(type(v4[0]))
# print(type(v5[0]))

#将不同的特征合并起来做为输入
merge = []
for i in range(len(y_low)):
# for i in range(1):
    # merge.append(list(itertools.chain(v1[i] + v2[i] + v3[i] + v4[i] + v5[i].tolist())))
    # print(v5[i].tolist())
    X.append(v1[i][0]+v2[i][0]+v3[i][0]+v4[i][0]+v5[i].tolist()[0])
    # print(X)

# print(X[0])
# merge=sum(X,[])

# result = list(merge)
# print(merge[0])
print('Xtrain:',np.array(X).shape)
print('ytrain:',np.array(y_high).shape)

col = []
for i in dic_city.keys():
    col.append(i)
for i in dic_time.keys():
    col.append(i)
for i in dic_edu.keys():
    col.append(i)
for i in dic_com.keys():
    col.append(i)
for i in dic_cap.keys():
    col.append(i)

# print(type(y_high))
# mean = np.array(y_high).mean()
# std = np.array(y_high).std()
# y_high = [(y-mean)/std for y in y_high]
# print(type(y_high))
train1, test1, train_label1, test_label1 = train_test_split(X, y_low, test_size = 0.2, random_state = 2020)

print(np.array(test1[1]).shape)
#处理源数据  1、每张图片的像素点拉成一维向量 28*28=784 在以后学习CNN的时候时不需要拉直的，拉直后其实损失了很多图像的信息
#2、标准化 像素点的值在0-255之间 变成 -1~1之间
#3、转为tensor
def data_tf(x):
    x = np.array(x,dtype='float32')
    x = x.reshape(x.shape[0],-1)
    # x = x/255
    # x = (x-0.5)/0.5 #(-1,1)
    x = torch.from_numpy(x)
    return x

train = data_tf(train1)#[60000,784]
test = data_tf(test1)#[10000,784]
test_label = data_tf(test_label1)
train_label = data_tf(train_label1)
#TensorDataset 时torch自带的 将数据变成 dataset格式 数据+对应标签进行绑定
train_set = TensorDataset(train,train_label)
test_set = TensorDataset(test,test_label)


#训练集数据量太大，需要分成一定批次输入进神经网络 定义迭代器
#batch_size 每个小数据集包含的图片数 shuffle 是否打乱数据集的顺序
train_data = DataLoader(dataset=train_set,batch_size=64,shuffle=True)
test_data  = DataLoader(dataset=test_set,batch_size=64,shuffle=False)
train_data_graph  = DataLoader(dataset=train_set,batch_size=1,shuffle=False)
test_data_graph  = DataLoader(dataset=test_set,batch_size=1,shuffle=False)


class Net(torch.nn.Module):# 继承 torch 的 Module
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        # 定义每层用什么样的形式

        self.layer1 = torch.nn.Linear(n_feature, 30)   #
        self.layer2 = torch.nn.Linear(30, n_output)

    def forward(self, x):   # 这同时也是 Module 中的 forward 功能
        x = self.layer1(x)
        x = torch.relu(x)      #
        x = self.layer2(x)
        return x
net = Net(67,1)
# print(net)
#反向传播算法 SGD Adam等
Learning_rate = 1e-2
# optimizer = torch.optim.SGD(net.parameters(), lr=Learning_rate)
optimizer = torch.optim.RMSprop(net.parameters(),lr=Learning_rate)
# optimizer = torch.optim.Adam(net.parameters(),lr=Learning_rate)
# optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
#损失函数 交叉熵
criterion =	torch.nn.MSELoss()

small = 100000
y_pred = []
y_true = []
y_pred2 = []
y_true2 = []
losses = []#记录每次迭代后训练的loss
acces = []#记录每次迭代后训练的精准度
eval_losses = []#测试的
eval_acces = []
for i in range(500):
    train_loss = 0
    train_acc = 0
    net.train() #网络设置为训练模式 暂时可加可不加
    for tdata,tlabel in train_data:
        #tdata [64,784] tlabel [64]
        #前向传播
        y_ = net(tdata)
        #记录单批次一次batch的loss
        loss = criterion(y_, tlabel)
        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #累计单批次误差
        train_loss = train_loss + loss.item()
        #计算分类的准确率
        _, pred = y_.max(1)#求出每行的最大值 值与序号pred
        num_correct = (pred == tlabel).sum().item()
        acc = num_correct/tlabel.shape[0]
        train_acc = train_acc + acc
    losses.append(train_loss/len(train_data))
    acces.append(train_acc/len(train_data))
    print('epoch: {}, trainloss: {},trainacc: {}'.format(i, train_loss/len(train_data), train_acc/len(train_data)))
#
    #测试集进行测试
    eval_loss = 0
    eval_acc = 0
    net.eval() #可加可不加
    for edata,elabel in test_data:
        #前向传播
        y_ = net(edata)
        #记录单批次一次batch的loss，测试集就不需要反向传播更新网络了
        loss = criterion(y_, elabel)
        #累计单批次误差
        eval_loss = eval_loss + loss.item()
        #计算分类的准确率
        _, pred = y_.max(1)#求出每行的最大值 值与序号pred
        num_correct = (pred == elabel).sum().item()
        acc = num_correct/elabel.shape[0]
        eval_acc = eval_acc + acc
    # if (eval_loss/len(test_data))<small:
    #     y_pred = []
    #     y_true = []
    #     y_pred2 = []
    #     y_true2 = []
    #     # test_label1
    #     for edata, elabel in test_data_graph:
    #         # 前向传播
    #         # y_ = net(edata)
    #         y_pred.append(net(edata).detach().numpy().tolist()[0][0])
    #         y_true.append(elabel.detach().numpy().tolist()[0][0])
    #     for edata, elabel in train_data_graph:
    #         # 前向传播
    #         # y_ = net(edata)
    #         y_pred2.append(net(edata).detach().numpy().tolist()[0][0])
    #         y_true2.append(elabel.detach().numpy().tolist()[0][0])
    eval_losses.append(eval_loss/len(test_data))
    eval_acces.append(eval_acc/len(test_data))
    print('epoch: {}, evalloss: {},evalacc: {}'.format(i, eval_loss/len(test_data), eval_acc/len(test_data)))

# 将预测的数据存入表格中，以便后续的可视化处理
dataframe = pd.DataFrame({'训练集loss':losses,'测试集loss':eval_losses,})
dataframe.to_csv(r'D:\\pythonProject_al\\data\\bp_RMSprop_predict_loss_l.csv', index=False)
# dataframe2 = pd.DataFrame({'adam预测值':y_pred,'adam真实值':y_true})
# dataframe2.to_csv(r'D:\\pythonProject_al\\data\\bp_adam_predicted_l.csv', index=False)
# dataframe3 = pd.DataFrame({'adam预测值':y_pred2,'adam真实值':y_true2})
# dataframe3.to_csv(r'D:\\pythonProject_al\\data\\bp_adam_predicted2_l.csv', index=False)

# plt.plot(losses)
# plt.plot(eval_loss)
# plt.show()