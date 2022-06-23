import pandas as pd
import copy
#读取python岗爬取的所有数据
# path_csv='C:\\Users\\林映廷\\Desktop\\allkong.csv'
#读取java岗爬取的所有数据
path_csv='C:\\Users\\林映廷\\Desktop\\all-javas2.csv'
#读取c/c++岗爬取的所有数据
# path_csv='C:\\Users\\林映廷\\Desktop\\allc2.csv'
# data=pd.read_csv(path_csv,usecols=[1,2])
data=pd.read_csv(path_csv)
# print(data["2"])
# print(data["2"][1].split("·")[0])

#将数据转为后续数据处理部分易使用数据格式
a = [i.split("·")[0] for i in data["2"]]
c = [i.split("，") for i in data["9"]]
d = data["4"]
e = [i.split(" ") for i in data["8"]]
f = data["7"]
g = data["5"]

dic_time = {'10年以上':6,'5-10年':5,'3-5年':4,'1-3年':3,'1年以内':2,'在校/应届':1,'经验不限':0}
#工资上限
h = []
#工资下限
l = []
#工资平均值
b = []
mid1 = -1
mid2 = -1
mid3 = []
month = 0
low = 0
high = 0
fin = 0
index = -1

print(c[1])
for i in data["3"]:
    index = index+1
    if "薪" in i:
        mid1 = i.find("薪")
        mid2 = i.find("K")
        mid3 = i[0:mid2]
        x = mid3.split("-")
        month = i[mid2+2:mid1]
        # fin = (int(x[0])+int(x[1]))*int(month)/12.0
        h.append((0+int(x[1]))*int(month)/12)
        l.append((int(x[0])+0)*int(month)/12)
        fin = (int(x[0])+int(x[1]))*int(month)/24
        b.append(fin)
        continue
    if "K" in i :
        mid2 = i.find("K")
        mid3 = i[0:mid2]
        x = mid3.split("-")
        h.append((0 + int(x[1]))/1.0)
        l.append((0 + int(x[0]))/1.0)
        fin = (int(x[0]) + int(x[1]))/2.0
        # print(fin)
        # print("\n")
        b.append(fin)
        continue
    if "天" in i:
        mid2 = i.find("元")
        mid3 = i[0:mid2]
        x = mid3.split("-")
        fin = (int(x[0]) + int(x[1]))*30/2.0/1000
        h = (0 + int(x[1]))*30/1000
        l = (int(x[0]) + 0)*30/1000
        # print(i)
        # print(fin)
        # print("\n")
        b.append(fin)
        continue
    a.pop(index)
    c.pop(index)
    d.pop(index)
    e.pop(index)
    f.pop(index)
    g.pop(index)
    index = index-1



index = -1
length = copy.deepcopy(len(d))
d_ = copy.deepcopy(d)
print(length)
for i in d_:
    index = index + 1
    if i in dic_time.keys():
        continue
    a.pop(index)
    b.pop(index)
    c.pop(index)
    d.pop(index)
    e.pop(index)
    f.pop(index)
    g.pop(index)
    h.pop(index)
    l.pop(index)
    index = index-1

# 将处理后的数据存储在文件中，等待数据处理部分的程序读取
dataframe = pd.DataFrame({'市':a,'月薪':b,'福利':c,'工作经验':d,'工作能力':e,'公司行业':f,'学历':g,'月薪上限':h,'月薪下限':l})
dataframe.to_csv(r'D:\\pythonProject_al\\data\\predict_data.csv', index=False)

