import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score ,calinski_harabasz_score,davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

path_csv = 'D:\\pythonProject_al\\data\\test2.csv'
data_1 = pd.read_csv(path_csv)

#对java岗的福利数据做k-means处理，得到聚类结果
dic = {'长沙': 15611, '天津': 15513, '厦门': 17252, '北京': 17811, '苏州': 18371, '郑州': 16496, '西安': 17301, '杭州': 17411, '上海': 17413, '深圳': 17068, '武汉': 17664, '重庆': 16085, '成都': 16166}
data_radio = {}
welfare_seek = ['五险一金','节日福利','带薪年假','年终奖','定期体检','员工旅游']
for city in dic.keys():
    # '五险一金''节日福利''带薪年假'年终奖'定期体检''员工旅游'
    data_radio[city] = [0, 0, 0, 0, 0, 0]
for i in range(len(data_1)):
    for j in eval(data_1["福利"][i]):
        if data_1["市"][i] in dic.keys():
            if j in welfare_seek:
                data_radio[data_1["市"][i]][welfare_seek.index(j)] = data_radio[data_1["市"][i]][welfare_seek.index(j)]+1

# print(data_radio)
# print(data_radio.values())
X_mid = [i for i in data_radio.values()]
X = []
for i in X_mid:
    X_sum = sum(i)
    X.append([j/X_sum for j in i])
s1 = []
s2 = []
s3 = []
s4 = []
s5 = []
s6 = []
s7 = []
s8 = []
s9 = []

print(X)
for i in range(10):
    est = KMeans(n_clusters=i+2, random_state=1)#k-means聚类
    est.fit(X)
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=i+2)  # 层次聚类
    clustering.fit(X)
    # print(clustering.labels_)
    gmms = GaussianMixture(n_components=i+2)  # 混合高斯模型聚类
    gmms.fit(X)
    # print(est.labels_)

    s1.append(silhouette_score(X, est.labels_, metric='euclidean')) # 计算轮廓系数
    s2.append(silhouette_score(X, clustering.labels_, metric='euclidean')) # 计算轮廓系数
    s3.append(silhouette_score(X, gmms.predict(X), metric='euclidean')) # 计算轮廓系数
    s4.append(calinski_harabasz_score(X, est.labels_)) # 计算轮廓系数
    s5.append(calinski_harabasz_score(X, clustering.labels_)) # 计算轮廓系数
    s6.append(calinski_harabasz_score(X, gmms.predict(X))) # 计算轮廓系数

    s7.append(davies_bouldin_score(X, est.labels_)) # 计算轮廓系数
    s8.append(davies_bouldin_score(X, clustering.labels_)) # 计算轮廓系数
    s9.append(davies_bouldin_score(X, gmms.predict(X))) # 计算轮廓系数
    # s3.append(davies_bouldin_score(X,est.labels_))    # 计算 DBI
# print(X)
# print(KMeans(n_clusters=4, random_state=4).fit(X).labels_)
#选取4做为簇数
map_mid = KMeans(n_clusters=4, random_state=4).fit(X).labels_
# map_mid = KMeans(n_clusters=4, random_state=4).fit(X).labels_
map_keans = {}
mid = 0
for i in data_radio.keys():
    map_keans[i] = map_mid[mid]
    mid = mid+1
print(map_keans)
print(s1)
print(s2)
print(s3)
# plt.plot(s2)
# plt.legend("clus")
# plt.plot(s3)
# plt.legend("gmms")

map_keans_fin1 = [k for k in map_keans.keys()]
map_keans_fin2 = [v for v in map_keans.values()]
dataframe = pd.DataFrame({'市':map_keans_fin1,'聚类簇号':map_keans_fin2})
dataframe.to_csv(r'D:\\pythonProject_al\\data\\kmeans.csv', index=False)
# plt.figure(12)
# plt.subplot(331)
x = [i+2 for i in range(len(s1))]
plt.plot(x,s1)

#画出三种聚类方法在不同聚类指标下，不同n_clusters的图
# plt.subplot(332)
# plt.plot(s2)
#
# plt.subplot(333)
# plt.plot(s3)
#
# plt.subplot(334)
# plt.plot(s4)
#
# plt.subplot(335)
# plt.plot(s5)
#
# plt.subplot(336)
# plt.plot(s6)
#
# plt.subplot(337)
# plt.plot(s7)
#
# plt.subplot(338)
# plt.plot(s8)
#
# plt.subplot(339)
# plt.plot(s9)


plt.show()

