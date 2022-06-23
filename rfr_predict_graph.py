import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,make_scorer,r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,train_test_split
import pyecharts.options as opts
from pyecharts.charts import Line, Grid,Bar,Page

#读取信息
path_csv = 'D:\\pythonProject_al\\data\\predict_data.csv'
data_1 = pd.read_csv(path_csv)
#数据预处理，将不同特征转为onehot向量
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

# 将不同特征合成一个向量输入
merge = []
for i in range(len(y_low)):
    X.append(v1[i][0]+v2[i][0]+v3[i][0]+v4[i][0]+v5[i].tolist()[0])

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

# 使用较好的超参数利用随机森林进行预测
#####################################
# 预测最高工资
####################################
y_low_log = np.log(y_low)
x_train, x_test, y_train, y_test = train_test_split(X, y_high, test_size = 0.2, random_state = 2020)

model = RandomForestRegressor(max_depth=6,n_estimators=30)
model.fit(x_train,y_train)
predict_test = model.predict(x_test)
predict_train = model.predict(x_train)

msetest=mean_squared_error(y_test,predict_test)
msetrain=mean_squared_error(y_test,predict_test)
maetrain=mean_absolute_error(y_test,predict_test)
print(msetest)
print(msetrain)
print(maetrain)


##################################
# 画图
##################################

id_1 = list(range(len(y_train)))
# print(type(id))
print(type(y_train))
print(type(predict_train))
# print(len(id))
print(len(y_train))
print(len(predict_train))
predict_train = predict_train.tolist()
l1 = (
    Line()
    .add_xaxis(xaxis_data=id_1)
    # .add_xaxis(xaxis_data=timeData)
    .add_yaxis(
        series_name="实际最高工资",
        y_axis=y_train,
        # y_axis=water_flowData,
        # symbol_size=8,
        is_hover_animation=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=1.5),
        is_smooth=True,
    )
    .add_yaxis(
        series_name="预测最高工资",
        y_axis=predict_train,
        # y_axis=water_flowData,
        # symbol_size=8,
        is_hover_animation=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=1.5),
        is_smooth=True,
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            # title="雨量流量关系图", subtitle="数据来自西安兰特水电测控技术有限公司", pos_left="center"
            title="训练集", pos_left="center"
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        axispointer_opts=opts.AxisPointerOpts(
            is_show=True, link=[{"xAxisIndex": "all"}]
        ),
        datazoom_opts=[
            opts.DataZoomOpts(
                is_show=True,
                is_realtime=True,
                # start_value=30,
                # end_value=70,
                # xaxis_index=[0, 1],
            )
        ],
        xaxis_opts=opts.AxisOpts(
            type_="category",
            boundary_gap=False,
            axisline_opts=opts.AxisLineOpts(is_on_zero=True),
        ),
        yaxis_opts=opts.AxisOpts(max_=120, name="工资(K)"),
        legend_opts=opts.LegendOpts(pos_left="left"),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature={
                "dataZoom": {"yAxisIndex": "none"},
                "restore": {},
                "saveAsImage": {},
            },
        ),
    )
    # .render("line_base.html")
)

id_2 = list(range(len(y_test)))
predict_test = predict_test.tolist()
l2 = (
    Line()
    .add_xaxis(xaxis_data=id_2)
    # .add_xaxis(xaxis_data=timeData)
    .add_yaxis(
        series_name="实际最高工资",
        y_axis=y_test,
        # y_axis=water_flowData,
        # symbol_size=8,
        is_hover_animation=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=1.5),
        is_smooth=True,
    )
    .add_yaxis(
        series_name="预测最高工资",
        y_axis=predict_test,
        # y_axis=water_flowData,
        # symbol_size=8,
        is_hover_animation=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=1.5),
        is_smooth=True,
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="测试集", pos_left="center"
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        axispointer_opts=opts.AxisPointerOpts(
            is_show=True, link=[{"xAxisIndex": "all"}]
        ),
        datazoom_opts=[
            opts.DataZoomOpts(
                is_show=True,
                is_realtime=True,
                # start_value=30,
                # end_value=70,
                # xaxis_index=[0, 1],
            )
        ],
        xaxis_opts=opts.AxisOpts(
            type_="category",
            boundary_gap=False,
            axisline_opts=opts.AxisLineOpts(is_on_zero=True),
        ),
        yaxis_opts=opts.AxisOpts(max_=120, name="工资(K)"),
        legend_opts=opts.LegendOpts(pos_left="left"),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature={
                "dataZoom": {"yAxisIndex": "none"},
                "restore": {},
                "saveAsImage": {},
            },
        ),
    )
    # .render("line_base2.html")
)

print(type(model.feature_importances_))
print(col)
mid = model.feature_importances_.tolist()
index = 0
for i in range(len(mid)):
    if mid[index] == 0:
        mid.pop(index)
        col.pop(index)
        continue
    index = index+1
#画出各特征的重要性图
b1 = (
    Bar()
    .add_xaxis(col)
    .add_yaxis("JAVA",mid)
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),

        title_opts=opts.TitleOpts(title="最高工资各特征重要性图", subtitle="最高工资"),
        visualmap_opts=opts.VisualMapOpts(
            is_calculable=True,
            dimension=1,
            pos_left="10",
            pos_top="center",
            range_text=["High", "Low"],
            range_color=["lightskyblue", "yellow", "orangered"],
            textstyle_opts=opts.TextStyleOpts(color="#ddd"),
            min_=0,
            max_=0.1,
        ),
    )
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    # .render("bar_base1.html")
)

#####################################
# 预测最低工资
####################################
x_train2, x_test2, y_train2, y_test2 = train_test_split(X, y_low, test_size = 0.2, random_state = 2020)

model2 = RandomForestRegressor(max_depth=6,n_estimators=30)
model2.fit(x_train2,y_train2)
predict_test2 = model2.predict(x_test2)
predict_train2 = model2.predict(x_train2)


dataframe = pd.DataFrame({'rfr预测值':predict_test2,'rfr真实值':y_test2})
dataframe.to_csv(r'D:\\pythonProject_al\\data\\rfr_predicted.csv', index=False)

maetrain2=mean_absolute_error(y_test2,predict_test2)
# print(msetest)
# print(msetrain)
print(maetrain2)


##################################
# 画图
##################################

id_1 = list(range(len(y_train2)))
# print(type(id))
print(type(y_train2))
print(type(predict_train2))
# print(len(id))
print(len(y_train2))
print(len(predict_train2))
predict_train2 = predict_train2.tolist()
#画随机森林预测结果与真实结果的折线图
l3 = (
    Line()
    .add_xaxis(xaxis_data=id_1)
    # .add_xaxis(xaxis_data=timeData)
    .add_yaxis(
        series_name="实际最低工资",
        y_axis=y_train2,
        # y_axis=water_flowData,
        # symbol_size=8,
        is_hover_animation=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=1.5),
        is_smooth=True,
    )
    .add_yaxis(
        series_name="预测最低工资",
        y_axis=predict_train2,
        # y_axis=water_flowData,
        # symbol_size=8,
        is_hover_animation=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=1.5),
        is_smooth=True,
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            # title="雨量流量关系图", subtitle="数据来自西安兰特水电测控技术有限公司", pos_left="center"
            title="训练集", pos_left="center"
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        axispointer_opts=opts.AxisPointerOpts(
            is_show=True, link=[{"xAxisIndex": "all"}]
        ),
        datazoom_opts=[
            opts.DataZoomOpts(
                is_show=True,
                is_realtime=True,
                # start_value=30,
                # end_value=70,
                # xaxis_index=[0, 1],
            )
        ],
        xaxis_opts=opts.AxisOpts(
            type_="category",
            boundary_gap=False,
            axisline_opts=opts.AxisLineOpts(is_on_zero=True),
        ),
        yaxis_opts=opts.AxisOpts(max_=120, name="工资(K)"),
        legend_opts=opts.LegendOpts(pos_left="left"),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature={
                "dataZoom": {"yAxisIndex": "none"},
                "restore": {},
                "saveAsImage": {},
            },
        ),
    )
    # .render("line_base.html")
)

id_2 = list(range(len(y_test2)))
predict_test2 = predict_test2.tolist()
l4 = (
    Line()
    .add_xaxis(xaxis_data=id_2)
    # .add_xaxis(xaxis_data=timeData)
    .add_yaxis(
        series_name="实际最低工资",
        y_axis=y_test2,
        # y_axis=water_flowData,
        # symbol_size=8,
        is_hover_animation=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=1.5),
        is_smooth=True,
    )
    .add_yaxis(
        series_name="预测最低工资",
        y_axis=predict_test2,
        # y_axis=water_flowData,
        # symbol_size=8,
        is_hover_animation=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=1.5),
        is_smooth=True,
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="测试集", pos_left="center"
            # title="雨量流量关系图", subtitle="数据来自西安兰特水电测控技术有限公司", pos_left="center"
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        axispointer_opts=opts.AxisPointerOpts(
            is_show=True, link=[{"xAxisIndex": "all"}]
        ),
        datazoom_opts=[
            opts.DataZoomOpts(
                is_show=True,
                is_realtime=True,
                # start_value=30,
                # end_value=70,
                # xaxis_index=[0, 1],
            )
        ],
        xaxis_opts=opts.AxisOpts(
            type_="category",
            boundary_gap=False,
            axisline_opts=opts.AxisLineOpts(is_on_zero=True),
        ),
        yaxis_opts=opts.AxisOpts(max_=120, name="工资(K)"),
        legend_opts=opts.LegendOpts(pos_left="left"),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature={
                "dataZoom": {"yAxisIndex": "none"},
                "restore": {},
                "saveAsImage": {},
            },
        ),
    )
    # .render("line_base2.html")
)

# print(type(model.feature_importances_))
# print(col)
mid2 = model2.feature_importances_.tolist()
# index = 0
# for i in range(len(mid)):
#     if mid[index] == 0:
#         mid.pop(index)
#         col.pop(index)
#         continue
#     index = index+1

b2 = (
    Bar()
    .add_xaxis(col)
    .add_yaxis("JAVA",mid2)
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),

        title_opts=opts.TitleOpts(title="最低工资各特征重要性图", subtitle="最低工资"),
        visualmap_opts=opts.VisualMapOpts(
            is_calculable=True,
            dimension=1,
            pos_left="10",
            pos_top="center",
            range_text=["High", "Low"],
            range_color=["lightskyblue", "yellow", "orangered"],
            textstyle_opts=opts.TextStyleOpts(color="#ddd"),
            min_=0,
            max_=0.1,
        ),
    )
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    # .render("bar_base1.html")
)



#生成可布局网页
# page = Page(layout=Page.DraggablePageLayout)
# page.add(l1,l2,b1,l3,l4,b2)
# page.render("rfr_all.html")

#生成布局完成的网页
# Page.save_resize_html("rfr_all.html", cfg_file="D:\\pythonProject_al\\chart_config4.json", dest="rfr_all2.html")


#####################################
#用网格法判断随机森林的n_estimators较佳取值
####################################

# MSE_mses = []       #储存利用不同超参数十折交叉验证的过程中产生的MSE，MAE,R2
# MAE_mses = []
# R2_mses = []
#
# def r2_secret_mse(estimator, X_test, y_test):
#     predictions = estimator.predict(X_test)
#     MSE_mses.append(mean_squared_error(y_test, predictions))
#     MAE_mses.append(mean_absolute_error(y_test, predictions))
#     R2_mses.append(r2_score(y_test, predictions))
#     return r2_score(y_test, predictions)
#
# param_test1 = {"n_estimators":range(1,101,10)}
# gsearch1 = GridSearchCV(estimator=RandomForestRegressor(max_depth=6),param_grid=param_test1,cv=10,scoring=r2_secret_mse)#不同超参数十折交叉验证
# gsearch1.fit(X,y_high)
##画出不同超参数对应的不同评价指标图
# plt.subplot(131)
# plt.plot(MAE_mses,color = "b")
# plt.scatter(30, MAE_mses[30], color='r',marker='o')
# plt.legend(['MAE'])
#
# plt.subplot(132)
# plt.plot(MSE_mses,color = "0.5")
# plt.scatter(30, MSE_mses[30], color='r',marker='o')
# plt.legend(['MSE'])
#
# plt.subplot(133)
# plt.plot(R2_mses,color = 'g')
# plt.scatter(30, R2_mses[30], color='r',marker='o')
# plt.legend(['R2'])
#
# plt.show()




