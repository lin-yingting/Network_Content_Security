import pandas as pd
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts
from pyecharts.charts import Page

#读取之前的三种方法的预测结果并画图
path_csv1 = 'D:\\pythonProject_al\\data\\xgb_predicted_l.csv'
path_csv2 = 'D:\\pythonProject_al\\data\\rfr_predicted_l.csv'
path_csv3 = 'D:\\pythonProject_al\\data\\bp_adam_predicted_l.csv'
path_csv4 = 'D:\\pythonProject_al\\data\\xgb_predicted.csv'
path_csv5 = 'D:\\pythonProject_al\\data\\rfr_predicted.csv'
path_csv6 = 'D:\\pythonProject_al\\data\\bp_adam_predicted.csv'

data = []
data.append(pd.read_csv(path_csv1))
data.append(pd.read_csv(path_csv2))
data.append(pd.read_csv(path_csv3))
data.append(pd.read_csv(path_csv4))
data.append(pd.read_csv(path_csv5))
data.append(pd.read_csv(path_csv6))

x_data0_h = [i for i in data[3]['xgb预测值']]
y_data0_h = [i for i in data[3]['xgb真实值']]
x_data1_h = [i for i in data[4]['rfr预测值']]
y_data1_h = [i for i in data[4]['rfr真实值']]
x_data2_h = [i for i in data[5]['adam预测值']]
y_data2_h = [i for i in data[5]['adam真实值']]

x_data0 = [i for i in data[0]['xgb预测值']]
y_data0 = [i for i in data[0]['xgb真实值']]
x_data1 = [i for i in data[1]['rfr预测值']]
y_data1 = [i for i in data[1]['rfr真实值']]
x_data2 = [i for i in data[2]['adam预测值']]
y_data2 = [i for i in data[2]['adam真实值']]
#
# for i in range(len(data)):


import pyecharts.options as opts
from pyecharts.charts import Scatter
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#计算使用不同预测方法预测后的评价指标
maetrain = ['MAE']
msetrain = ['MSE']
r2 = ['R-Squared']
maetrain.append('%.3f'%mean_absolute_error(x_data0,y_data0))
maetrain.append('%.3f'%mean_absolute_error(x_data1,y_data1))
maetrain.append('%.3f'%mean_absolute_error(x_data2,y_data2))
# print(maetrain0)
# print(maetrain1)
# print(maetrain2)

msetrain.append('%.3f'%mean_squared_error(x_data0,y_data0))
msetrain.append('%.3f'%mean_squared_error(x_data1,y_data1))
msetrain.append('%.3f'%mean_squared_error(x_data2,y_data2))
# print(msetrain0)
# print(msetrain1)
# print(msetrain2)

r2.append('%.3f'%r2_score(y_data0,x_data0))
r2.append('%.3f'%r2_score(y_data1,x_data1))
r2.append('%.3f'%r2_score(y_data2,x_data2))
# print(r2_0)
# print(r2_1)
# print(r2_2)

maetrain.append('MAE')
msetrain.append('MSE')
r2.append('R-Squared')

maetrain.append('%.3f'%mean_absolute_error(x_data0_h,y_data0_h))
maetrain.append('%.3f'%mean_absolute_error(x_data1_h,y_data1_h))
maetrain.append('%.3f'%mean_absolute_error(x_data2_h,y_data2_h))
# print(maetrain0)
# print(maetrain1)
# print(maetrain2)

msetrain.append('%.3f'%mean_squared_error(x_data0_h,y_data0_h))
msetrain.append('%.3f'%mean_squared_error(x_data1_h,y_data1_h))
msetrain.append('%.3f'%mean_squared_error(x_data2_h,y_data2_h))
# print(msetrain0)
# print(msetrain1)
# print(msetrain2)

r2.append('%.2f'%r2_score(y_data0_h,x_data0_h))
r2.append('%.2f'%r2_score(y_data1_h,x_data1_h))
r2.append('%.2f'%r2_score(y_data2_h,x_data2_h))

#利用评价指标画图
table = Table()

headers = ["模型", "XGBoost", "Random Forest", "BP"]
rows = [
    maetrain[:4],
    msetrain[:4],
    r2[:4]
]
table.add(headers, rows)
table.set_global_opts(
    title_opts=ComponentTitleOpts(title="最低工资")
)

table2 = Table()

headers2 = ["模型", "XGBoost", "Random Forest", "BP"]
rows2 = [
    maetrain[4:],
    msetrain[4:],
    r2[4:]
]
table2.add(headers2, rows2)
table2.set_global_opts(
    title_opts=ComponentTitleOpts(title="最高工资")
)

# table2.render("table_base2.html")
# print(r2_0)
# print(r2_1)
# print(r2_2)
#利用预测值和真实值画散点图
s1 = (
    Scatter()
    .add_xaxis(xaxis_data=x_data0)
    .add_yaxis(
        series_name="",
        y_axis=y_data0,
        symbol_size=5,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_series_opts()
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="value",
            splitline_opts=opts.SplitLineOpts(is_show=False),
            name="预测值(K)",
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),
            name="真实值(K)",
        ),
        tooltip_opts=opts.TooltipOpts(is_show=False),
        title_opts=opts.TitleOpts(
            title="XGBoost", pos_left="center"
        ),
    )
    # .render("basic_scatter_chart3.html")
)

s2 = (
    Scatter()
    .add_xaxis(xaxis_data=x_data1)
    .add_yaxis(
        series_name="",
        y_axis=y_data1,
        symbol_size=5,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_series_opts()
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="value",
            splitline_opts=opts.SplitLineOpts(is_show=False),
            name="预测值(K)",
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),
            name="真实值(K)",
        ),
        tooltip_opts=opts.TooltipOpts(is_show=False),
        title_opts=opts.TitleOpts(
            title="Random Forest", pos_left="center"
        ),
    )
    # .render("basic_scatter_chart3.html")
)

s3 = (
    Scatter()
    .add_xaxis(xaxis_data=x_data2)
    .add_yaxis(
        series_name="",
        y_axis=y_data2,
        symbol_size=5,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_series_opts()
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="value",
            splitline_opts=opts.SplitLineOpts(is_show=False),
            name="预测值(K)",
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=False),
            splitline_opts=opts.SplitLineOpts(is_show=False),
            name="真实值(K)",
        ),
        tooltip_opts=opts.TooltipOpts(is_show=False),
        title_opts=opts.TitleOpts(
            title="BP", pos_left="center"
        ),
    )
    # .render("basic_scatter_chart3.html")
)

#生成可排版的网页
# page = Page(layout=Page.DraggablePageLayout)
# page.add(s1,s2,s3,table,table2)
# page.render("predict_all.html")

#生成排版好的网页
Page.save_resize_html("predict_all.html", cfg_file="D:\\pythonProject_al\\chart_config6.json", dest="predect_all2.html")

