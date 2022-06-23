import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Line,Page


# ###########################
# #最高工资预测
# ###########################
# path_csv1 = 'D:\\pythonProject_al\\data\\bp_sgd_predict_loss.csv'
path_csv2 = 'D:\\pythonProject_al\\data\\bp_adam_predict_loss.csv'
path_csv3 = 'D:\\pythonProject_al\\data\\bp_RMSprop_predict_loss.csv'
path_csv4 = 'D:\\pythonProject_al\\data\\bp_adam_predicted.csv'
path_csv5 = 'D:\\pythonProject_al\\data\\bp_adam_predicted2.csv'

data = []
# data.append(pd.read_csv(path_csv1))
data.append(pd.read_csv(path_csv2))
data.append(pd.read_csv(path_csv3))
data.append(pd.read_csv(path_csv4))
data.append(pd.read_csv(path_csv5))

losses = []
eval_losses = []

# for i in data[2]['adam预测值']:
#     print(i)
#     # print('\n')
for i in range(2):
    losses.append([i for i in data[i]['训练集loss']])
    eval_losses.append([i for i in data[i]['测试集loss']])

x = list(range(len(losses[0])))
# x = [i+1 for i in x]
# 画不同梯度下降法方法，迭代次数和MSE的关系折线图
l1 = (
    Line()
    .add_xaxis(x[1:100])
    # .add_yaxis("SGD", losses[0][50:100],label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("Adam", losses[0][1:100],label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("RMSprop", losses[1][1:100],label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(title_opts=opts.TitleOpts(title="训练集",subtitle='最高工资'))
    # .render("line_bp_high1.html")
)

l2 = (
    Line()
    .add_xaxis(x[1:100])
    # .add_yaxis("SGD", losses[0][50:100],label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("Adam", eval_losses[0][1:100],label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("RMSprop", eval_losses[1][1:100],label_opts=opts.LabelOpts(is_show=False))
    # .add_yaxis("商家A", losses[1])
    # .add_yaxis("商家A", losses[2])
    .set_global_opts(title_opts=opts.TitleOpts(title="测试集",subtitle='最高工资'))
    # .render("line_bp_high2.html")
)
id_1 = list(range(len(data[2]['adam真实值'])))
# 画adam梯度下降法方法，预测值和真实值的折线图
l3 = (
    Line()
    .add_xaxis(xaxis_data=id_1)
    # .add_xaxis(xaxis_data=timeData)
    .add_yaxis(
        series_name="实际最高工资",
        y_axis=[i for i in data[2]['adam真实值']],
        # y_axis=water_flowData,
        # symbol_size=8,
        is_hover_animation=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=1.5),
        is_smooth=True,
    )
    .add_yaxis(
        series_name="预测最高工资",
        y_axis=[i for i in data[2]['adam预测值']],
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
    # .render("line_bp_high3.html")
)

id_2 = list(range(len(data[3]['adam真实值'])))
l4 = (
    Line()
    .add_xaxis(xaxis_data=id_2)
    # .add_xaxis(xaxis_data=timeData)
    .add_yaxis(
        series_name="实际最高工资",
        y_axis=[i for i in data[3]['adam真实值']],
        # y_axis=water_flowData,
        # symbol_size=8,
        is_hover_animation=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=1.5),
        is_smooth=True,
    )
    .add_yaxis(
        series_name="预测最高工资",
        y_axis=[i for i in data[3]['adam预测值']],
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
    # .render("line_bp_high4.html")
)

###########################
#最低工资预测
###########################

path_csv2_l = 'D:\\pythonProject_al\\data\\bp_adam_predict_loss_l.csv'
path_csv3_l = 'D:\\pythonProject_al\\data\\bp_RMSprop_predict_loss_l.csv'
path_csv4_l = 'D:\\pythonProject_al\\data\\bp_adam_predicted_l.csv'
path_csv5_l = 'D:\\pythonProject_al\\data\\bp_adam_predicted2_l.csv'

data_l = []
# data.append(pd.read_csv(path_csv1))
data_l.append(pd.read_csv(path_csv2_l))
data_l.append(pd.read_csv(path_csv3_l))
data_l.append(pd.read_csv(path_csv4_l))
data_l.append(pd.read_csv(path_csv5_l))

losses_l = []
eval_losses_l = []

# for i in data[2]['adam预测值']:
#     print(i)
#     # print('\n')
for i in range(2):
    losses_l.append([i for i in data_l[i]['训练集loss']])
    eval_losses_l.append([i for i in data_l[i]['测试集loss']])

x_l = list(range(len(losses_l[0])))
# x = [i+1 for i in x]
# 画不同梯度下降法方法，迭代次数和MSE的关系折线图
l1_l = (
    Line()
    .add_xaxis(x_l[1:100])
    # .add_yaxis("SGD", losses[0][50:100],label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("Adam", losses_l[0][1:100],label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("RMSprop", losses_l[1][1:100],label_opts=opts.LabelOpts(is_show=False))
    # .add_yaxis("商家A", losses[1])
    # .add_yaxis("商家A", losses[2])
    .set_global_opts(title_opts=opts.TitleOpts(title="训练集",subtitle='最低工资'))
    # .render("line_bp_low1.html")
)
# 画adam梯度下降法方法，预测值和真实值的折线图
l2_l = (
    Line()
    .add_xaxis(x_l[1:100])
    # .add_yaxis("SGD", losses[0][50:100],label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("Adam", eval_losses_l[0][1:100],label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("RMSprop", eval_losses_l[1][1:100],label_opts=opts.LabelOpts(is_show=False))
    # .add_yaxis("商家A", losses[1])
    # .add_yaxis("商家A", losses[2])
    .set_global_opts(title_opts=opts.TitleOpts(title="测试集",subtitle='最低工资'))
    # .render("line_bp_low2.html")
)
id_1_l = list(range(len(data_l[2]['adam真实值'])))
l3_l = (
    Line()
    .add_xaxis(xaxis_data=id_1_l)
    # .add_xaxis(xaxis_data=timeData)
    .add_yaxis(
        series_name="实际最低工资",
        y_axis=[i for i in data_l[2]['adam真实值']],
        # y_axis=water_flowData,
        # symbol_size=8,
        is_hover_animation=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=1.5),
        is_smooth=True,
    )
    .add_yaxis(
        series_name="预测最低工资",
        y_axis=[i for i in data_l[2]['adam预测值']],
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
    # .render("line_bp_low3.html")
)

id_2_l = list(range(len(data_l[3]['adam真实值'])))
l4_l = (
    Line()
    .add_xaxis(xaxis_data=id_2_l)
    # .add_xaxis(xaxis_data=timeData)
    .add_yaxis(
        series_name="实际最低工资",
        y_axis=[i for i in data_l[3]['adam真实值']],
        # y_axis=water_flowData,
        # symbol_size=8,
        is_hover_animation=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(width=1.5),
        is_smooth=True,
    )
    .add_yaxis(
        series_name="预测最低工资",
        y_axis=[i for i in data_l[3]['adam预测值']],
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
    # .render("line_bp_low4.html")
)

#获得可布局的网络页面
# page = Page(layout=Page.DraggablePageLayout)
# page.add(l1,l2,l3,l4,l1_l,l2_l,l3_l,l4_l)
# page.render("bp_all.html")

#渲染布局以后的网络页面
Page.save_resize_html("bp_all.html", cfg_file="D:\\pythonProject_al\\chart_config5.json", dest="bp_all2.html")


