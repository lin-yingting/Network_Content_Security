
import pandas as pd
import operator
import copy

import pyecharts.options as opts
from pyecharts.globals import ThemeType
from pyecharts.commons.utils import JsCode
from pyecharts.charts import Timeline, Grid, Bar, Map, Pie,Radar


# 将省会转成省的字典
capitals = {'湖南':'长沙','湖北':'武汉','广东':'深圳','广西':'南宁','河北':'石家庄','河南':'郑州','山东':'济南','山西':'太原','江苏':"苏州",'浙江':'杭州','江西':'南昌','黑龙江':'哈尔滨','新疆':'乌鲁木齐','云南':'昆明','贵州':'贵阳','福建':'厦门','吉林':'长春','安徽':'合肥','四川':'成都','西藏':'拉萨','宁夏':'银川','辽宁':'沈阳','青海':'西宁','海南':'海口','甘肃':'兰州','陕西':'西安','内蒙古':'呼和浩特','台湾':'台北','北京':'北京','上海':'上海','天津':'天津','重庆':'重庆','香港':'香港','澳门':'澳门'}

pie_data1 = [[],[],[]]


# 根据字典的值value获得该值对应的key
def get_dict_key(dic, value):
    key = list(dic.keys())[list(dic.values()).index(value)]
    return key

def get_year_chart(year: int):
######################
# 数据预处理
#####################
    path_csv='D:\\pythonProject_al\\data\\test'+str(year)+'.csv'
    # data=pd.read_csv(path_csv,usecols=[1,2])
    data_1 = pd.read_csv(path_csv)
    data_2 = pd.read_csv(path_csv)
    # 地图和柱状图工资数据
    dic = {}
    dic_count = {}
    dic_graph = {}
    for i in range(len(data_1)):
        city = data_1["市"][i]
        if city in dic.keys():
            dic[city] = dic[city]+data_1["月薪"][i]
            dic_count[city] = dic_count[city]+1
            # dic_graph[city] =
        else:
            dic[city] = data_1["月薪"][i]
            dic_count[city] = 1

    for i in dic_count.keys():
        if dic_count[i]<30:
            dic.pop(i)
        else:
            dic[i] = int(dic[i]/dic_count[i]*1000)
    print(dic)
    min_data, max_data = (
        min(dic.values()),
        max(dic.values()),
    )
    # dic=sorted(dic.items(),key=lambda x:x[1])
    # dic=sorted(dic.items(),key=operator.itemgetter(1))
    data_city = [(k,v) for k,v in dic.items()]
    data_province=[(get_dict_key(capitals, k),v) for k,v in dic.items()]
    # data_province=sorted(data_province.items(),key=lambda x:x[1])
    print(data_province)
######################
# 画图
#####################
# 在地图上画出不同省市的平均薪资所代表的颜色
    map_chart = (
        Map()
        .add(
            "",
            data_province,
            "china",
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="中国市级地图"),
            visualmap_opts=opts.VisualMapOpts(
                is_calculable=True,
                dimension=0,
                pos_left="10",
                pos_top="center",
                range_text=["High", "Low"],
                range_color=["lightskyblue", "yellow", "orangered"],
                textstyle_opts=opts.TextStyleOpts(color="#ddd"),
                min_=min_data,
                max_=max_data,
            ),
        )
    )
    bar_x_data = [x[0] for x in data_province]
    test_data_1=sorted(data_province,key=lambda x:x[0])
    bar_y_data = [{"name": x[0], "value": x[1]} for x in test_data_1]

    # bar_y_data = [{"name": x[0], "value": x[1]} for x in data_province]
    print(bar_y_data)
    print(len(bar_y_data))
# 画出不同地区平均薪资的柱状图
    bar = (
        Bar()
        .add_xaxis(xaxis_data=bar_x_data)
        .add_yaxis(
            series_name="",
            yaxis_index=1,
            y_axis=bar_y_data,
            # yaxis_data=bar_y_data,
            label_opts=opts.LabelOpts(
                is_show=True, position="right", formatter="{b}: {c}"
            ),
        )
        .reversal_axis()
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=False)),
            yaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(is_show=False)),
            tooltip_opts=opts.TooltipOpts(is_show=False),
            visualmap_opts=opts.VisualMapOpts(
                is_calculable=True,
                dimension=0,
                pos_left="10",
                pos_top="center",
                range_text=["High", "Low"],
                range_color=["lightskyblue", "yellow", "orangered"],
                textstyle_opts=opts.TextStyleOpts(color="#ddd"),
                min_=min_data,
                max_=max_data,
            ),
            graphic_opts=[
                opts.GraphicGroup(
                    graphic_item=opts.GraphicItem(
                        rotation=JsCode("Math.PI / 4"),
                        bounding="raw",
                        right=110,
                        bottom=110,
                        z=100,
                    ),
                    children=[
                        opts.GraphicRect(
                            graphic_item=opts.GraphicItem(left="center", top="center", z=100),
                            graphic_shape_opts=opts.GraphicShapeOpts(width=400, height=50),
                            graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(
                                fill="rgba(0,0,0,0.3)"
                            ),
                        ),
                        opts.GraphicText(
                            graphic_item=opts.GraphicItem(left="center", top="center", z=100),
                            graphic_textstyle_opts=opts.GraphicTextStyleOpts(
                                text=name_list[y-1],
                                font="bold 26px Microsoft YaHei",
                                graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(fill="#fff"),
                            ),
                        ),
                    ],
                )
            ],
        )
    )

# 将柱状图和地图合在一个元素中
    grid_chart = (
        Grid()
        .add(
            bar,
            grid_opts=opts.GridOpts(
                pos_left="10", pos_right="70%", pos_top="70%", pos_bottom="5"
            ),
        )
        # .add(pie,grid_opts=opts.GridOpts(),)
        .add(map_chart, grid_opts=opts.GridOpts())
    )

    return grid_chart

# 将不同编程语言的平均薪资与地区的关系都画出来
time_list = [1,2,3]
name_list = ['python','java','c/c++']
timeline = Timeline(
    # init_opts=opts.InitOpts(theme=ThemeType.DARK)
    init_opts=opts.InitOpts(width="1550px", height="800px", theme=ThemeType.DARK)
)
for y in time_list:
    g = get_year_chart(year=y)
    timeline.add(g, time_point=name_list[y-1])
    # timeline.add(g, time_point=str(y))

timeline.add_schema(
    orient="vertical",
    is_auto_play=True,
    is_inverse=True,
    play_interval=5000,
    pos_left="null",
    pos_right="5",
    pos_top="20",
    pos_bottom="20",
    width="50",
    label_opts=opts.LabelOpts(is_show=True, color="#fff"),
)

#渲染网页
timeline.render("work1.html")


