import pandas as pd
import operator
import copy

from pyecharts.faker import Faker
import pyecharts.options as opts
from pyecharts.globals import ThemeType
from pyecharts.commons.utils import JsCode
from pyecharts.charts import Timeline, Grid, Bar, Map, Pie, Radar,Page


# 将省会转成省名的字典
capitals = {'湖南': '长沙', '湖北': '武汉', '广东': '深圳', '广西': '南宁', '河北': '石家庄', '河南': '郑州', '山东': '济南', '山西': '太原', '江苏': "苏州",
            '浙江': '杭州', '江西': '南昌', '黑龙江': '哈尔滨', '新疆': '乌鲁木齐', '云南': '昆明', '贵州': '贵阳', '福建': '厦门', '吉林': '长春',
            '安徽': '合肥', '四川': '成都', '西藏': '拉萨', '宁夏': '银川', '辽宁': '沈阳', '青海': '西宁', '海南': '海口', '甘肃': '兰州', '陕西': '西安',
            '内蒙古': '呼和浩特', '台湾': '台北', '北京': '北京', '上海': '上海', '天津': '天津', '重庆': '重庆', '香港': '香港', '澳门': '澳门'}

pie_data1 = [[], [], []]


# 根据字典的值value获得该值对应的key
def get_dict_key(dic, value):
    key = list(dic.keys())[list(dic.values()).index(value)]
    return key


# def get_year_chart(year: int):
######################
# 数据预处理
#####################
# 经k-means处理后的数据
path_csv = 'D:\\pythonProject_al\\data\\kmeans.csv'
# data=pd.read_csv(path_csv,usecols=[1,2])

data_2 = pd.read_csv(path_csv)
# 地图和柱状图工资数据
dic2 = {}

for i in range(len(data_2)):
    dic2[data_2['市'][i]] = data_2['聚类簇号'][i]

m = [('湖南', 2), ('天津', 2), ('福建', 2), ('北京', 3), ('江苏', 1), ('河南', 1), ('陕西', 4), ('浙江', 4), ('上海', 3), ('广东', 4),
     ('湖北', 3), ('重庆', 2), ('四川', 3)]
data_province = [(get_dict_key(capitals, k), v + 1) for k, v in dic2.items()]
# data_province=sorted(data_province.items(),key=lambda x:x[1])
print(data_province)
print(m)
######################
# 画图表现出不同地区所属的簇
#####################
map_chart = (
    Map()
        .add(
        "",
        # data_province,
        m,
        "china",
        label_opts=opts.LabelOpts(is_show=False),
    )
        .set_global_opts(
        title_opts=opts.TitleOpts(title="JAVA工作福利"),
        visualmap_opts=opts.VisualMapOpts(
            is_calculable=True,
            dimension=0,
            pos_left="10",
            pos_top="center",
            range_text=["High", "Low"],
            range_color=["lightskyblue", "yellow", "orangered"],
            textstyle_opts=opts.TextStyleOpts(color="#ddd"),
            min_=4,
            max_=1,
        ),
    )
)

##############################
# 雷达图
##############################

path_csv2 = 'D:\\pythonProject_al\\data\\test2.csv'
# data=pd.read_csv(path_csv,usecols=[1,2])
# data_1 = pd.read_csv(path_csv)
data_1 = pd.read_csv(path_csv2)
# 地图和柱状图工资数据
dic1 = {}
dic_count1 = {}
dic_graph1 = {}
for i in range(len(data_1)):
    city = data_1["市"][i]
    if city in dic1.keys():
        dic1[city] = dic1[city] + data_1["月薪"][i]
        dic_count1[city] = dic_count1[city] + 1
        # dic_graph[city] =
    else:
        dic1[city] = data_1["月薪"][i]
        dic_count1[city] = 1

for i in dic_count1.keys():
    if dic_count1[i] < 30:
        dic1.pop(i)
    else:
        dic1[i] = int(dic1[i] / dic_count1[i] * 1000)
data_radio = {}
welfare_seek = ['五险一金', '节日福利', '带薪年假', '年终奖', '定期体检', '员工旅游']
for city in dic1.keys():
    # '五险一金''节日福利''带薪年假'年终奖'定期体检''员工旅游'
    data_radio[city] = [0, 0, 0, 0, 0, 0]
for i in range(len(data_1)):
    for j in eval(data_1["福利"][i]):
        if data_1["市"][i] in dic1.keys():
            if j in welfare_seek:
                data_radio[data_1["市"][i]][welfare_seek.index(j)] = data_radio[data_1["市"][i]][
                                                                        welfare_seek.index(j)] + 1

for city in data_radio.keys():
    for i in range(6):
        data_radio[city][i] = data_radio[city][i] / dic_count1[city]
print(data_radio)
print('\n')
print(data_radio[capitals['湖南']])

c_schema = []
for i in welfare_seek:
    c_schema.append({"name": i, "max": 1, "min": 0})
c1 = (
    Radar(init_opts=opts.InitOpts(width="580px", height="540px"))
    # Radar(init_opts=opts.InitOpts(width="580px", height="540px",theme=ThemeType.DARK))

        .add('湖南', [data_radio[capitals['湖南']]], color="#A9A9A9")
        .add('天津', [data_radio[capitals['天津']]], color="#FFEBCD")
        .add('福建', [data_radio[capitals['福建']]], color="#D2691E")
        .add('重庆', [data_radio[capitals['重庆']]], color="#b3e4a1")
        .add_schema(schema=c_schema, shape="circle")
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title=""))
    # .render("radar2.html")
)

c2 = (
    Radar(init_opts=opts.InitOpts(width="580px", height="540px"))
    # Radar(init_opts=opts.InitOpts(width="580px", height="540px",theme=ThemeType.DARK))
        .add('江苏', [data_radio[capitals['江苏']]], color="#A9A9A9")
        .add('河南', [data_radio[capitals['河南']]], color="#D2691E")

        .add_schema(schema=c_schema, shape="circle")
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title=""))
    # .render("radar1.html")
)

c3 = (
    Radar(init_opts=opts.InitOpts(width="580px", height="540px"))
    # Radar(init_opts=opts.InitOpts(width="580px", height="540px",theme=ThemeType.DARK))
        .add('北京', [data_radio[capitals['北京']]], color="#A9A9A9")
        .add('上海', [data_radio[capitals['上海']]], color="#FFEBCD")
        .add('湖北', [data_radio[capitals['湖北']]], color="#D2691E")
        .add('四川', [data_radio[capitals['四川']]], color="#D2691E")

        .add_schema(schema=c_schema, shape="circle")
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title=""))
    # .render("radar3.html")
)

c4 = (
    Radar(init_opts=opts.InitOpts(width="580px", height="540px"))
    # Radar(init_opts=opts.InitOpts(width="580px", height="540px",theme=ThemeType.DARK))
        .add('陕西', [data_radio[capitals['陕西']]], color="#A9A9A9")
        .add('浙江', [data_radio[capitals['浙江']]], color="#FFEBCD")
        .add('广东', [data_radio[capitals['广东']]], color="#D2691E")

        .add_schema(schema=c_schema, shape="circle")
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title=""))
    # .render("radar4.html")
)

bar = (
    Bar()
        .add_xaxis(Faker.choose())
        .add_yaxis("Label A", Faker.values())
        .add_yaxis("Label B", Faker.values())
        .set_global_opts(title_opts=opts.TitleOpts(title="组合：柱状图部分"))
)

grid_chart = (
    Grid(init_opts=opts.InitOpts(width="1500px", height="800px"))
    .add(map_chart,
         grid_opts=opts.GridOpts(
            # pos_left="10", pos_right="70%", pos_top="70%", pos_bottom="5"
         ),
    )
)
# return grid_chart
grid_chart.render("kmeans_radar.html")

#生成可布局网页
# page = Page(layout=Page.DraggablePageLayout)
# page.add(grid_chart,c4,c3,c2,c1)
# page.render("radar_all.html")

#生成布局之后的网页
Page.save_resize_html("radar_all.html", cfg_file="D:\\pythonProject_al\\chart_config.json", dest="radar_all2.html")



