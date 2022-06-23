from pyecharts import options as opts
from pyecharts.charts import Boxplot
import pandas as pd
from pyecharts.charts import WordCloud,Pie,Page
from pyecharts.globals import SymbolType
import copy


#数据预处理
v = [[],[],[]]
cloud = []
dic_cloud = {}
cloud_mid = []
dic_pie = {}
dic_pie_mid = {}
pie_data = []
mid = []
name_list = ['python','java','c/c++']
for i in range(3):
    dic_cloud = {}
    dic_pie = {}
    path_csv = 'D:\\pythonProject_al\\data\\test' + str(i+1) + '.csv'
    # data=pd.read_csv(path_csv,usecols=[1,2])
    data_1 = pd.read_csv(path_csv)
    mid = [i for i in data_1['月薪']]
    v[i] = mid
    for i in range(len(data_1)):
        for word in eval(data_1['工作能力'][i]):
            if word in dic_cloud.keys():
                dic_cloud[word] = dic_cloud[word]+1
            else:
                dic_cloud[word] = 1
        if data_1['公司行业'][i] in dic_pie.keys():
            dic_pie[data_1['公司行业'][i]] = dic_pie[data_1['公司行业'][i]]+1
        else:
            dic_pie[data_1['公司行业'][i]] = 1

    print(dic_cloud)
    cloud_mid = [(k,v) for k,v in dic_cloud.items()]
    print(cloud_mid)
    cloud.append(cloud_mid)
    dic_pie['其他行业'] = 0
    dic_pie_mid = copy.deepcopy(dic_pie)
    for i in dic_pie_mid.keys():
        if i == '其他行业':
            continue
        if dic_pie[i] <100:
            dic_pie['其他行业'] = dic_pie['其他行业']+dic_pie[i]
            dic_pie.pop(i)

    pie_data.append([[k,v] for k,v in dic_pie.items()])

print(v[0])


####################################
# 画出不同编程语言岗薪资的箱型图
####################################
c = Boxplot()
c.add_xaxis(["编程语言"])
c.add_yaxis(name_list[0], c.prepare_data([v[0]]))
c.add_yaxis(name_list[1], c.prepare_data([v[1]]))
c.add_yaxis(name_list[2], c.prepare_data([v[2]]))
# c.add_yaxis("B", c.prepare_data(v2))
c.set_global_opts(title_opts=opts.TitleOpts(title="薪资对比"))
# c.render("boxplot_base.html")


####################################
# 画出不同编程语言岗所涉及到的工作能力云图
####################################
d0 = (
    WordCloud()
    .add("", cloud[0], word_size_range=[20, 100], shape=SymbolType.DIAMOND)
    .set_global_opts(title_opts=opts.TitleOpts(title=name_list[0]))
    # .render("wordcloud_diamond.html")
)
d1 = (
    WordCloud()
    .add("", cloud[1], word_size_range=[20, 100], shape=SymbolType.DIAMOND)
    .set_global_opts(title_opts=opts.TitleOpts(title=name_list[1]))
    # .render("wordcloud_diamond.html")
)
d2 = (
    WordCloud()
    .add("", cloud[2], word_size_range=[20, 100], shape=SymbolType.DIAMOND)
    .set_global_opts(title_opts=opts.TitleOpts(title=name_list[2]))
    # .render("wordcloud_diamond.html")
)

#################################
# 画出不同编程语言岗所涉及到的行业饼图
#################################
p1 = (
    Pie()
    # Pie(init_opts=opts.InitOpts(width="1600px", height="800px", bg_color="#2c343c"))
    .add(
        series_name="访问来源",
        data_pair=pie_data[0],
    )
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=name_list[0],
            pos_left="center",
            pos_top="20",
            # title_textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        legend_opts=opts.LegendOpts(is_show=False),
    )
    # .render("pie1.html")
)

p2 = (
    Pie()
    # Pie(init_opts=opts.InitOpts(width="1600px", height="800px", bg_color="#2c343c"))
    .add(
        series_name="访问来源",
        data_pair=pie_data[1],
    )
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=name_list[1],
            pos_left="center",
            pos_top="20",
            # title_textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        legend_opts=opts.LegendOpts(is_show=False),
    )
    # .render("pie2.html")
)

p3 = (
    Pie()
    # Pie(init_opts=opts.InitOpts(width="1600px", height="800px", bg_color="#2c343c"))
    .add(
        series_name="访问来源",
        data_pair=pie_data[2],
        label_opts=opts.LabelOpts(is_show=False, position="center")
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=name_list[2],
            pos_left="center",
            pos_top="20",
            # title_textstyle_opts=opts.TextStyleOpts(color="#fff"),
        ),
        legend_opts=opts.LegendOpts(is_show=False),
    )
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    # .set_global_opts(title_opts=opts.TitleOpts(title=name_list[2]))
    # .render("pie3.html")
)

#生成可布局页面
# page = Page(layout=Page.DraggablePageLayout)
# page.add(c,d0,d1,d2,p1,p2,p3)
# page.render("compare_all.html")

#生成布局之后的页面
Page.save_resize_html("compare_all.html", cfg_file="D:\\pythonProject_al\\chart_config2.json", dest="compare_all2.html")


