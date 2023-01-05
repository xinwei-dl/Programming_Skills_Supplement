#!/usr/bin/env python
# coding: utf-8

import folium
from folium.plugins import HeatMap
from osgeo import ogr, osr
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import jieba
import re

data = pd.read_excel(r"./data/covidloc.xlsx", usecols=[4, 5], header=None)
order = [5, 4]
data = data[order]
data['new_col'] = 1

print(data)
data = data.values
print(data)
m = folium.Map([35, 110], zoom_start=5)
HeatMap(data).add_to(m)

m.save("./result/Mapa_Folium_Tasa_Paro_Provincias.html")

# excel to shapefile

shp_fn = r'./data/covidloc.shp'
driver = ogr.GetDriverByName('ESRI Shapefile')
shp_ds = ogr.Driver.CreateDataSource(driver, shp_fn)

sr = osr.SpatialReference(osr.SRS_WKT_WGS84_LAT_LONG)
shp_lyr = shp_ds.CreateLayer('typhoon', sr, ogr.wkbPoint)
shp_row = ogr.Feature(shp_lyr.GetLayerDefn())
data = pd.read_excel(r"./data/covidloc.xlsx", usecols=[4, 5], header=None)
data = data.values
for row in data:
    shp_pt = ogr.Geometry(ogr.wkbPoint)
    shp_pt.AddPoint(row[0], row[1])
    shp_row.SetGeometry(shp_pt)
    shp_lyr.CreateFeature(shp_row)
del shp_ds


plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

k = 0
faceco1or = [0.9375, 0.9375, 0.859375]
r = 1
i = 0
listk = []  # The number of points that fall into the plane
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.LambertConformal(central_latitude=90, central_longitude=105))
ax.set_extent([70, 140.5, 0, 55], ccrs.Geodetic())
border_shp = shpreader.Reader(r"./data/课前准备/data/CHINA_国界线p.shp")
ax.add_geometries(border_shp.geometries(), crs=ccrs.PlateCarree(), edgecolor='black', linewidths=0.5, facecolor='none')
ax.annotate('2022年5月4日晚新浪微博疫情相关舆情空间统计', xy=(0.28, 0.95), rotation=0, fontsize=10, xycoords='axes fraction')
for state in shpreader.Reader(r"./data/课前准备/data/districtgeo.shp").geometries():
    faceco1or = [0.9375, 0, 0]
    edgecolor = 'black'
    for point in shpreader.Reader(r"./data/covidloc.shp").geometries():
        if state.intersects(point):  # intersect
            k = k + 1
    listk.append(k)
    k = 0
print(listk)
cmap = mpl.cm.Blues
for state in shpreader.Reader(r"./data/课前准备/data/districtgeo.shp").geometries():
    a = listk[i] * 10  # Because the number of points is small, it is enlarged by 10 letters to enhance the display effect
    i = i + 1
    ax.add_geometries([state], ccrs.PlateCarree(), facecolor=cmap(a, 1), edgecolor=edgecolor)
plt.savefig('./result/spatialstatics', dpi=300,bbox_inches='tight')
plt.show()



plt.figure(figsize=(5, 6.5))
ax = plt.axes([0., 0., 1., 1.], projection=ccrs.PlateCarree())
ax.set_extent([70, 140.5, 15, 55], ccrs.Geodetic())
ax.annotate('2022年5月4日晚新浪微博疫情相关舆情K-means聚类分析', xy=(0.18, 0.92), rotation=0, fontsize=10, xycoords='axes fraction')
# ax=fig.add_axes([0,0,1,1],projection=ccrs.LambertConformal(central_latitude=90,central_longitude=105))
states_shp = shpreader.Reader(r"./data/课前准备/data/districtgeo.shp")
ax.add_geometries(states_shp.geometries(), ccrs.Geodetic(),
                  facecolor='none', edgecolor=edgecolor)
data = pd.read_excel(r"./data/covidloc.xlsx", usecols=[4, 5], header=None)
text = pd.read_excel(r"./data/covidloc.xlsx", usecols=[2], header=None)
data = data.values
# print(data)
estimator = KMeans(n_clusters=3)  # Construct a suitable clusterer, construct a suitable - number of clusters is 3 class limiter
estimator.fit(data)
label_pred = estimator.labels_  # Get the cluster labels
centroids = estimator.cluster_centers_  # Get the cluster centers
inertia = estimator.inertia_  # Obtain the sum of the clustering criteria
mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  # Here '0 'represents the '0' in the circle, 'wide' represents the pre-color red, and so on
j = 0
class1 = ''
class2 = ''
class3 = ''
for i in label_pred:
    plt.plot(data[j:j + 1, 0], data[j:j + 1, 1], mark[i], markersize=5)
    if i == 2:
        class3 = class3 + str(text.values[j])
        # print(str(text.values[j]))
    if i == 0:
        class1 = class1 + str(text.values[j])
    j += 1
plt.savefig('./result/kmeans', dpi=300,bbox_inches='tight')
plt.show()




def removePunctuation(text):  # Remove special symbols
    punctuation = '!,;:?"#【)\·"。:@/.1234567890abicdefjgu了展开全文的'
    text = re.sub(r'[{}]+'.format(punctuation), '', text)
    return text.strip().lower()


def wordclous(text):  # Word cloud rendering
    wordlist_after_jieba = jieba.cut(removePunctuation(str(text)), cut_all=True)
    wl_space_split = " ".join(wordlist_after_jieba)
    my_wordcloud = WordCloud(background_color="white", max_words=50, width=1000, height=860, collocations=False,
                             font_path=r'./data/课前准备/data/方正兰亭细黑_GBK.TTF').generate(wl_space_split)

    plt.imshow(my_wordcloud)
    plt.axis("off")
    plt.savefig('./result/cloud', dpi=300, bbox_inches='tight')
    plt.show()


wordclous(class3)
wordclous(class1)

