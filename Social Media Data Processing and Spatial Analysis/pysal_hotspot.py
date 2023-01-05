#!/usr/bin/env python
# coding: utf-8

from esda.getisord import G_Local
import geopandas as gpd
import libpysal
import esda

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import cartopy.io.shapereader as shpreader
import matplotlib.patches as patches

# Calculate the global Moran index
sh = gpd.read_file(r'.\data\课前准备\data\sentiment_cityp.shp')
print(sh['sentiment'])
wq = libpysal.weights.Queen.from_shapefile('.\data\课前准备\data\sentiment_cityp.shp',
                                           "id")  # Calculate the spatial weight matrix
print(wq)
moran = esda.Moran(sh['sentiment'], wq)
moran.I

w = libpysal.weights.DistanceBand.from_shapefile('.\data\课前准备\data\sentiment_cityp.shp', 5,
                                                 "id")  # Calculation of spatial weight matrix The larger the threshold, the larger the range of hotspots
lg = G_Local(sh['sentiment'], w, transform='B')
Moran_Local = esda.moran.Moran_Local(sh['sentiment'], w)  # Local moran
print(Moran_Local.z_sim)  # Morcan I for each value
print(Moran_Local.p_sim)  # Confidence probability value

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

proj = ccrs.LambertConformal(central_latitude=90, central_longitude=105)
# Set figure size
fig = plt.figure(figsize=[12, 12])
# Set projection and plot the main figure
ax = plt.axes(projection=proj)
# Add ocean, land, rivers and lakes
ax.add_feature(cfeature.OCEAN.with_scale('50m'))
ax.add_feature(cfeature.LAND.with_scale('50m'))
ax.add_feature(cfeature.RIVERS.with_scale('50m'))
ax.add_feature(cfeature.LAKES.with_scale('50m'))

# Plot gridlines
ax.gridlines(linestyle='--')
# Set figure extent
ax.set_extent([60, 150, 0, 55])

sentiment_city_shp = shpreader.Reader(r"D.\data\课前准备\data\sentiment_cityp.shp")
liskp = []
i = 0
listz = lg.Zs  # Morcan I for each value

liskp = lg.p_norm / 2
# listq=Moran_Local.q
# Render the surface according to the sentiment value
cmap = mpl.cm.Blues
ax.add_geometries(sentiment_city_shp.geometries(), crs=ccrs.PlateCarree(), edgecolor='gray', linewidths=0.5,
                  facecolor='#DCDCDC')
plt.title('热点分析', fontsize=16)
for state in sentiment_city_shp.geometries():
    # pick a default color for the land with a black outline,
    z = listz[i]
    p = liskp[i]  # Hot spot：Z>0 and p<0.05cold point：Z<0 and p<0.05not significant:p>0.05

    if z > 0 and p < 0.05:
        ax.add_geometries([state], ccrs.PlateCarree(), facecolor='Pink')
    if z < 0 and p < 0.05:
        ax.add_geometries([state], ccrs.PlateCarree(), facecolor='Blue')
    if p >= 0.05:
        ax.add_geometries([state], ccrs.PlateCarree(), facecolor='#DCDCDC')
    if z == 0:
        ax.add_geometries([state], ccrs.PlateCarree(), facecolor='#DCDCDC')
    i = i + 1

labels = ['hot spot', 'cold spot', 'not sig']

h = patches.Rectangle((0, 0), 1, 1, facecolor='Pink')
c = patches.Rectangle((0, 0), 1, 1, facecolor='Blue')
n = patches.Rectangle((0, 0), 1, 1, facecolor='#DCDCDC')
ax.legend([h, c, n], labels, loc='lower left')

plt.savefig('./result/hotpoint', dpi=300, bbox_inches='tight')
plt.show()
