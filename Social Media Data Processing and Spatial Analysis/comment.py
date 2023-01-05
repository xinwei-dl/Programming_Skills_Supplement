import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# 检查-网络-Fetch/XHR
def fetchUrl(pid, uid, max_id):
    url = "https://weibo.com/ajax/statuses/buildComments"
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36",
    }
    params = {
        "flow" : 0,
        "is_reload" : 1,
        "id" : pid,
        "is_show_bulletin" : 2,
        "is_mix" : 0,
        "max_id" : max_id,
        "count" : 20,
        "uid" : uid,
    }
    r = requests.get(url, headers = headers, params = params)
    return r.json()

def parseJson(jsonObj):
    data = jsonObj["data"]
    max_id = jsonObj["max_id"]
    commentData = []
    for item in data:
        # id
        comment_Id = item["id"]
        # 内容content
        content = BeautifulSoup(item["text"], "html.parser").text
        # 时间time
        created_at = item["created_at"]
        # 点赞数likes
        like_counts = item["like_counts"]
        # 评论数comments
        total_number = item["total_number"]
        # 定位地点location
        source=item["source"][2:]
        # id，name，city
        user = item["user"]

        userID = user["id"]
        userName = user["name"]
        userCity = user["location"]
        #Get the latitude and longitude of the location
        try:
            addinfo = getlocation(source)
            lon = addinfo[0]['lng']
            lat = addinfo[0]['lat']
        except:
            source = '其他'
            lon = 0
            lat = 0
        dataItem = [comment_Id, created_at, userID, userName, source, lat, lon, like_counts, total_number, content]
        print(dataItem)
        commentData.append(dataItem)
    return commentData, max_id

def save_data(data, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(path + filename, encoding='utf_8_sig', mode='a', index=False, sep=',', header=False)

# Get latitude and longitude from Baidu API
def getlocation(location):
    ak = 'lweZOODTGGm0pIpggImnC5vu79orQwUR'
    url = "http://api.map.baidu.com/geocoding/v3/"+'?address='+location+'&output=json&ak='+ak
    header = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'}
    payload ={'output':'json', 'ak':ak}
    addinfo = []
    payload['address'] = location
    try:
        content = requests.get(url, params=payload, headers=header).json()
        # print(content)
        addinfo.append(content['result']['location'])
    except:
        pass
    return(addinfo)


if __name__ == "__main__":
    pid = 4767986391716451      # weibo id
    uid = 1711530911            # user id
    max_id = 0
    path = "./result/"           # The path to save the file
    filename = "shanghai.csv"   # file name
    csvHeader = [["评论id", "发布时间", "⽤户id", "⽤户昵称", "定位地址", "纬度","经度","点赞数", "回复数", "评论内容"]]
    save_data(csvHeader, path, filename)
    while(True):
        html = fetchUrl(pid, uid, max_id)
        comments, max_id = parseJson(html)
        save_data(comments, path, filename)
        # When max_id is 0, finished
        if max_id == 0:
            break
