# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:47:51 2017

@author: 5558
"""

#import numpy as np
from sentinelsat import SentinelAPI,read_geojson,geojson_to_wkt
import os 
#from geojson import Point
from geojson import Polygon
import geojson
import tempfile
import datetime as dt 

api = SentinelAPI('meet.saparia','9909404659')
#coordinates = [(76.58234065478575,17.25958245029804),(76.94821117999994,17.089067089549573),(77.21592132040054,17.50654956833391),(76.74296673902613,17.617150362421967),(76.58234065478575,17.25958245029804),(76.58234065478575,17.25958245029804)]
#POLYGON())" ) AND ( beginPosition:[2017-11-11T00:00:00.000Z TO 2017-12-11T23:59:59.999Z] AND endPosition:[2017-11-11T00:00:00.000Z TO 2017-12-11T23:59:59.999Z] )

coordinates = [(76.89941158791933,17.223384891246383),(77.00050005239353,17.223384891246383),(77.00050005239353,17.276782301151),(76.89941158791933,17.276782301151),(76.89941158791933,17.223384891246383)]
#start_date = dt.date(int("2017"),int("09"),int("27"))
#end_date = dt.date(int("2017"),int("11"),int("27"))

 #footprint:"Intersects(POLYGON())") AND ( beginPosition:[2017-09-27T00:00:00.000Z TO 2017-11-27T23:59:59.999Z] AND endPosition:[2017-09-27T00:00:00.000Z TO 2017-11-27T23:59:59.999Z] ) AND (platformname:Sentinel-2)


#POLYGON())") AND ( beginPosition:[2017-09-27T00:00:00.000Z TO 2017-11-27T23:59:59.999Z] AND endPosition:[2017-09-27T00:00:00.000Z TO 2017-11-27T23:59:59.999Z] ) 



def product_list(coordinates,start_date,end_date) :
    polygon_geojson = Polygon([coordinates])
    tmp_file = tempfile.mkstemp(suffix='.geojson')
    with open(tmp_file[1], 'w') as outfile:        
        geojson.dump(polygon_geojson, outfile)
    footprint = geojson_to_wkt(read_geojson(tmp_file[1]))
    products = api.query(footprint,date=("%sT00:00:00.000Z"%(start_date),"%sT23:59:59.999Z"%(end_date)),producttype='S2MSI1C')
    return products

def download(key,start_date,end_date):
    path = 'F:/Agriculture/satellite_images/%s_%s'%(start_date,end_date)
    if not os.path.exists(path):
        os.makedirs(path)
    api.download(key,directory_path=path)
    

total_images = int("6")
monthly_gap = int("1")
count = int("1")
#title = 'S2A_MSIL1C_20171123T052131_N0206_R062_T43QFV_20171123T091125'
title = 'S2A_MSIL1C_20171123T052131_N0206_R062_T43QFV_20171123T091125'

def key_generator(title,products) :
    list_it = [(value, key) for key, value in products.items()]
    item_tuple = [items for items in list_it if items[0]['title']==title ]
    return item_tuple

    

while (count <= total_images-monthly_gap+1) :
    if count == 1 :
        start_month = int("11")
        end_month = int("12")
        start_date = dt.date(int("2017"),start_month,int("11"))
        end_date = dt.date(int("2017"),end_month,int("11"))
        products = product_list(coordinates,start_date,end_date)
        key_tuple = key_generator(title,products)
        tuple_conv = [(value, key) for key, value in key_tuple]
        conv = tuple_conv.pop()
        final_key = conv[0]
        item_dict = conv[1]['tileid']
        item_pltid = conv[1]['platformserialidentifier']
        item_title = conv[1]['title']
        item_title_wrd = item_title.split("_")
        bfr_tid = item_title_wrd[4]
        if not os.path.exists('F:/Agriculture/satellite_images/%s_%s/%s.zip'%(start_date,end_date,item_title)):
            download(final_key,start_date,end_date)
        count = count + 1
    else :
        end_month = start_month
        start_month = end_month - monthly_gap
        start_date = dt.date(int("2017"),start_month,int("11"))
        end_date = dt.date(int("2017"),end_month,int("11"))
        products = product_list(coordinates,start_date,end_date)
        total_len = len(products) 
        a_dict = {}
        while (total_len > 0) :
            item1 = products.popitem()
            item_key = item1[1]['uuid']
            item_title1 = item1[1]['title']
            try : 
                item_dict1 = item1[1]['tileid']
            except :
                SyntaxError 
            else:
                item_dict1 = item_title1.split('_')[5].split('T')[1]
            item_pltid1 = item1[1]['platformserialidentifier']
            item_title1_wrd = item_title1.split("_")
            bfr_tid1 = item_title1_wrd[4]
            itm_date1 = item_title1_wrd[6]
            int_date = int(item_title1_wrd[6].split("T")[0])
            if (item_dict1 == item_dict) & (item_pltid == item_pltid1) & (bfr_tid1==bfr_tid) :
                a_dict[item_key] = int_date
            total_len = total_len - 1
        if len(a_dict) == 1 :
            if not os.path.exists('F:/Agriculture/satellite_images/%s_%s/%s.zip'%(start_date,end_date,item_title1)):
                download(a_dict.popitem()[0],start_date,end_date)
        else :
            list_itms = [(value, key) for key, value in a_dict.items()]
            if not os.path.exists('F:/Agriculture/satellite_images/%s_%s/%s.zip'%(start_date,end_date,item_title1)):
                download(max(list_itms)[1],start_date,end_date)
        count = count + 1
    
    
    

total_images = int("6")
monthly_gap = int("1")
count = int("1")
#title = 'S2A_MSIL1C_20171123T052131_N0206_R062_T43QFV_20171123T091125'
title = 'S2A_OPER_PRD_MSIL1C_PDMC_20161128T133732_R062_V20161128T052152_20161128T052152'


while (count <= total_images-monthly_gap+1) :
    if count == 1 :
        start_month = int("11")
        end_month = int("12")
        start_date = dt.date(int("2016"),start_month,int("11"))
        end_date = dt.date(int("2016"),end_month,int("11"))
        products = product_list(coordinates,start_date,end_date)
        key_tuple = key_generator(title,products)
        tuple_conv = [(value, key) for key, value in key_tuple]
        conv = tuple_conv.pop()
        final_key = conv[0]
        item_pltid = conv[1]['platformserialidentifier']
        item_title = conv[1]['title']
        item_title_wrd = item_title.split("_")
        bfr_tid = item_title_wrd[4]
        if not os.path.exists('F:/Agriculture/satellite_images/%s_%s/%s.zip'%(start_date,end_date,item_title)):
            download(final_key,start_date,end_date)
        count = count + 1
    else :
        end_month = start_month
        start_month = end_month - monthly_gap
        start_date = dt.date(int("2016"),start_month,int("11"))
        end_date = dt.date(int("2016"),end_month,int("11"))
        products = product_list(coordinates,start_date,end_date)
        total_len = len(products) 
        a_dict = {}
        while (total_len > 0) :
            item1 = products.popitem()
            item_key = item1[1]['uuid']
            item_title1 = item1[1]['title']
            item_pltid1 = item1[1]['platformserialidentifier']
            item_title1_wrd = item_title1.split("_")
            bfr_tid1 = item_title1_wrd[4]
            itm_date1 = item_title1_wrd[6]
            int_date = int(item_title1_wrd[5].split("T")[0])
            if (item_pltid == item_pltid1) & (bfr_tid1==bfr_tid) :
                a_dict[item_key] = int_date
            total_len = total_len - 1
        if len(a_dict) == 1 :
            if not os.path.exists('F:/Agriculture/satellite_images/%s_%s/%s.zip'%(start_date,end_date,item_title1)):
                download(a_dict.popitem()[0],start_date,end_date)
        else :
            list_itms = [(value, key) for key, value in a_dict.items()]
            if not os.path.exists('F:/Agriculture/satellite_images/%s_%s/%s.zip'%(start_date,end_date,item_title1)):
                download(max(list_itms)[1],start_date,end_date)
        count = count + 1
    






   
    

#S2A_MSIL1C_20171103T052001_N0206_R062_T43QFV_20171103T091057



