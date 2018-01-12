

footprint:"Intersects(POLYGON((76.83460782454188 17.329723965642216,76.83568842536901 17.33293504810389,
76.83640301623858 17.3340164881039,76.83091286687488 17.333783563103736,
76.83084315069249 17.32965741411637,76.83460782454188 17.329723965642216,
76.83460782454188 17.329723965642216)))") 
AND ( beginPosition:[2017-09-27T00:00:00.000Z TO 2017-11-27T23:59:59.999Z] 
AND endPosition:[2017-09-27T00:00:00.000Z TO 2017-11-27T23:59:59.999Z] )

75.94731630555209 17.08793537940518,76.96684575691108 17.08793537940518,
76.96684575691108 18.08319719860654,75.94731630555209 18.08319719860654,
75.94731630555209 17.08793537940518)))" ) 
AND ( beginPosition:[2017-01-01T00:00:00.000Z TO 2017-02-11T23:59:59.999Z] 
AND endPosition:[2017-01-01T00:00:00.000Z TO 2017-02-11T23:59:59.999Z] ) AND (platformname:Sentinel-2)
                                                                         

import gdal 
from datetime import datetime
import os
import numpy as np
import rasterio
from rasterio.tools.mask import mask
import cv2
from osgeo import osr
import math
from matplotlib import pyplot as plt
from matplotlib.path import Path
from dateutil.parser import parse
import struct
import matplotlib.pyplot as plt
#import diagonal_crop

geoms = [{'type':'Polygon','coordinates':[[(76.834296,17.329731,),(76.801128,17.335697),(76.814175,17.363879),(76.839237,17.359619)]]}]
#farm_coordinates = [(17.372948, 76.901725) , (17.373414, 76.903280) , (17.376036, 76.902304) , (17.375028, 76.900739)]
#farm_coordinates = [(17.145027, 76.292649) , (17.145237, 76.293630) , (17.143925, 76.293877) , (17.143745, 76.293040)]

###farm_coordinates = [(17.475519, 76.560705) , (17.475389, 76.561423) , (17.474448, 76.561234) , (17.474864, 76.560584)]

##farm_coordinates = [(17.338497, 76.391528) , (17.338378, 76.392435) , (17.340187, 76.392998) , (17.340299, 76.391786)]

####farm_coordinates = [(17.354940, 76.623445) , (17.35508, 76.624218) , (17.354065, 76.624320) , (17.353993, 76.623515)]

farm_coordinates = [(17.636198, 76.815034) , (17.636096, 76.816249) , (17.636873, 76.816467) , (17.36865, 76.815171)]

year_lst = os.listdir('F:/Agriculture/satellite_images')
dates_lst = [os.listdir('F:/Agriculture/satellite_images/%s'%(fname)) for fname in year_lst]
lst16 = dates_lst[0] 
lst17 = dates_lst[1] 
lst17 = [date.split('-')[0]+'-'+date.split('-')[1]+'-'+date.split('-')[2][2:] for date in lst17 ]
lst16 = [date.split('-')[0]+'-'+date.split('-')[1]+'-'+date.split('-')[2][2:] for date in lst16 ]
lst17 = [datetime.strptime(date,'%d-%m-%y') for date in lst17 ]
lst16 = [datetime.strptime(date,'%d-%m-%y') for date in lst16 ]
lst17 = sorted(lst17)
lst16 = sorted(lst16)
lst17 = [str(date).split(' ')[0] for date in lst17 ]
lst16 = [str(date).split(' ')[0] for date in lst16 ]
lst17 = [date.split('-')[2]+'-'+date.split('-')[1]+'-'+date.split('-')[0] for date in lst17]
lst16 = [date.split('-')[2]+'-'+date.split('-')[1]+'-'+date.split('-')[0] for date in lst16]

titles17 = [ os.listdir('F:/Agriculture/satellite_images/2017/%s'%(dates)) for dates in lst17  ]
titles17 = [title[0].split(".")[0] for title in titles17 ]
date_dict17 = np.vstack((lst17,titles17))
date_dict17 = np.transpose(date_dict17)
#title = 'S2A_MSIL1C_20170924T051651_N0205_R062_T43QFV_20170924T053050'

def coordi_to_pixel(elem,lon,lat):
    src = osr.SpatialReference()
    src.SetWellKnownGeogCS("WGS84")
    padfTransform = elem.GetGeoTransform()
    projection = elem.GetProjection()
    dst = osr.SpatialReference(projection)
    ct = osr.CoordinateTransformation(src, dst)
    xy = ct.TransformPoint(lon, lat)
    pixel_x = abs((xy[0] - padfTransform[0])/padfTransform[1])
    pixel_y = abs((xy[1] - padfTransform[3])/padfTransform[5])
    return math.floor(pixel_x),math.floor(pixel_y)


def image_cropping(farm_coordinates,img_file,npath):
    elem = gdal.Open(npath + img_file)               
    crop_pixel = [coordi_to_pixel(elem,lon,lat) for lat,lon in farm_coordinates ]
    lat1 = farm_coordinates[0][0]
    lon1 = farm_coordinates[0][1]
    new_name = img_file.split(".")[0]
    if not os.path.exists(npath+'cropped_images_'+str(lon1)+'_'+str(lat1)+'/'+new_name+'.TIFF'):  
        cv_open = cv2.imread(npath+img_file)
        arr = np.zeros(cv_open.shape[0:2])
        # masking according to pixel_coordinates.
        points = np.indices(arr.shape).reshape(2, -1).T
        path = Path(crop_pixel)   
        mask = path.contains_points(points)
        mask = mask.reshape(arr.shape).astype(arr.dtype)
        mask1 = np.dstack((mask,mask))
        mask = np.dstack((mask1,mask))
        final_pixels = cv_open*mask
        x_values = [x for x,y in crop_pixel ]
        y_values = [y for x,y in crop_pixel ]
        xmin,ymin,xmax,ymax = [np.min(x_values),np.min(y_values),np.max(x_values),np.max(y_values)]
        final_image =final_pixels[(xmin-2):(xmax+2),(ymin-2):(ymax+2)]
        if not os.path.exists(npath+'cropped_images_'+str(lon1)+'_'+str(lat1)):
            os.makedirs(npath+'cropped_images_'+str(lon1)+'_'+str(lat1))     
        cv2.imwrite(npath+'cropped_images_'+str(lon1)+'_'+str(lat1)+'/'+new_name+'.TIFF',final_image)


def raster_cropping(farm_coordinates,img_file,npath) :
    elem = gdal.Open(npath + img_file)
    lat1 = farm_coordinates[0][0]
    lon1 = farm_coordinates[0][1]
    new_name = img_file.split(".")[0]
    if not os.path.exists(npath+'cropped_images_'+str(lon1)+'_'+str(lat1)+'/'+new_n3ame+'.jp2'):     
        narr = elem.ReadAsArray()           
        arr = np.zeros(narr.shape[0:2])
        crop_pixel = [coordi_to_pixel(elem,lon,lat) for lat,lon in farm_coordinates ]
        points = np.indices(narr.shape).reshape(2, -1).T
        path = Path(crop_pixel)   
        mask = path.contains_points(points)
        mask = mask.reshape(arr.shape).astype(arr.dtype)
        final_pixels = narr*mask
        x_values = [x for x,y in crop_pixel ]
        y_values = [y for x,y in crop_pixel ]
        xmin,ymin,xmax,ymax = [np.min(x_values),np.min(y_values),np.max(x_values),np.max(y_values)]
        final_image =final_pixels[(xmin-1):(xmax+1),(ymin-1):(ymax+1)]
        cv2.imwrite(npath+'cropped_images_'+str(lon1)+'_'+str(lat1)+'/'+new_name+'.jp2',final_image)

def getting_listfiles(year,pic_date,title) :
    dte = os.listdir(('F:/Agriculture/satellite_images/%s/%s/%s.SAFE/GRANULE')%(year,pic_date,title))
    path = "F:/Agriculture/satellite_images/%s/%s/%s.SAFE/GRANULE/%s/IMG_DATA/" %(year,pic_date,title,dte[0])                                       
    #F:\Agriculture\satellite_images\2017-07-27_2017-09-27\S2A_MSIL1C_20170924T051651_N0205_R062_T43QFV_20170924T053050.SAFE\GRANULE\L1C_T43QFV_A011785_20170924T053050\IMG_DATA
    listfiles = os.listdir(path)
    return [listfiles,path] 

def histogram_calculation(path,img_file):
    image = cv2.imread(path+img_file) 
    if not np.sum(image) == 0 :    
        mm = image > 0
        coords = np.argwhere(mm)
        x0, y0 , z0 = coords.min(axis=0)
        x1, y1 , z1 = coords.max(axis=0) + 1 
        image = image[x0:x1, y0:y1 , z0:z1]    
    img_hist,_ = np.histogram(image.ravel(),bins=32)
    img_hist = img_hist/np.sum(img_hist)
    return img_hist



year = 2017
for i in range(0,len(date_dict17)) :
    list_path = getting_listfiles(year,date_dict17[i][0],date_dict17[i][1])
    image_files = [ img_file for img_file in list_path[0]  if img_file.endswith(".jp2") ]
    for img_file in image_files :
        image_cropping(farm_coordinates,img_file,list_path[1])
   


year = 2017

for i in range(0,len(date_dict17)) :
    list_path = getting_listfiles(year,date_dict17[i][0],date_dict17[i][1])
    fpath = list_path[1]+'cropped_images_76.560705_17.475519'+'/'
    images = os.listdir(fpath)
    image_files = [ img_file for img_file in images  if img_file.endswith(".TIFF") ]
    image_files = image_files[:-1]
    if i ==0 :
        temp1 = []
        for img_file in image_files :
            hist = histogram_calculation(fpath,img_file)
            temp1.append(hist)
        temp1 = np.array(temp1)
        temp1 = np.transpose(temp1)
    else :
        temp = []
        for img_file in image_files :
            hist = histogram_calculation(fpath,img_file)
            temp.append(hist)
        temp = np.array(temp)
        temp = np.transpose(temp)
        temp1 = np.dstack((temp1,temp))
        final = np.rollaxis(temp1,2)
        
final_tensor = np.rollaxis(final,1)


#NDVI Calculation .

def NDVI_calc(b04,b08):
    sum_mat = b04 + b08
    #sum_mat[sum_mat==0] = 1    
    diff_mat = b08 - b04
    #diff_mat[diff_mat==0] = 1
    sum_mat.astype('float32')
    diff_mat.astype('float32')
    final_mat = np.divide(diff_mat,sum_mat)
    return final_mat
    

temp1 = []
year = 2017
for i in range(0,len(date_dict17)) :
    list_path = getting_listfiles(year,date_dict17[i][0],date_dict17[i][1])
    image_files = [ img_file for img_file in list_path[0]  if img_file.endswith(".jp2") ]
    for img_file in image_files :
        if img_file.endswith("04.jp2") : 
            elem_b04 = gdal.Open(list_path[1]+img_file)
            b04 = elem_b04.ReadAsArray() 
            crop_pixel = [coordi_to_pixel(elem_b04,lon,lat) for lat,lon in farm_coordinates]
            arr = np.zeros(b04.shape[0:2])
            # masking according to pixel_coordinates.
            points = np.indices(arr.shape).reshape(2, -1).T
            path = Path(crop_pixel)   
            mask = path.contains_points(points)
            mask = mask.reshape(arr.shape).astype(arr.dtype)
            final_pixels = b04*mask
            x_values = [x for x,y in crop_pixel ]
            y_values = [y for x,y in crop_pixel ]
            xmin,ymin,xmax,ymax = [np.min(x_values),np.min(y_values),np.max(x_values),np.max(y_values)]
            final_b04 =final_pixels[(xmin-1):(xmax+1),(ymin-1):(ymax+1)]
        if img_file.endswith("08.jp2") : 
            elem_b08 = gdal.Open(list_path[1]+img_file)
            b08 = elem_b08.ReadAsArray()
            crop_pixel = [coordi_to_pixel(elem_b04,lon,lat) for lat,lon in farm_coordinates]
            arr = np.zeros(b04.shape[0:2])
            # masking according to pixel_coordinates.
            points = np.indices(arr.shape).reshape(2, -1).T
            path = Path(crop_pixel)   
            mask = path.contains_points(points)
            mask = mask.reshape(arr.shape).astype(arr.dtype)
            final_pixels = b08*mask
            x_values = [x for x,y in crop_pixel ]
            y_values = [y for x,y in crop_pixel ]
            xmin,ymin,xmax,ymax = [np.min(x_values),np.min(y_values),np.max(x_values),np.max(y_values)]
            final_b08 =final_pixels[(xmin-1):(xmax+1),(ymin-1):(ymax+1)]
    ndvi = NDVI_calc(final_b04,final_b08)
    temp1.append(ndvi)




ras_b04 = elem_b04.GetRasterBand(1)
numLines = ras_b04.YSize
red_scanline = ras_b04.ReadRaster()
        # Unpack the line of data to be read as floating point data
ss = int(ras_b04.XSize)
red_tuple = struct.unpack(60280200*'f', red_scanline)



ras_b04 = elem_b04.GetRasterBand(1)
numLines = ras_b04.YSize
red_scanline = ras_b04.ReadRaster(xoff=0, yoff=0,xsize=ras_b04.XSize, ysize=1,buf_xsize=ras_b04.XSize, buf_ysize=1,buf_type=gdal.GDT_Float32)
# Unpack the line of data to be read as floating point data
ss = int(ras_b04.XSize)
red_tuple = struct.unpack('f'*ras_b04.XSize, red_scanline)





for lin in range(numLines) :
    print(lin)

ttt = temp1

del ttt[7]

ll = [ttt.append(ff) for ff in temp1]

arr_ndvi = [ crop(arr) for arr in temp1 ]

mean_arr = [mean_cal(arr) for arr in arr_ndvi]

def crop(arr) :
    mm = arr > 0
    coords = np.argwhere(mm)
    x0, y0  = coords.min(axis=0)
    x1, y1  = coords.max(axis=0) + 1 
    image = arr[x0:x1, y0:y1 ]
    return image 

def mean_cal(ddd):
    where_are_NaNs = np.isnan(ddd)
    ddd[where_are_NaNs] = 0
    return np.mean(ddd)


count = 0
while (count < 2):
    if count == 0:
        temp1 = []
        start = int('10') 
        end = int('11')
        list_path = os.listdir('F:/Agriculture/satellite_images/2016-%s-11_2016-%s-11'%(start,end))
        image_files = [ img_file for img_file in list_path  if img_file.endswith(".jp2") ]
        for img_file in image_files :
            if img_file.endswith("04.jp2") : 
                elem_b04 = gdal.Open('F:/Agriculture/satellite_images/2016-%s-11_2016-%s-11'%(start,end)+'/'+img_file)
                #elem_b04 = gdal.Open('C:/Users/5558/Downloads/T43QFV_20170107T052201_B04.jp2')
                b04 = elem_b04.ReadAsArray() 
                crop_pixel = [coordi_to_pixel(elem_b04,lon,lat) for lat,lon in farm_coordinates]
                arr = np.zeros(b04.shape[0:2])
                # masking according to pixel_coordinates.
                points = np.indices(arr.shape).reshape(2, -1).T
                path = Path(crop_pixel)   
                mask = path.contains_points(points)
                mask = mask.reshape(arr.shape).astype(arr.dtype)
                final_pixels = b04*mask
                x_values = [x for x,y in crop_pixel ]
                y_values = [y for x,y in crop_pixel ]
                xmin,ymin,xmax,ymax = [np.min(x_values),np.min(y_values),np.max(x_values),np.max(y_values)]
                final_b04 =final_pixels[(xmin-1):(xmax+1),(ymin-1):(ymax+1)]
            if img_file.endswith("08.jp2") : 
                elem_b08 = gdal.Open('F:/Agriculture/satellite_images/2016-%s-11_2016-%s-11'%(start,end)+'/'+img_file)
                #elem_b08 = gdal.Open('C:/Users/5558/Downloads/T43QFV_20170107T052201_B08.jp2')
                b08 = elem_b08.ReadAsArray()
                crop_pixel = [coordi_to_pixel(elem_b04,lon,lat) for lat,lon in farm_coordinates]
                arr = np.zeros(b04.shape[0:2])
                # masking according to pixel_coordinates.
                points = np.indices(arr.shape).reshape(2, -1).T
                path = Path(crop_pixel)   
                mask = path.contains_points(points)
                mask = mask.reshape(arr.shape).astype(arr.dtype)
                final_pixels = b08*mask
                x_values = [x for x,y in crop_pixel ]
                y_values = [y for x,y in crop_pixel ]
                xmin,ymin,xmax,ymax = [np.min(x_values),np.min(y_values),np.max(x_values),np.max(y_values)]
                final_b08 =final_pixels[(xmin-1):(xmax+1),(ymin-1):(ymax+1)]
        ndvi = NDVI_calc(final_b04,final_b08)
        temp1.append(ndvi)
        count = count + 1
    else :
        start = end
        end = start + int('1')
        list_path = os.listdir('F:/Agriculture/satellite_images/2016-%s-11_2016-%s-11'%(start,end))
        image_files = [ img_file for img_file in list_path  if img_file.endswith(".jp2") ]
        for img_file in image_files :
            if img_file.endswith("04.jp2") : 
                elem_b04 = gdal.Open('F:/Agriculture/satellite_images/2016-%s-11_2016-%s-11'%(start,end)+'/'+img_file)
                b04 = elem_b04.ReadAsArray() 
                crop_pixel = [coordi_to_pixel(elem_b04,lon,lat) for lat,lon in farm_coordinates]
                arr = np.zeros(b04.shape[0:2])
                # masking according to pixel_coordinates.
                points = np.indices(arr.shape).reshape(2, -1).T
                path = Path(crop_pixel)   
                mask = path.contains_points(points)
                mask = mask.reshape(arr.shape).astype(arr.dtype)
                final_pixels = b04*mask
                x_values = [x for x,y in crop_pixel ]
                y_values = [y for x,y in crop_pixel ]
                xmin,ymin,xmax,ymax = [np.min(x_values),np.min(y_values),np.max(x_values),np.max(y_values)]
                final_b04 =final_pixels[(xmin-1):(xmax+1),(ymin-1):(ymax+1)]
            if img_file.endswith("08.jp2") : 
                elem_b08 = gdal.Open('F:/Agriculture/satellite_images/2016-%s-11_2016-%s-11'%(start,end)+'/'+img_file)
                b08 = elem_b08.ReadAsArray()
                crop_pixel = [coordi_to_pixel(elem_b04,lon,lat) for lat,lon in farm_coordinates]
                arr = np.zeros(b04.shape[0:2])
                # masking according to pixel_coordinates.
                points = np.indices(arr.shape).reshape(2, -1).T
                path = Path(crop_pixel)   
                mask = path.contains_points(points)
                mask = mask.reshape(arr.shape).astype(arr.dtype)
                final_pixels = b08*mask
                x_values = [x for x,y in crop_pixel ]
                y_values = [y for x,y in crop_pixel ]
                xmin,ymin,xmax,ymax = [np.min(x_values),np.min(y_values),np.max(x_values),np.max(y_values)]
                final_b08 =final_pixels[(xmin-1):(xmax+1),(ymin-1):(ymax+1)]
        ndvi = NDVI_calc(final_b04,final_b08)
        temp1.append(ndvi)
        count = count + 1








count = 0
year = 2016
while (count < len(titles)):
    if count == 0:
        start = int('10') 
        end = int('11')
        list_path = getting_listfiles(start,end,titles[count])
        image_files = [ img_file for img_file in list_path[0]  if img_file.endswith(".jp2") ]
        for img_file in image_files :
            if img_file.endswith("04.jp2") | img_file.endswith("08.jp2") : 
                raster_cropping(farm_coordinates,img_file,list_path[1])
        count = count + 1
    else :
        start = end
        end = start + int('1')
        list_path = getting_listfiles(start,end,titles[count])
        image_files = [ img_file for img_file in list_path[0]  if img_file.endswith(".jp2") ]
        for img_file in image_files :
            if img_file.endswith("04.jp2") | img_file.endswith("08.jp2") : 
                raster_cropping(farm_coordinates,img_file,list_path[1])
        count = count + 1


import pandas as pd


ndvi1 = pd.read_excel('C:/Users/5558/Desktop/17.338497_76.391528_ndvi.xlsx')

1ndvi2 = pd.read_excel('C:/Users/5558/Desktop/17.475519_76.560705_ndvi.xlsx')

ndvi3 = pd.read_excel('C:/Users/5558/Desktop/17.354940_76.623445_ndvi.xlsx')

plt.plot(range(0,len(ndvi1)),ndvi2,ndvi3)







