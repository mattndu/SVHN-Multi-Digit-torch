import os.path
import os
import json
import csv 
import re 
import random
import shutil
import cv2

from openalpr import Alpr

alpr = Alpr('us','/etc/openalpr/openalpr.conf', '/usr/local/share/openalpr/runtime_data')
alpr.set_top_n(1)
alpr.set_country("us")	

input_dir = "/orpix/data/tds/labeled_sets/"
output_dir = "/orpix/data/orpix_lp/plate_images_2"
csv_file  = os.path.join(output_dir, "labels.csv")
csvf = open(csv_file,'w+')
dirs = os.listdir(input_dir)

for d in dirs:
    fullpath = os.path.join(input_dir, d)
    if os.path.isdir(fullpath):
        
        #get images
        labels_file = os.path.join(fullpath, "labels.txt")
        labels = {}
        if os.path.exists(labels_file):
            lines = open(labels_file).readlines()
            for line in lines:
                toks = line.split('\t')
                fpath = toks[0]
                fpath = fpath.replace('C:', '')
                fpath = fpath.replace('\\', '/')
                fname = os.path.basename(fpath)
                plate_read = toks[1]
                labels[fname] = plate_read.strip()
                
        images = os.listdir(fullpath)
        for f in images:
            print "processing image %s" % f
            if os.path.splitext(f)[1] == ".png":
                
                gtlabel = None
                if f in labels:
                    gtlabel = labels[f]
                
                #if we have a gt list we skip images that don't have gt
                if gtlabel == None and len(labels) > 0:
                    continue
                
                image_path = os.path.join(fullpath, f)
                r = alpr.recognize_file(image_path)
                results = r["results"]
                #if the result is empty we skip (unless we have gt)
                if len(results) == 0 and (gtlabel == None or not d.endswith("_cropped")):
                    continue
                
                
                
                output_name = d + "_" + f
                output_path = os.path.join(output_dir, output_name)
                
                if gtlabel == None:
                    line = ".,%s,,%s [%s]\n" % (output_name, r["results"][0]['plate'], r["results"][0]['confidence'])                 
                else:
                    line = ".,%s,,%s [1.0]\n" % (output_name, gtlabel)                 
                csvf.write(line)
                csvf.flush()
                
                if d.endswith("_cropped"):
                    shutil.copyfile(image_path, output_path)
                else:
                    coords = r["results"][0]['coordinates']
                    minx = 100000
                    maxx = 0
                    miny = 100000
                    maxy = 0
                    for pt in coords:
                        minx = min(minx, pt['x'])
                        maxx = max(maxx, pt['x'])
                        miny = min(miny, pt['y'])
                        maxy = max(maxy, pt['y'])
                    
                    img = cv2.imread(image_path)
                    if len(img.shape) ==3:
                        cropped = img[miny:maxy, minx:maxx, :]
                    else:
                        cropped = img[miny:maxy, minx:maxx]
                    
                    cv2.imwrite(output_path, cropped)
                
                
csvf.close()