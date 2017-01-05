import os.path
import os
import json
import csv 
import re 
import random
import shutil
import cv2

ocr_reads_file = '/orpix/data/orpix_lp/plate_images/labels.csv'
lp_image_dir = '/orpix/data/orpix_lp/plate_images/'
output_dir = '/orpix/data/orpix_lp/plate_gt_torch/'
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)  

test_labels_path = os.path.join(output_dir, 'test_labels.txt')
train_labels_path = os.path.join(output_dir, 'train_labels.txt')
testf = open(test_labels_path, 'w')
trainf = open(train_labels_path, 'w')
test_count = 1
train_count = 1

pcttest = .1
lp_extension = "_enhance_car_lp.jpeg"
with open(ocr_reads_file, 'rb') as csvfile:
    #csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    
    first_line = True
    lines = csvfile.readlines()
    for line in lines:
        row = line.split(',')
        if first_line:
            first_line = False
            continue
        
        #if row[0] == ".":
        filename = row[1]
        #else:
        #    filename = row[0]
        plate_read1 = row[2]
        if plate_read1 == "\"" or plate_read1 == "":
            plate_read1 = None
            
        ocr_result2 = row[3]
        m = re.match("(?:\")*(.*?)\s+?\[(.*?)\]", ocr_result2)
        
        plate_conf = 0
        plate_read2 = None
        if m:
            plate_read2 = m.group(1)
            plate_conf = float(m.group(2))
        
        if plate_read1 == None and plate_read2 == None:
            continue
        
        if plate_read1 == plate_read2:
            plate_read = plate_read1
        elif plate_conf > .85:
            plate_read = plate_read2
        elif not plate_read1 == None:
            plate_read = plate_read1
        
        plate_read = plate_read.strip()
        m = re.match('^[A-Za-z0-9]+$', plate_read)
        if m == None or len(plate_read) < 4 or len(plate_read) > 8:
            continue
        
        src_lp_path = os.path.join(lp_image_dir, filename)
        if not os.path.exists(src_lp_path):
            print "image %s wasn't found - skipping" % src_lp_path
            continue
        
        istest = False
        #figure out test vs train
        rnd = random.random()
        if rnd < .1:
            istest = True
        
        img = cv2.imread(src_lp_path)
        if img == None:
            print "image could not be read, skipping"
            continue

        #destination is counter.jpg
        if istest:
            dstpath = os.path.join(test_dir, "%d.png" % test_count)
            testf.write(plate_read + "\n")
            test_count += 1
        else:
            dstpath = os.path.join(train_dir, "%d.png" % train_count)
            trainf.write(plate_read + "\n")
            train_count += 1
        
        grey = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        resized = cv2.resize(grey, (136,95))
        cv2.imwrite(dstpath, resized)        
        
        
        
        
trainf.close()
testf.close()
        