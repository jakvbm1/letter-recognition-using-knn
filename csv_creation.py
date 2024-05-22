import os
from functions import tl2

directory = 'literki'
file_csv = open('letter-recognition-new.csv', 'w')
file_csv.write('letter,xbox,ybox,width,height,onpix,xbar,ybar,x2bar,y2bar,xybar,x2ybar,xy2bar,xedge,xedgey,yedge,yedgex')
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        data = tl2(f)
        file_csv.write('\n' + f[len(f)-5] + ',' + data)
