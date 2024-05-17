import os
import numpy as np
from PIL import Image


def xed(np_img):
    edge_count = 0
    position_count = 0
    for index, row in enumerate(np_img):
        prev_pixel = 0
        for pixel in row:
            if pixel == 255 and prev_pixel < 255:
                edge_count += 1
                position_count += len(np_img) - index
            prev_pixel = pixel
        if prev_pixel < 255:  # jezeli ostatni pixel jest "on" to doliczam tez krawedz z granica
            edge_count += 1
            position_count += len(np_img) - index
    return edge_count, position_count


def yed(np_img):
    edge_count = 0
    position_count = 0
    for index, col in enumerate(np_img.T):
        prev_pixel = 0
        for pixel in reversed(col):
            if pixel == 255 and prev_pixel < 255:
                edge_count += 1
                position_count += len(np_img.T) - index
            prev_pixel = pixel
        if prev_pixel < 255:  # jezeli ostatni pixel jest "on" to doliczam tez krawedz z granica
            edge_count += 1
            position_count += len(np_img.T) - index
    return edge_count, position_count


def data_processing(img):
    base_width = 45
    wpercent = (base_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((base_width, hsize), Image.Resampling.LANCZOS)

    threshold = 127
    img_gs = img.convert("L")
    img_gs = img_gs.point(lambda p: 255 if p > threshold else 0)

    np_img = np.array(img_gs)
    return np_img

def tl2(filepath):
    img = Image.open(filepath)
    #base_width = 45
    #wpercent = (base_width / float(img.size[0]))
    #hsize = int((float(img.size[1]) * float(wpercent)))
    #img = img.resize((base_width, hsize), Image.Resampling.LANCZOS)

    #img_gs = img.convert("L")
    #np_img = np.array(img_gs)
    np_img = data_processing(img)

    on_pixels = np.where(np_img < 255)
    on_pixels = np.array(on_pixels).T  # added later

    # tutaj uznalem ze moze lepiej bedzie najpierw policzyc koordynaty wszystkie a potem dopiero je znormalizowac
    xbox = (np.max(on_pixels[:, 1]) + np.min(on_pixels[:, 1])) / 2
    ybox = (np.max(on_pixels[:, 0]) + np.min(on_pixels[:, 0])) / 2
    width = np.max(on_pixels[:, 1]) - np.min(on_pixels[:, 1])
    height = np.max(on_pixels[:, 0]) - np.min(on_pixels[:, 0])
    onpix = len(on_pixels[:, 0])
    xbar = (np.mean(on_pixels[:, 1]) - xbox) / width
    ybar = (np.mean(on_pixels[:, 0]) - ybox) / height

    # zmienne pomocnicze
    xd_square = []  # kwadraty odleglosci od srodka pudelka poziome i pionowe
    yd_square = []
    x_distance = []
    y_distance = []
    xy_distance = []

    x2y_d = []
    xy2_d = []
    for i in range(len(on_pixels[:, 1])):
        xd_square.append((on_pixels[:, 1][i] - xbox) ** 2)
        yd_square.append((on_pixels[:, 0][i] - ybox) ** 2)
        x_distance.append((on_pixels[:, 1][i] - xbox))
        y_distance.append((on_pixels[:, 0][i] - ybox))
        xy_distance.append(x_distance[i] * y_distance[i])
        x2y_d.append(xd_square[i] * y_distance[i])
        xy2_d.append(yd_square[i] * x_distance[i])

    x2bar = np.mean(xd_square)
    y2bar = np.mean(yd_square)
    xybar = np.mean(xy_distance)
    x2ybar = np.mean(x2y_d)
    xy2bar = np.mean(xy2_d)

    # te cztery jeszcze nie sa zaimplementowane
    (xedge, xedgey) = xed(np_img)
    (yedge, yedgex) = yed(np_img)
    xedge = xedge / width
    yedge = yedge / height

    return f'{xbox},{ybox},{width},{height},{onpix},{xbar},{ybar},{x2bar},{y2bar},{xybar},{x2ybar},{xy2bar},{xedge},{xedgey},{yedge},{yedgex}'


directory = 'literki'
file_csv = open('letter-recognition-new.csv', 'w')
file_csv.write('letter,xbox,ybox,width,height,onpix,xbar,ybar,x2bar,y2bar,xybar,x2ybar,xy2bar,xedge,xedgey,yedge,yedgex')
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        data = tl2(f)
        file_csv.write('\n' + f[len(f)-5] + ',' + data)
