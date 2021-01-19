import cv2
import json
import os
import numpy as np
# from bbox import BoundingBox
# img = cv2.imread('SVHN_data/test/new/58.png')
# print(img.shape)
# cv2.imshow('img',img[44:222,47:85])
# cv2.waitKey(0)

# def xyhwtoxyxy(bbox):
#     x,y,h,w = bbox
#     xminxmax = [int(x-w/2),int(x+w/2)]
#     yminymax = [int(y-h/2),int(y+h/2)]
#     return xminxmax, yminymax

def xyxytoxyhw(xmin,ymin,xmax,ymax):
    # xmin,ymin,xmax,ymax = bbox
    xcenter = (xmin + xmax)/2
    ycenter = (ymin + ymax)/2
    w = 2*(xmax-xcenter)
    h = 2*(ymax-ycenter)
    return xcenter,ycenter,h,w
# box = [66.78260869565217, 133.56521739130434, 178.08695652173913, 37.84347826086957]
# xyxy = box
# xyxy[2:] += xyxy[:2]
# box = [int(i) for i in box]
# box[:, 2:4] 
# print(xyxytoxyhw([47,44,85,222]))
# print(xyhwtoxyxy(box))


# file_names = os.listdir('images/CHELSEA-WESTHAM/')
# for file in file_names:
#     file_name = 'images/CHELSEA-WESTHAM/'+file
#     img = cv2.imread(file_name)
#     cv2.imshow('img', img)
#     k = cv2.waitKey(0)
#     # print('accept press \'a\' or another key:')
#     if k == ord('a'):
#         cv2.destroyAllWindows()
#     else:
#         cv2.destroyAllWindows()
#         os.remove(file_name)

# import the necessary packages
import cv2
import numpy as np
# a = {'file': image,
#     'label': '6 digits - array 11 via array[10]= 1 if None',
#     'possition': 'possition of number'}


ref_point = []
index = 0
x0, x1, y0, y1 = 0, 0, 0, 0
def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, index, x0, x1, y0, y1

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))
        # draw a rectangle around the region of interest
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 1)
        cv2.imshow("image", image)
        (x0,y0), (x1,y1) = ref_point[0], ref_point[1]
        # print(x0,y0,x1,y1)
        index += 1
        # return x0,y0,x1,y1

datasets = []
folder = 'images/CHELSEA-WESTHAM/'
files = os.listdir(folder)
for file in files:
    file_path = os.path.join(folder,file)
    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(file_path)
    image = cv2.resize(image,(170,256))
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", shape_selection)

    # keep looping until the 'q' key is pressed
    # while True:
        # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(0)

    # press 'r' to reset the window
    try:
        if key == ord("r"):
            image = clone.copy()
        elif key == ord("a"):
            print('Number for label')
            numbers = input()
            label = np.zeros((6,11),dtype=np.int8)
            row = 0
            for number in str(numbers):
                label[row][int(number)] = 1
                row += 1
            print(label.tolist())
            x,y,h,w = xyxytoxyhw(x0,y0,x1,y1)
            datasets.append({
                "file": file_path,
                "label": label.tolist(),
                "possition": [x,y,h,w]
            })
        elif key == ord("c") or key == ord("q"):
            os.remove(file_path)
            cv2.destroyAllWindows() 
    except ValueError:
        os.remove(file_path)
        cv2.destroyAllWindows() 
        print(ValueError)
with open('CHELSEA-WESTHAM.json','w',encoding='utf-8') as f:
    json.dump(datasets,f)
