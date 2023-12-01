import cv2
import os

# classes = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam', 
#         'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield','harbor', 
#         'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle','windmill']
classes = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
palette = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
        (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
        (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
        (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
        (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
        (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
        (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
        (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
        (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
        (134, 134, 103), (145, 148, 174), (255, 208, 186),
        (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
        (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
        (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
        (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
        (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
        (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
        (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
        (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
        (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
        (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
        (191, 162, 208)]


image_path = 'work_dirs/vis'
# image_path = '/disk2/lhd/datasets/attack/dota/images'
# label_path = '/disk2/lhd/datasets/attack/dota/labelTxt'
# name = 'P1373__1__1200___3200'
# img = cv2.imread(os.path.join(image_path, name+'.png'))
# with open(os.path.join(label_path, name+'.txt'), 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         splitline = line.strip().split(' ')
#         bbox = list(map(int, list(map(float, splitline[:8]))))
#         label = splitline[8]
#         wh, base = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5,thickness=2)
#         cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[4], bbox[5]), color = palette[classes.index(label)], thickness=2)
#         cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+wh[0], bbox[1]+wh[1]), \
#                         thickness=-1, color=(0,0,0))
#         cv2.putText(img, label,(bbox[0], bbox[1]+base), \
#                         fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4,thickness=1, color=(255,255,255))
#     cv2.imwrite('work_dirs/vis_new/'+name+'_gt.png', img)
images = os.listdir(image_path)
# labels = os.listdir(label_path)
for image in images:
    img = cv2.imread(os.path.join(image_path, image))
    # img1 = img[:, :800]
    img2 = img[:, 800:]
    # cv2.imwrite(os.path.join('work_dirs/vis_new', image[:-4] + '_gt.png'), img1)
    cv2.imwrite(os.path.join('work_dirs/vis_new', image[:-4] + '_att.png'), img2)
print()
