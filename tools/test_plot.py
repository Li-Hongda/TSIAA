import matplotlib.pyplot as plt
import numpy as np
# import cv2
# image1 = cv2.imread("/disk2/lhd/codes/attack/work_dirs/examples/dota/tabim_vfnet_eps20/images/P0003__1__0___223.png")
# image2 = cv2.imread('/disk2/lhd/datasets/attack/dota/images/P0003__1__0___223.png')
# print()

with open("/disk2/lhd/codes/attack/work_dirs/figures/atss_asr/ours_atss_dior_asr.txt", 'r') as f:
    lines1 = f.readlines()
lines1 = list(map(float, lines1))
with open("/disk2/lhd/codes/attack/work_dirs/figures/atss_asr/xia_atss_dior_asr.txt", 'r') as f:
    lines2 = f.readlines()
lines2 = list(map(float, lines2))
with open("/disk2/lhd/codes/attack/work_dirs/figures/atss_asr/tog_atss_dior_asr.txt", 'r') as f:
    lines3 = f.readlines()
lines3 = list(map(float, lines3))
with open("/disk2/lhd/codes/attack/work_dirs/figures/atss_asr/bim_atss_dior_asr.txt", 'r') as f:
    lines4 = f.readlines()
lines4 = list(map(float, lines4))

x = np.arange(0.5, 1, 0.01)
y1 = np.asarray(lines1)
y2 = np.asarray(lines2)
y3 = np.asarray(lines3)
y4 = np.asarray(lines4)
plt.plot(x, y1, color='r')
plt.plot(x, y2, color='g')
plt.plot(x, y3, color='b')
plt.plot(x, y4)
plt.savefig("work_dirs/figures/test.png")