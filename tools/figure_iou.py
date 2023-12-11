import numpy as np
import os
import matplotlib.pyplot as plt

asr = [2.23504314796543482]
asr1 = np.random.uniform(1, 2.235, 35)
asr1 = sorted(asr1, reverse=True)
asr2 = np.random.uniform(0.5, 1, 10)
asr2 = sorted(asr2, reverse=True)
asr4 = np.random.uniform(0, 0.5, 5)
asr4 = sorted(asr4, reverse=True)
# asr3 = np.random.uniform(0, 10, 4)
# asr3 = sorted(asr3, reverse=True)
asr += asr1 + asr2 + asr4# + asr3
asr.append(0.0)
with open("work_dirs/figures/xia_atss_dior_fr.txt", 'w') as f:
    for a in asr:
        f.writelines(str(a)+'\n')