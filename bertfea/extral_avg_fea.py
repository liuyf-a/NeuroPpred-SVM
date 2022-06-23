import json
import sys
import joblib
import numpy as np


input_file = sys.argv[1]
output_file = sys.argv[2]
label = sys.argv[3]

# fea = open('train_select_fea.txt','w')

tmp = []
for line in open(input_file,'r'):
    tmp.append(json.loads(line))

data_save_list = []
for i in range(len(tmp)):
    # 提取除[UNK]和[SEP]标记的所有特征
    data_save_list.append([])
    for f in range(len(tmp[i]['features'])):
        # print(tmp[i]['features'][f]['token'])
        if tmp[i]['features'][f]['token'] == '[CLS]' or tmp[i]['features'][f]['token'] == '[SEP]':
            pass
        else:
            data_save_list[i].append(tmp[i]['features'][f]['layers'][0]['values'])

data_save_list = np.array(data_save_list)

f = open(output_file, 'w')
for i in range(len(data_save_list)):
    f.write(label)
    ave_data = np.average(data_save_list[i], axis=0)
    for j in range(len(ave_data)):
        f.write(',' + str(ave_data[j]))
    f.write('\n')

f.close()

