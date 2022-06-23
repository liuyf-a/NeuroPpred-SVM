from sklearn.preprocessing import MinMaxScaler
import numpy as np


f = open('./neuropeptides/aap_loggeneral.txt', 'r')
minmax_scaler = open('./neuropeptides/aap_minmaxscaler_general.txt', 'w')
value = []
ff = f.readlines()
for i in ff:
    # print(i.split(' ')[1])
    value.append(i.split(' ')[1])
    # scaler.write(i.split(' ')[0])
value = np.array(value).reshape(len(value), 1)

tool = MinMaxScaler(feature_range=(0, 1))
data = tool.fit_transform(value)
data = data.tolist()

for i in range(len(ff)):
    minmax_scaler.write(ff[i].split(' ')[0])
    for x in data[i]:
        minmax_scaler.write(' ' + str(x) + '\n')

f.close()
minmax_scaler.close()
