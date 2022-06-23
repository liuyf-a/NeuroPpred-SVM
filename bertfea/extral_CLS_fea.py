import json
import sys

input_file = sys.argv[1]
outpu_tfile = sys.argv[2]
label = sys.argv[3]

fea = open(outpu_tfile,'w')
tmp = []
for line in open(input_file,'r'):
    tmp.append(json.loads(line))

for i in range(len(tmp)):
    fea.write(label)
    s = len(tmp[i]['features'][0]['layers'])
    for j in range(s):
        # print(tmp[i]['features'][0]['layers'][j]['values'])
        ss = len(tmp[i]['features'][0]['layers'][j]['values'])
        for k in range(ss):
            # print(type(tmp[i]['features'][0]['layers'][k]['values']))
            fea.write(',' + str(tmp[i]['features'][0]['layers'][j]['values'][k]))
    fea.write('\n')

fea.close()


