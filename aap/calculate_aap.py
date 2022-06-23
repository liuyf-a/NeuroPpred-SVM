import itertools
import math

def count_aap(filename):
    vocab = [''.join(xs) for xs in itertools.product('ARNDCQEGHILKMFPSTWYV', repeat=2)]
    n_dict = {}
    for i in range(len(vocab)):
        n_dict[vocab[i]] = 0
    num = 0
    f = open(filename, 'r')
    for line in f.readlines():
        if line[0] != '>':
            for i in range(len(line.strip()) - 1):
                x = line[i:i+2]
                num = num + 1
                # print(x)
                if x in n_dict.keys():
                    n_dict[x] = n_dict[x] + 1
                else:
                    n_dict[x] = 0
    return n_dict, num

#pos_dict, pos_num = count_aap('../source_data/amp/dbaasp_train.txt')
pos_dict, pos_num = count_aap('../datasets/neuropeptides/Pos_train_fasta.txt')
neg_dict, neg_num = count_aap('../source_data/uniport_swiss-port/uniprot.fasta')

print(pos_num)
print(neg_num)

aap = open('./neuropeptides/aap_general.txt', 'w')
log_aap = open('./neuropeptides/aap_loggeneral.txt', 'w')
for k1 ,v1 in pos_dict.items():
    for k2, v2 in neg_dict.items():
        if k1 == k2:
            if v2 == 0 or v1 == 0:
                aap.write(k1 + ' ' + str(0) + '\n')
                log_aap.write(k1 + ' ' + str(0) + '\n')
            else:
                score = (v1/pos_num) / (v2/neg_num)
                aap.write(k1 + ' ' + str(score) + '\n')
                log_score = math.log((v1/pos_num) / (v2/neg_num))
                log_aap.write(k1 + ' ' + str(log_score) + '\n')
'''
for k1, v1 in pos_dict.items():
    if k1 not in neg_dict.keys():
        aap.write(k1 + ' ' + str(0) + '\n')
        log_aap.write(k1 + ' ' + str(0) + '\n')
for k2, v2 in neg_dict.items():
    if k2 not in pos_dict.keys():
        aap.write(k2 + ' ' + str(0) + '\n')
        log_aap.write(k2 + ' ' + str(0) + '\n')
'''
aap.close()
log_aap.close()
