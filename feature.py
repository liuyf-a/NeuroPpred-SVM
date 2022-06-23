from sklearn.calibration import CalibratedClassifierCV as cc, calibration_curve
from Bio import SeqIO
from pydpi.pypro import PyPro
import sys
import numpy as np
import warnings
import os
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")
import pandas as pd
from itertools import *

protein = PyPro()


def readAAP(file):  # read AAP features from the AAP textfile
    try:
        aapdic = {}
        aapdata = open(file, 'r')
        for l in aapdata.readlines():
            aapdic[l.split()[0]] = float(l.split()[1])
        aapdata.close()
        return aapdic
    except:
        print("Error in reading AAP feature file. Please make sure that the AAP file is correctly formatted")
        sys.exit()


def readAAT(file):  # read AAT features from the AAT textfile
    try:
        aatdic = {}
        aatdata = open(file, 'r')
        for l in aatdata.readlines():
            aatdic[l.split()[0][0:3]] = float(l.split()[1])
        aatdata.close()
        return aatdic
    except:
        print("Error in reading AAT feature file. Please make sure that the AAT file is correctly formatted")
        sys.exit()


def aap(pep, aapdic, avg):  # return AAP features for the peptides
    feature = []
    for a in pep:
        # print(a)
        if int(avg) == 0:
            score = []
            count = 0
            for i in range(0, len(a) - 1):
                try:
                    score.append(round(float(aapdic[a[i:i + 2]]), 4))
                    # score += float(aapdic[a[i:i + 3]])
                    count += 1
                except KeyError:
                    # print(a[i:i + 3])
                    score.append(float(-1))
                    # score += -1
                    count += 1
                    continue
            # averagescore = score / count
            feature.append(score)
        if int(avg) == 1:
            score = 0
            count = 0
            for i in range(0, len(a) - 1):
                try:
                    score += float(aapdic[a[i:i + 2]])
                    count += 1
                except KeyError:
                    score += -1
                    count += 1
                    continue
            if count != 0:
                averagescore = score / count
            else:
                averagescore = 0
            feature.append(round(float(averagescore), 4))
    return feature


def aat(pep, aatdic, avg):  # return AAT features for the peptides
    feature = []
    for a in pep:
        if int(avg) == 0:
            # print(a)
            score = []
            count = 0
            for i in range(0, len(a) - 2):
                try:
                    score.append(round(float(aatdic[a[i:i + 3]]), 4))
                    # score += float(aapdic[a[i:i + 3]])
                    count += 1
                except KeyError:
                    # print(a[i:i + 3])
                    score.append(float(-1))
                    # score += -1
                    count += 1
                    continue
            # averagescore = score / count
            feature.append(score)
        if int(avg) == 1:
            score = 0
            count = 0
            for i in range(0, len(a) - 2):
                try:
                    score += float(aatdic[a[i:i + 3]])
                    count += 1
                except KeyError:
                    score += -1
                    count += 1
                    continue
            # print(a, score)
            if count != 0:
                averagescore = score / count
            else:
                averagescore = 0
            feature.append(round(float(averagescore), 4))
    return feature


letters = list('ACDEFGHIKLMNPQRSTVWY')
Amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q',
               'R', 'S', 'T', 'V', 'W', 'Y']

Amino_acids_ = list(product(Amino_acids, Amino_acids))
Amino_acids_ = [i[0] + i[1] for i in Amino_acids_]
"""GGAP"""
def GGAP(seqs):
    seqs_ = []
    for seq in seqs:
        GGAP_feature = []
        num = 0
        for i in range(len(seq) - 3):
            GGAP_feature.append((seq[i] + seq[i + 3]))

        seqs_.append([GGAP_feature.count(i) / (len(seq) - 3) for i in Amino_acids_])

    return seqs_


"""ASDC"""
def ASDC(seqs):
    seqs_ = []
    for seq in seqs:
        ASDC_feature = []
        skip = 0
        for i in range(len(seq)):
            ASDC_feature.extend(Skip(seq, skip))
            skip += 1
        seqs_.append([ASDC_feature.count(i) / len(ASDC_feature) for i in Amino_acids_])
    return seqs_


def Skip(seq, skip):
    element = []
    for i in range(len(seq) - skip - 1):
        element.append(seq[i] + seq[i + skip + 1])
    return element


"""PSAAC"""
def PSAAC(seqs):
    seqs_ = []
    PSAAC_profile_forward = []
    PSAAC_profile_backward = []
    forward_seq = []
    backward_seq = []
    i = 0
    for seq in seqs:
        forward_seq.append(list(seq[:5]))
        backward_seq.append(list(seq[-5:]))

    for position in range(5):
        PSAAC_profile_forward.append(
            [list(np.array(forward_seq)[:, position]).count(amino) / len(seqs) for amino in Amino_acids])

    for position in range(5):
        PSAAC_profile_backward.append(
            [list(np.array(backward_seq)[:, position]).count(amino) / len(seqs) for amino in Amino_acids])

    for seq in forward_seq:
        num = 0
        new_seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for amino in seq:
            index_ = Amino_acids.index(amino)
            new_seq[index_] = np.array(PSAAC_profile_forward)[num, index_]
            num += 1

        seqs_.append(new_seq)

    for seq in backward_seq:
        num = 0
        new_seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for amino in seq:
            index_ = Amino_acids.index(amino)
            new_seq[index_] = np.array(PSAAC_profile_backward)[num, index_]
            num += 1

        seqs_[i].extend(new_seq)
        i += 1
    return seqs_


def CTD(pep):  # Chain-Transition-Ditribution feature
    feature = []
    name = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        ctd = protein.GetCTD()
        feature.append(list(ctd.values()))
        name = list(ctd.keys())
    return feature, name


def AAC(pep):  # Single Amino Acid Composition feature
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        aac = protein.GetAAComp()
        feature.append(list(aac.values()))
        name = list(aac.keys())
    return feature, name


def DPC(pep):  # Dipeptide Composition feature
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        dpc = protein.GetDPComp()
        feature.append(list(dpc.values()))
        name = list(dpc.keys())
    return feature, name


def PAAC(pep):
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        # paac=protein.GetMoranAuto()
        paac = protein.GetPAAC(lamda=4)
        feature.append(list(paac.values()))
        name = list(paac.keys())
    return feature, name


def bertfea(file):
    feature = []
    with open(file, 'r') as f:
        for line in f:
            line = line.split(',')
            feature.append([float(x) for x in line[1:769]])
    return feature


def QSO(pep):
    feature = []
    for seq in pep:
        protein.ReadProteinSequence(seq)
        # paac=protein.GetMoranAuto()
        qso = protein.GetQSO(maxlag=5)
        feature.append(list(qso.values()))
        name = list(qso.keys())
    return feature, name
