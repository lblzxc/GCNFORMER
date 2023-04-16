import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import obonet
import networkx as nx
import math
doid = open('doid.obo', 'r', encoding="utf-8")
HDO_Net = obonet.read_obo(doid)
def get_SV(disease, w):
    S = HDO_Net.subgraph(nx.descendants(HDO_Net, disease) | {disease})
    SV = dict()
    shortest_paths = nx.shortest_path(S, source=disease)
    for x in shortest_paths:
        SV[x] = math.pow(w, (len(shortest_paths[x]) - 1))
    return SV
def get_similarity(d1, d2, w):
    SV1 = get_SV(d1, w)
    SV2 = get_SV(d2, w)
    intersection_value= 0
    for disease in (set(SV1.keys()) & set(SV2.keys())):
        intersection_value = intersection_value + SV1[disease]
        intersection_value = intersection_value + SV2[disease]
    return intersection_value / (sum(SV1.values()) + sum(SV2.values()))
def getDiSiNet(dilen, diseases, w):
    diSiNet = np.zeros((dilen, dilen))
    for d1 in range(dilen):
        if diseases[d1] in HDO_Net.nodes:
            for d2 in range(d1 + 1, dilen):
                if diseases[d2] in HDO_Net.nodes:
                    diSiNet[d1, d2] = diSiNet[d2, d1] = get_similarity(diseases[d1], diseases[d2], w)
    return diSiNet
def PBPA(RNA_i, RNA_j, di_sim, rna_di):
    diseaseSet_i = rna_di[RNA_i] > 0
    diseaseSet_j = rna_di[RNA_j] > 0
    diseaseSim_ij = di_sim[diseaseSet_i][:, diseaseSet_j]
    ijshape = diseaseSim_ij.shape
    if ijshape[0] == 0 or ijshape[1] == 0:
        return 0
    return (sum(np.max(diseaseSim_ij, axis=0)) + sum(np.max(diseaseSim_ij, axis=1))) / (ijshape[0] + ijshape[1])
def getRNASiNet(RNAlen, diSiNet, rna_di):
    RNASiNet = np.zeros((RNAlen, RNAlen))
    for i in range(RNAlen):
        for j in range(i + 1, RNAlen):
            RNASiNet[i, j] = RNASiNet[j, i] = PBPA(i, j, diSiNet, rna_di)
    return RNASiNet
def cancatenate(lnclen, dilen, milen, Lnc_di, Lnc_mi, Mi_di, lncSiNet, diSiNet, miSiNet):
    A = np.zeros((lnclen + dilen + milen, lnclen + dilen + milen))
    A[: lnclen, lnclen: lnclen + dilen] = Lnc_di
    A[lnclen: lnclen + dilen, : lnclen] = Lnc_di.T
    A[: lnclen, lnclen + dilen: ] = Lnc_mi
    A[lnclen + dilen: , : lnclen] = Lnc_mi.T
    A[lnclen: lnclen + dilen, lnclen + dilen: ] = Mi_di.T
    A[lnclen + dilen: , lnclen: lnclen + dilen] = Mi_di
    A[: lnclen, : lnclen] = lncSiNet
    A[lnclen: lnclen + dilen, lnclen: lnclen + dilen] = diSiNet
    A[lnclen + dilen: , lnclen + dilen: ] = miSiNet
    return A
lncRNAs = list(pd.read_csv('./Data/LncName.csv', header=None)[0])
miRNAs = list(pd.read_csv('./Data/MiName.csv', header=None)[0])
diseases = pd.read_csv('./Data/DisName.csv', header=None)[0]
lnc_mi = pd.read_csv('./Data/Lnc_mi.csv.csv', header=None)
lnc_di = pd.read_csv('./Data/Lnc_di.csv.csv', header=None)
mi_di = pd.read_csv('./Data/Mi_di.csv.csv', header=None)
selected = np.sum(lnc_di, axis=0) > 0
diseases = list(diseases[selected])
lnc_di = lnc_di.values[:, selected]
lnc_mi = lnc_mi.values
mi_di = mi_di.values[:, selected]
print(lnc_di.shape, lnc_mi.shape, mi_di.shape)
print(np.sum(lnc_di), np.sum(lnc_mi), np.sum(mi_di))
pd.DataFrame(lnc_di, index=lncRNAs, columns=diseases).to_csv('data/lnc_di.csv')
pd.DataFrame(lnc_mi, index=lncRNAs, columns=miRNAs).to_csv('data/lnc_mi.csv')
pd.DataFrame(mi_di, index=miRNAs, columns=diseases).to_csv('data/mi_di.csv')
dilen = len(diseases)
lnclen = len(lncRNAs)
milen = len(miRNAs)
diSiNet = getDiSiNet(dilen=dilen, diseases=diseases, w=0.5)
plt.matshow(diSiNet, cmap= plt.cm.coolwarm, vmin=0, vmax=1)
miSiNet = getRNASiNet(RNAlen=milen, diSiNet=copy.copy(diSiNet), rna_di=copy.copy(mi_di))
plt.matshow(miSiNet, cmap= plt.cm.coolwarm, vmin=0, vmax=1)
lncSiNet = getRNASiNet(RNAlen=lnclen, diSiNet=copy.copy(diSiNet), rna_di=copy.copy(lnc_di))
Sim_Adjacency = cancatenate(lnclen=lnclen, dilen=dilen, milen=milen, lnc_di=lnc_di, lnc_mi=lnc_mi, mi_di=mi_di, lncSiNet=lncSiNet, diSiNet=diSiNet, miSiNet=miSiNet)#(1147,1147)
plt.matshow(Sim_Adjacency, cmap= plt.cm.coolwarm, vmin=0, vmax=1)
np.savetxt('data/data_sim_result.csv', Sim_Adjacency)
DisLnc = np.loadtxt('./data/Lnc_di.csv', dtype=np.float64)
x1 = []
x2 = []
for i in range(DisLnc.shape[0]):
    for j in range(DisLnc.shape[1]):
        if DisLnc[i][j] == 1:
            x1.append(j)
            x2.append(i)
with open("diseaseFeature.txt") as xh:
    with open('lncRNAFeature.txt') as yh:
        with open("PositiveSampleFeature.txt", "w") as zh:
            xlines = xh.readlines()
            ylines = yh.readlines()
            for k in range(len(x1)):
                for i in range(len(xlines)):
                    for j in range(len(ylines)):
                        if i == x1[k] and j==x2[k]:
                            line = xlines[i].strip() + ' ' + ylines[j]
                            zh.write(line)
DisLnc = pd.read_csv('./data/Lnc_di.csv', dtype=np.float64)
x1 = []
x2 = []
for i in range(DisLnc.shape[0]):
    for j in range(DisLnc.shape[1]):
        if DisLnc[i][j] == 0:
            x1.append(j)
            x2.append(i)
with open("./data/lncfeature.txt") as xh:
    with open('./data/diseasefeature.txt') as yh:
        with open("Negative_samples.txt", "w") as zh:
            xlines = xh.readlines()
            ylines = yh.readlines()
            selected = random.sample(range(len(x1)), 1569)
            for k in range(len(selected)):
                for i in range(len(xlines)):
                    for j in range(len(ylines)):
                        if i == x1[selected[k]] and j==x2[selected[k]]:
                            line = xlines[i].strip() + ' ' + ylines[j]
                            zh.write(line)
PositiveSampleFeature = np.loadtxt('Positive_samples.txt', dtype=np.float64)
NegativeSampleFeature = np.loadtxt('Negative_samples.txt', dtype=np.float64)
Features = []
Features.extend(PositiveSampleFeature)
Features.extend(NegativeSampleFeature)
Features= np.array(Features)

