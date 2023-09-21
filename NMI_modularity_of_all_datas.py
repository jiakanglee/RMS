from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
import pandas as pd
# Import libraries
import sklearn
import numpy as np
import pandas as pd #For reading dataset files
import networkx as nx #For network creation/analysis
import community.community_louvain as cv
from networkx.algorithms import community
import karateclub
import infomap as im
from functools import *
from sklearn import metrics
import matplotlib.pyplot as plt #For plotting graphs
from sklearn.metrics import fowlkes_mallows_score
import networkx.algorithms.community as nx_comm

######数据集参数设置及基本数据的处理##############################################################################################
all_edges=0.0  #所有节点权值之和
k=[]   #每个节点的度数
d=[]   #社区内部所有点的度数之和
inner_edges=[]
node_map=[]
true_map=[]
test_choice=int(input("1.cellphone  2.enron  3.lesmis   4.netscience  5.Sandi_authors.csv   6.US_airport.csv  7.LFR1  8.dolphins.csv  9.LRF_750  10.LRF_1000  11.football:  12.book     "))
mat_size_map=[0,400,148,77,1589,86,500,500,62,750,1000,115,105]
mat_size=mat_size_map[test_choice]
# max_num=[0,41000,820,100,2.6,8,2254000,17.0,2.0,17.0,15.0,2.0]
# # equal_num=[0,410.0,82.0,10.0,2.6,8.0,2254,17.0,2.0,17.0,15.0,2.0]
# # div_num=[1,100.0,10.0,10.0,1,1,1000.0,1.0,1.0,1.0,1.0,1.0]
dataset_choice=['CellPhoneCallRecords.csv','email-Enron-full-proj-graph.csv','lesmis.csv','netscience.csv','Sandi_authors.csv','US_airport.csv','LFR.csv','dolphins.csv','LFR_750.csv','LFR_1000.csv','misc-football.csv',"Kreb's_book.csv"]
ground_truth=['community.csv','Dolphins_label.csv','community_750.csv','community_1000.csv','football_ground_truth.csv',"Kreb's_book_ground_truth.csv"]
truth_num=test_choice-7
node_from_one=[1,0,1,1,0,1]
for i in range(0,1600):
    k.append(0.0)
    d.append(0.0)
    inner_edges.append(0.0)

data = pd.read_csv(dataset_choice[test_choice-1])
matrix = np.zeros((mat_size,mat_size),dtype=float)
###################################################################################################################
for i in range(mat_size):
    true_map.append(0)

#此处在计算nmi时才开启
# datas = pd.read_csv(ground_truth[truth_num])
# for items in datas.iterrows():
#     true_map[items[1][0]-node_from_one[truth_num]]= items[1][1]

def map_node2label(classes,size):
    node_map.clear()
    for i in range(size):
        node_map.append(0)
    i=1
    for item in classes:
        for node in item:
            node_map[node]=i
        i=i+1


def convert(x,t): #for 库karateclub only,convert and map
    temp1 = {}
    for i in range(len(t)):
        temp1[t[i]] = i
    return temp1[x]

def Count_Inner_Edges(com):
    i=0
    for t in range(0, 1600):
        d[t]=0
        inner_edges[t]=0

    for item in com:
        for ii in item:
            d[i]=d[i]+k[ii]
            for jj in item:
                if matrix[ii][jj]!=0:
                    inner_edges[i]=inner_edges[i]+matrix[ii][jj]
        i=i+1


def Get_Modularity(class_num):
    modularity=0
    for i in range(class_num):
        modularity=modularity+inner_edges[i]/all_edges-(d[i]/all_edges)**2
    return modularity

def getValue(x):
    label=np.zeros(mat_size)
    i=0
    for item in x:
       for pos in item:
           label[pos]=i
       i=i+1
    return label.tolist()

#################算法部分####################################
def distance(d1, d2):
    res = 0
    for i in range(len(d1)):
        # 将每一行数据两两对应相减，计算距离
        res += (float(d1[i]) - float(d2[i])) ** 2
    return res ** 0.5

def Distance_Matrix(data):
    arr = [list(data[it]) for it in data]
    points = [[item[i] for item in arr] for i in range(len(arr[0]))]
    n = len(points)
    distance_matrix = [[0 for item in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            distance_matrix[i][j] = distance(points[i], points[j])
    return distance_matrix

def K_matrix(distance_matrix, K):
    disK = [sorted(list(filter(lambda x: x != point, range(len(distance_matrix)))),
                   key=lambda x: distance_matrix[point][x],reverse=True)[:K] for point in range(len(distance_matrix))]
    disK = [{'sum': sum(map(lambda x: distance_matrix[i][x], disK[i])), 'points': disK[i]} for i in range(len(disK))]
    return disK

class MeanShift_knn(object):

    def __init__(self, data, dis_max):
        if len(dis_max) == 0:
            self.dis_matrix = Distance_Matrix(data)
        else:
            self.dis_matrix = dis_max
        self.labels_ = {}
        for i in range(0,1600):
            self.labels_[i] = []
        self.label = []
        self.label_Modu = []
        self.DBIscore = 0
        self.CHscore = 0
        self.SCscore = 0

    def fit(self, K):
        disK = K_matrix(self.dis_matrix, K)
        s1, s2 = set(range(len(disK))), set()
        while True:
            for i in s1:
                temp = (reduce(lambda x, y: x if disK[x]['sum'] > disK[y]['sum'] else y, [i] + disK[i]['points']))
                s2.add(temp)
            if s1 == s2:
                break
            s1, s2 = s2, set()
        self.centers_ = list(s1)
        for center in self.centers_:
            self.labels_[center] = center
        s1, s2 = set(range(len(disK))), set()
        listemp = []
        for i in range(len(disK)):
            listemp.append([])

        while True:
            for i in s1:
                temp = (reduce(lambda x, y: x if disK[x]['sum'] > disK[y]['sum'] else y, [i] + disK[i]['points']))
                s2.add(temp)
                if i not in self.centers_:
                    listemp[i].append(temp)
                    for t in range(len(disK)):
                        if i in listemp[t]:
                            listemp[t].append(temp)

            if s1 == s2:
                for i in range(len(disK)):
                    if listemp[i] == []:
                        listemp[i].append(i)
                for i in range(len(disK)):
                    tem = listemp[i]
                    self.labels_[i] = tem[-1]
                break
            s1, s2 = s2, set()

        for i in range(len(matrix)):
            self.label.append(self.labels_[i])
        for center in self.centers_:

           tempset = set({})
           for i in range(len(matrix)):
               if self.labels_[i]== center:
                   tempset.add(i)
           self.label_Modu.append(tempset)

        #self.DBIscore = sklearn.metrics.davies_bouldin_score(matrix, Mk.label)
        #self.CHscore = sklearn.metrics.calinski_harabasz_score(matrix, Mk.label)
        #self.SCscore = sklearn.metrics.silhouette_score(matrix, Mk.label)
        self.label_Modu = tuple(self.label_Modu)

###矩阵处理
for items in data.iterrows():
    if test_choice==1:
        matrix[items[1][0]][items[1][1]] += items[1][3]
    elif (test_choice==4) or (test_choice==6) or (test_choice==7) or (test_choice==9) or (test_choice==10) :
        matrix[int(items[1][0]) - 1][int(items[1][1]) - 1] += items[1][2]
    elif test_choice==8 or test_choice==12:
        matrix[items[1][0]][items[1][1]] += items[1][2]
    else:
        matrix[items[1][0]-1][items[1][1]- 1] += items[1][2]

for i in range(0,mat_size):
    for j in range(i + 1, mat_size):
        if test_choice==2 or test_choice==8:
            if matrix[i][j]!=0:
                matrix[j][i] = matrix[i][j]
            else:
                matrix[i][j]=matrix[j][i]
        else:
            matrix[i][j] += matrix[j][i]
            matrix[j][i] = matrix[i][j]

for i in range(0,mat_size):
    for j in range(0,mat_size):
        if matrix[i][j] != 0:
            all_edges+= matrix[i][j]
            k[i] = k[i] + matrix[i][j]

temp = []
temp_tuple = ()
for i in range(mat_size):
    for j in range(i+1,mat_size):
        if matrix[i][j] == 0:
          continue
        else:
            temp.append((i,j,matrix[i][j]))
G = nx.Graph()
G.add_weighted_edges_from(temp)
nx.draw_networkx(G)
# G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
def labelpropagation(graph):
    partition = community.label_propagation_communities(graph)
    print((partition))
    temp = ()
    for x in partition:
        temp += (x,)
    class_num = len(temp)
    print("类别为:",len(temp))
    print(temp)
    print("Label Propagation results:")
########################计算模块度
    Count_Inner_Edges(temp)
    t=Get_Modularity(class_num)

########################计算NMI
    # map_node2label(temp, mat_size)
    # t=metrics.normalized_mutual_info_score(node_map, true_map)

    print(t)
def Louvain_partition(graph):
    partition = cv.best_partition(graph)
    print(partition)
    temp_set = set()
    temp = ()
    for x in partition:
        temp_set.add(partition[x])
    for i in range(len(temp_set)):
        temp +=(set({}),)
    for x in partition:
        temp[partition[x]].add(x)
    print(temp)
    class_num = len(temp)
    print("Louvian results:")
    print("类别为:", class_num)
########################计算模块度
    Count_Inner_Edges(temp)
    t=Get_Modularity(class_num)

########################计算NMI
    # map_node2label(temp, mat_size)
    # t=metrics.normalized_mutual_info_score(node_map, true_map)

    print(t)
def SCD_partition(graph):
    #graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default')
    partition = karateclub.SCD(seed=25)
    partition.fit(graph)
    pase = partition.get_memberships()
    temp = set()
    temp_final = ()
    for x in pase:
        temp.add(pase[x])
    for i in range(len(temp)):
        temp_final += (set({}),)
    for x in pase:
        temp_final[convert(pase[x], list(temp))].add(x)
    class_num = len(temp_final)
    Count_Inner_Edges(temp_final)
    print(temp_final)
    print("类数为:",len(temp_final))
    print("SCD results:")
########################计算模块度
    Count_Inner_Edges(temp_final)
    t=Get_Modularity(class_num)

########################计算NMI
    # map_node2label(temp_final, mat_size)
    # t=metrics.normalized_mutual_info_score(node_map, true_map)

    print(t)


def infomap_partition(graph):
    ip = im.Infomap()
    #graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default')
    ip.add_networkx_graph(graph)
    ip.run()
    temp_set = set()
    temp_final = ()
    communities = ip.get_modules()
    nx.set_node_attributes(graph, communities, 'community')
    for x in communities:
        temp_set.add(communities[x])
    for i in range(len(temp_set)):
        temp_final += (set({}),)
    for x in communities:
        temp_final[communities[x] - 1].add(x)
    print(temp_final)
    class_num = len(temp_final)
    print("类数为:",len(temp_final))
    print("Infomap results:")
########################计算模块度
    Count_Inner_Edges(temp_final)
    t=Get_Modularity(class_num)

#######################计算NMI
    # map_node2label(temp_final, mat_size)
    # t=metrics.normalized_mutual_info_score(node_map, true_map)

    print(t)

def EdMot(graph):
    #graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default')
    partition = karateclub.EdMot()
    partition.fit(graph)
    pase = partition.get_memberships()
    temp = set()
    temp_final = ()
    for x in pase:
        temp.add(pase[x])
    for i in range(len(temp)):
        temp_final += (set({}),)
    for x in pase:
        temp_final[convert(pase[x], list(temp))].add(x)
    class_num = len(temp_final)
    print(temp_final)
    print("类数为:",len(temp_final))
    print("EdMot results:")
########################计算模块度
    Count_Inner_Edges(temp_final)
    t=Get_Modularity(class_num)

########################计算NMI
    # map_node2label(temp_final, mat_size)
    # t=metrics.normalized_mutual_info_score(node_map, true_map)

    print(t)

def GirvanNewman_partition(graph):
    partition = nx.algorithms.community.girvan_newman(graph)
    modularity = []
    i=1
    for x in partition:
        i=i+1
        temp = x
        temp = tuple(temp)
        print(len(temp))
        class_num = len(temp)
        print("Girvan Newman results:")
########################计算模块度
        Count_Inner_Edges(temp)
        t=Get_Modularity(class_num)

########################计算NMI
        # map_node2label(temp,mat_size)
        # t=metrics.normalized_mutual_info_score(node_map,true_map)

        print(t)
        modularity.append(t)
    # f=open("gn_airport.txt","w")
    # f.write(str(modularity))
    # f.close()
    # print(len(modularity))
    # # for the draw
    # y = modularity
    # x = np.arange(1, i, 1)
    # plt.figure(num=None, figsize=(8, 5))
    # plt.plot(x, y, color='red', marker="o")
    # plt.tick_params(labelsize=16)
    # plt.xlabel('k', fontsize=16)
    # plt.ylabel('Modularity', fontsize=16)
    # plt.rcParams['font.family'] = ['sans-serif']
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # # plt.legend(['数据点','预测点'],loc = 'upper center',fontsize = 16,frameon=False,ncol=2)
    # # plt.ylim(0,55)
    # plt.show()

def Meanshift_Knn():
    modularity = []
    for i in range(1, 32):
        Mk = MeanShift_knn(data, matrix)
        Mk.fit(i)
        class_num = len(set(Mk.label))
        print(Mk.label_Modu)
        print(len(set(Mk.label)))
########################计算模块度
        Count_Inner_Edges(Mk.label_Modu)
        t=Get_Modularity(class_num)

########################计算NMI
        # map_node2label(Mk.label_Modu,mat_size)
        # t=metrics.normalized_mutual_info_score(node_map,true_map)

        print(t)
        modularity.append(t)
    # f=open("msk_airport.txt","w")
    # f.write(str(modularity))
    # f.close()
    # f=open("msk_modularity.txt","r")
    # file=f.readlines()
    # print(file[0])
    # f.close()
    y = modularity
    x = np.arange(1, 32, 1)
    plt.figure(num=None, figsize=(8, 5))
    plt.plot(x, y, color='red', marker="o")
    # plt.plot(x,y2,color = 'blue',marker = "x")
    plt.tick_params(labelsize=16)
    plt.xlabel('k值', fontsize=16)
    plt.ylabel('Modularity', fontsize=16)
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.legend(['数据点','预测点'],loc = 'upper center',fontsize = 16,frameon=False,ncol=2)
    # plt.ylim(0,55)
    plt.show()


def GEMSEC(graph):
    #graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default')
    nmi=[]
    for i in range(1,mat_size+1):
        partition = karateclub.GEMSEC(clusters=i)
        partition.fit(graph)
        pase = partition.get_memberships()
        temp = set()
        temp_final = ()
        for x in pase:
            temp.add(pase[x])
        for i in range(len(temp)):
            temp_final += (set({}),)
        for x in pase:
            temp_final[convert(pase[x], list(temp))].add(x)
        class_num=len(temp_final)
        print(temp_final)
        print("类数为:", len(temp_final))
        print("GEMSEC results:")
        # ########################计算模块度
        Count_Inner_Edges(temp_final)
        t=Get_Modularity(class_num)

        #######################计算NMI
        # map_node2label(temp_final, mat_size)
        # t = metrics.normalized_mutual_info_score(node_map, true_map)
        print(t)
        nmi.append(t)
    y = nmi
    f=open("gesmes_US_airport.txt","w")
    f.write(str(nmi))
    f.close()
    x = np.arange(1, mat_size+1, 1)
    plt.figure(num=None, figsize=(8, 5))
    plt.plot(x, y, color='red', marker="o")
    plt.tick_params(labelsize=16)
    plt.xlabel('k', fontsize=16)
    plt.ylabel('Modularity', fontsize=16)
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.legend(['数据点','预测点'],loc = 'upper center',fontsize = 16,frameon=False,ncol=2)
    # plt.ylim(0,55)
    plt.show()


# Meanshift_Knn()
GEMSEC(G)
# EdMot(G)