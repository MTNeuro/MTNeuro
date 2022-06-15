import os
import sys 
import torch
import math
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils.features import extract_cell_stats,extract_axon_stats,extract_blood_stats
from utils.get_cutouts import get_cutout_data
from utils.latents import get_latents, get_unsup_latents
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.decomposition import PCA
import argparse

'''This script takes in an encoder file path and computes R2 scores between embeddings and different Semantic features as part of of Task 3'''

'SET : provide trained encoder path'
encoder_file_path = ''

'SET:  encoder type either ssl, supervised, PCA or NMF'
encoder_type = 'ssl'

def task3_semantic_features(encoder_file_path,encoder_type):
    if encoder_type == 'ssl':
        ssl_encoder = 1
        unsupervised = 0
    elif encoder_type == 'supervised':
        ssl_encoder =  0
        unsupervised = 0
    elif encoder_type == 'PCA' 
        unsupervised = 1
        set_pca = 1
    elif encoder_type == 'NMF'
        unsupervised = 1
        set_pca = 0
    else:
        print("Incorrectly specified encoder type")

    'Specify cutout coordinates'
    xrange_list = [[3700,3956], [4600,4856],[3063,3319],[1543,1799]]
    yrange_list = [[500,756],[900,1156],[850,1106],[650,906]]
    class_list = ["striatum","Cortex","VP","ZI"]
    zrange = [110,470]



    data_array_raw = []
    data_array_anno = []
    label_array  = []
    up_sample = 4 

    for i in range(0,len(xrange_list)):
        cutout_data_raw,cutout_data_anno = get_cutout_data(xrange_list[i],yrange_list[i],zrange,name=class_list[i])
        
        data_raw = cutout_data_raw[:,:,:]
        data_anno = cutout_data_anno[:,:,:]
        data_array_raw = np.concatenate((data_array_raw,data_raw),axis =0 ) if len(data_array_raw) else data_raw 
        
        data_array_anno = np.concatenate((data_array_anno,data_anno),axis =0 ) if len(data_array_anno) else data_anno
        
        labels = i*np.ones(up_sample*len(data_raw)).reshape(-1,)
        label_array  = np.concatenate((label_array ,labels),axis =0) if len(labels) else labels_train
        

    print('Extracting cell stats...')

    stats_cell= extract_cell_stats(np.copy(data_array_anno))


    print('Extracting axon stats...')
    stats_axon = extract_axon_stats(np.copy(data_array_anno))

    print('Extracting blood stats...')
    stats_blood = extract_blood_stats(np.copy(data_array_anno))





    'get results for different encoders'
    if encoder_type == 'ssl' or encoder_type == 'supervised'
        embeddings = get_latents(data_array_raw,encoder_file_path,ssl_encoder)
    elif encoder_type == 'unsupervised'
        embeddings = get_unsup_latents(data_array_raw,set_pca)



    'Get linear readout scores'
    X = embeddings

    y = stats_blood[:,1]
    reg = LinearRegression().fit(X,y)
    blood_vsl_score = reg.score(embeddings,stats_blood[:,1])
    print("Blood Vessel Score : {}".format(blood_vsl_score ))


    y = stats_cell[:,1]
    reg = LinearRegression().fit(X,y)
    numb_cell = reg.score(embeddings,stats_cell[:,1])
    print("Cell count:{}".format(numb_cell))


    y = stats_cell[:,2]
    reg = LinearRegression().fit(X,y)
    avg_dist_nn_cell = reg.score(embeddings,stats_cell[:,2])
    print("cell stat 2 (Avg distance to NN) :{}".format(avg_dist_nn_cell ))



    y = stats_cell[:,4]
    reg = LinearRegression().fit(X,y)
    cell_size = reg.score(embeddings,stats_cell[:,4])
    print("Cell Size :{}".format(cell_size))


    y = stats_axon[:,1]
    reg = LinearRegression().fit(X,y)
    axon_rslt = reg.score(embeddings,stats_axon[:,1])
    print("Axon  : {}".format(axon_rslt ))


    '''Visualizing latents overlayed with semantic features'''


    print('Visualizing Brain Areas')

    scaler = preprocessing.StandardScaler().fit(embeddings)
    embeddings_norm = scaler.transform(embeddings)
    obj  = umap.UMAP(n_neighbors= 100, random_state=42).fit(embeddings_norm)
    pca_proj = obj.embedding_
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    num_categories = 4
    title_font = 35
    pdb.set_trace()
    for lab in range(num_categories):
        indices =  label_array==lab
        ax.scatter(pca_proj[indices,0],pca_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = class_list[lab] ,alpha=0.8)
    ax.legend(fontsize=20, markerscale=2,loc='lower right')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Brain Area', fontsize=35)
    plt.show()



    print("visualizing blood vessel pixels")
    blood = stats_blood[:,1]
    title_font = 35
    leg_font = 15
    min_blood, max_blood = np.min(blood), np.max(blood)
    cmap = cm.get_cmap('Reds')
    fig, ax = plt.subplots(figsize=(8,8))
    s=0
    for i in range(math.floor(min_blood), math.ceil(max_blood),2):
        s+=1
        indices = (blood >i) * (blood <= (i+2))
        ax.scatter(pca_proj[indices,0],pca_proj[indices,1], c=np.array(cmap(30*(s+1))).reshape(1,4), label = i ,alpha=0.8)
    ax.legend(fontsize=leg_font, markerscale=2)
    ax.set_title('Blood Vessels (%)', fontsize=title_font)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()



    print("visualizing Cell count")

    cell = stats_cell[:,1]
    min_cells, max_cells = np.min(cell), np.max(cell)
    cmap = cm.get_cmap('Greens')
    fig, ax = plt.subplots(figsize=(8,8))
    k = 0
    for i in range(0,40,4):
        k+=1
        indices = (cell>i) * (cell<= (i+4))
        ax.scatter(pca_proj[indices,0],pca_proj[indices,1], c=np.array(cmap(30*(k+1))).reshape(1,4), label = i ,alpha=0.8)
    ax.legend(fontsize=leg_font, markerscale=2)
    ax.set_title('Cell Count', fontsize=title_font)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()


    print("Visualizing Cell Sizes")
    cell = stats_cell[:,4]
    min_cells, max_cells = np.min(cell), np.max(cell)
    cmap = cm.get_cmap('Purples')
    fig, ax = plt.subplots(figsize=(8,8))
    q= 0 
    for i in range(math.floor(min_cells), math.ceil(max_cells),15):
        q+=1
        indices = (cell>i) * (cell<= (i+15))
        ax.scatter(pca_proj[indices,0],pca_proj[indices,1], c=np.array(cmap(30*(q+1))).reshape(1,4), label = i ,alpha=0.8)
    ax.legend(fontsize=leg_font, markerscale=2)
    ax.set_title('Cell Size', fontsize=title_font)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()




    print("Visualzing nearest NN")
    cell = stats_cell[:,2]
    min_cellsk1, max_cellsk1 = np.min(cell), np.max(cell)
    cmap = cm.get_cmap('Blues')
    fig, ax = plt.subplots(figsize=(8,8))
    k = 0
    for i in range(0,50,5):
        k+=1
        indices = (cell>i) * (cell< (i+5))
        ax.scatter(pca_proj[indices,0],pca_proj[indices,1], c=np.array(cmap(30*(k+1))).reshape(1,4), label = i ,alpha=0.8)
    ax.legend(fontsize=leg_font, markerscale=2)
    ax.set_title('Cell Avg Dist to 1st NN', fontsize=title_font)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()




    print("Visualizing axon pixels")
    axon = stats_axon[:,1]
    min_axon, max_axon = np.min(axon), np.max(axon)
    cmap = cm.get_cmap('Oranges')
    fig, ax = plt.subplots(figsize=(8,8))
    s=0
    for i in range(math.floor(min_axon), math.ceil(max_axon),10):
        s+=1
        indices = (axon>=i) * (axon< (i+10))
        ax.scatter(pca_proj[indices,0],pca_proj[indices,1], c=np.array(cmap(30*(s+1))).reshape(1,4), label = i ,alpha=0.8)
    ax.legend(fontsize=leg_font, markerscale=2)
    ax.set_title('Axons (%)', fontsize=title_font)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()



if __name  == 'main':
    parser = argparse.ArgumentParser(description='flags for training')
    parser.add_argument('--encoder_path', default=" ",
                        help='encoder file path. Required for encoder type SSL, supervised ')
    parser.add_argument('--encoder_type', default="ssl", required = True,
                        help='encoder type: Options SSL, supervised, PCA, NMF')
    args = parser.parse_args()
    task3_semantic_features(args.encoder_path,args.encoder_type)
