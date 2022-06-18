import numpy as np
import skimage
import nrrd
from skimage import measure
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph, KDTree
from skimage import io
import cc3d
import pandas as pd
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *



def extract_cell_stats(anno_data):


    anno_data[anno_data != 2] = 0
    labels_in = anno_data
    
    connectivity = 6 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    labels_out = cc3d.connected_components(labels_in, connectivity=connectivity)

    samples, _, imsize = anno_data.shape
    csize = int(imsize/2)
    
    labels = np.zeros((4*samples,csize,csize), dtype = 'uint32')
    for i in range(4*samples):
        slice_id, corner_id = divmod(i, 4)
        if corner_id==0:
            labels[i] = labels_out[slice_id, :csize, :csize]
        elif corner_id==1:
            labels[i] = labels_out[slice_id, :csize, -csize:]
        elif corner_id==2:
            labels[i] = labels_out[slice_id, -csize:, :csize]
        else:
            labels[i] = labels_out[slice_id, -csize:, -csize:]
            
    CellStatsList = np.zeros((4*samples,6))
    for i in range(4*samples):
        props =  measure.regionprops(labels[i])

        #Image number
        CellStatsList[i,0] = i + 1

        #Cell count
        CellStatsList[i,1]= len(props)

        #Average distance to nearest neighbor
        CellCoor = np.zeros((len(props),2))

        num = len(props)
        if num < 2:
            CellStatsList[i,2] = 64
            continue

        for j in range(num):
            CellCoor[j,0] = props[j].centroid[0]
            CellCoor[j,1] = props[j].centroid[1]


        kdt = KDTree(CellCoor, leaf_size=30, metric='euclidean')
        dist = kdt.query(CellCoor, k = 2, return_distance=True)[0]

        dist[dist!= 0]
        avg_dist = np.mean(dist)
        CellStatsList[i,2] = avg_dist


        #kdt = KDTree(CellCoor, leaf_size=30, metric='euclidean')
        #dist = kdt.query(CellCoor, k = 4, return_distance=True)[0]

        dist[dist!= 0]
        avg_dist = np.mean(dist)
        CellStatsList[i,3] = avg_dist

        #Average cell size
        s=0
        for j in range(num):
            s += props[j].area
        CellStatsList[i,4] = s/num

        if len(props) > 0:
            CellStatsList[j,5] = (props[0].area)/(anno_data[j].size)*100


    cols=["Image Number","Number of Cells", "Avg Distance to NN", "Avg Distance to 3rd NN", "Avg Cell Size","Cell Pixel count"]
    stats_c = pd.DataFrame(data=CellStatsList, columns=cols)     
    return stats_c.to_numpy()
    
    
def extract_axon_stats(anno_data):

    anno_data[anno_data != 3] = 0
    labels_out = anno_data

    samples, _, imsize = anno_data.shape
    csize = int(imsize/2)
    
    labels = np.zeros((4*samples,csize,csize), dtype = 'uint32')
    for i in range(4*samples):
        slice_id, corner_id = divmod(i, 4)
        if corner_id==0:
            labels[i] = labels_out[slice_id, :csize, :csize]
        elif corner_id==1:
            labels[i] = labels_out[slice_id, :csize, -csize:]
        elif corner_id==2:
            labels[i] = labels_out[slice_id, -csize:, :csize]
        else:
            labels[i] = labels_out[slice_id, -csize:, -csize:]
            
    anno_data = labels
    
    AxonStatsList = np.zeros((4*samples,2))
    for j in range(4*samples):

        props =  measure.regionprops(anno_data[j])      
        AxonStatsList[j,0] = j + 1
        if len(props) > 0:
            AxonStatsList[j,1] = (props[0].area)/(anno_data[j].size)*100


    cols=["Image Number","Percent of Pixels"]
    axon_stats_c = pd.DataFrame(data=AxonStatsList, columns=cols)    
    return axon_stats_c.to_numpy()
    
    
def extract_blood_stats(anno_data):

    anno_data[anno_data != 1] = 0
    labels_out = anno_data

    samples, _, imsize = anno_data.shape
    csize = int(imsize/2)
    
    labels = np.zeros((4*samples,csize,csize), dtype = 'uint32')
    for i in range(4*samples):
        slice_id, corner_id = divmod(i, 4)
        if corner_id==0:
            labels[i] = labels_out[slice_id, :csize, :csize]
        elif corner_id==1:
            labels[i] = labels_out[slice_id, :csize, -csize:]
        elif corner_id==2:
            labels[i] = labels_out[slice_id, -csize:, :csize]
        else:
            labels[i] = labels_out[slice_id, -csize:, -csize:]
            
    anno_data = labels
    
    BloodStatsList = np.zeros((4*samples,2))
    for k in range(4*samples):
        props =  measure.regionprops(anno_data[k])
        BloodStatsList[k,0] = k + 1

        BloodStatsList[k,1]= (props[0].area)/(anno_data[k].size)*100

    cols=["Image Number","Percent of Pixels"]
    blood_stats_c = pd.DataFrame(data=BloodStatsList, columns=cols)
    return blood_stats_c.to_numpy()
    
    
def extract_features(xrange, yrange, zrange, name='region', save=True):

    connection_params={
        'protocol': 'https',
        'host': 'api.bossdb.io',
        'token': 'public'  # allows read access to the general public
    }
    
    data_params={
    # path="prasad/prasad2020 [?]/image [?]/ChannelResource() [change]/get_or_create"
        'prasad_coll':'prasad',
        'prasad_exp' : 'prasad_analysis',
        'prasad_channel' : 'image',
        'anno_channel': 'pixel_labels',
        'prasad_coord' : 'prasad_prasad2020',
        'prasad_res' : 0,  #Native resolution of dataset is 1.17um, 1.17um, 1.17um per voxel
    }

    rmt = BossRemote(connection_params)

    print('Downloading annos...')
    # get raw data for reference
    data_setup = ChannelResource(
        data_params['anno_channel'], data_params['prasad_coll'], data_params['prasad_exp'],
        type='annotation', datatype='uint64', sources=['image'])
    data_actual = rmt.get_project(data_setup)
    cutout_data = rmt.get_cutout(data_actual, data_params['prasad_res'], xrange, yrange, zrange)

    cutout_data = cutout_data.astype(np.int32)
    
    print('Extracting cell stats...')
    stats_cell = extract_cell_stats(np.copy(cutout_data))
    
    print('Extracting axon stats...')
    stats_axon = extract_axon_stats(np.copy(cutout_data))
    
    print('Extracting blood stats...')
    stats_blood = extract_blood_stats(np.copy(cutout_data))

    
    if save:
        print('Saving...')
        np.save(name + "_cell_stats.npy", stats_cell)
        np.save(name + "_axon_stats.npy", stats_axon)
        np.save(name + "_blood_stats.npy", stats_blood)

    return stats_cell, stats_axon, stats_blood
        
