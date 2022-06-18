from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *
import numpy as np

def get_cutout_data(xrange, yrange, zrange, name='region', save=True):
    '''fetching both raw images as well as annotated data for cutout data ranges'''

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
    cutout_data_anno = rmt.get_cutout(data_actual, data_params['prasad_res'], xrange, yrange, zrange)
    



    raw_chan_setup = ChannelResource(
        data_params['prasad_channel'], data_params['prasad_coll'], 'prasad2020', 
        type='image', datatype='uint8')
    data_actual = rmt.get_project(raw_chan_setup)
    cutout_data_img = rmt.get_cutout(data_actual, data_params['prasad_res'], xrange, yrange, zrange)




    

    cutout_data_img = cutout_data_img.astype(np.int32)
    cutout_data_anno = cutout_data_anno.astype(np.int32)
    
    return cutout_data_img,cutout_data_anno





