# dataset class for our dataset which is hosted on bossdb
# the array function comes from intern, which we use to interface with bossdb
# This dataset supports up to 4 regions, but each region must have the same size z,y,x cutout. 
from torch.utils.data import Dataset
from intern import array
import numpy as np
import torch
import matplotlib 
from os.path import exists
from requests.exceptions import HTTPError

class BossDBDataset(Dataset):

    def __init__(
        self, 
        task_config: dict,
        boss_config: dict, 
        mode="train",
        image_transform=None,
        mask_transform=None,
        retries = 5,
        download = True,
        download_path = './'
    ):
        x_cor = np.arange(task_config["tile_size"][0]/2, task_config["xrange_cor"][1]-task_config["xrange_cor"][0] ,task_config["tile_size"][0])
        y_cor = np.arange(task_config["tile_size"][1]/2, task_config["yrange_cor"][1]-task_config["yrange_cor"][0] ,task_config["tile_size"][1])

        x_stri = np.arange(task_config["tile_size"][0]/2, task_config["xrange_stri"][1]-task_config["xrange_stri"][0] ,task_config["tile_size"][0])
        y_stri = np.arange(task_config["tile_size"][1]/2, task_config["yrange_stri"][1]-task_config["yrange_stri"][0] ,task_config["tile_size"][1])

        x_vp = np.arange(task_config["tile_size"][0]/2, task_config["xrange_vp"][1]-task_config["xrange_vp"][0] ,task_config["tile_size"][0])
        y_vp = np.arange(task_config["tile_size"][1]/2, task_config["yrange_vp"][1]-task_config["yrange_vp"][0] ,task_config["tile_size"][1])

        x_zi = np.arange(task_config["tile_size"][0]/2, task_config["xrange_zi"][1]-task_config["xrange_zi"][0] ,task_config["tile_size"][0])
        y_zi = np.arange(task_config["tile_size"][1]/2, task_config["yrange_zi"][1]-task_config["yrange_zi"][0] ,task_config["tile_size"][1])

        if mode == "train":
            z_vals = task_config["z_train"]
        elif mode == "val":
            z_vals = task_config["z_val"]
        elif mode == "test":
            z_vals = task_config["z_test"]
            
        if download and exists(download_path+task_config['name']+mode+'images.npy'):
            image_array = np.load(download_path+task_config['name']+mode+'images.npy')
            mask_array = np.load(download_path+task_config['name']+mode+'labels.npy')
            self.image_array = image_array
            self.mask_array = mask_array
        else:
            self.config = boss_config

            reset_counter = 0
            
            while reset_counter<retries:
                try:
                    print('Downloading BossDB cutout...')
                    self.boss_image_array = array(task_config["image_chan"], boss_config=boss_config)
                    self.boss_mask_array = array(task_config["annotation_chan"], boss_config=boss_config)
                    #cortex
                    image_array =  self.boss_image_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_cor"][0] : task_config["yrange_cor"][1],
                            task_config["xrange_cor"][0] : task_config["xrange_cor"][1],
                        ]
                    mask_array =  self.boss_mask_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_cor"][0] : task_config["yrange_cor"][1],
                            task_config["xrange_cor"][0] : task_config["xrange_cor"][1],
                        ]
                    #striatum
                    image_array_temp =  self.boss_image_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_stri"][0] : task_config["yrange_stri"][1],
                            task_config["xrange_stri"][0] : task_config["xrange_stri"][1],
                        ]
                    mask_array_temp =  self.boss_mask_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_stri"][0] : task_config["yrange_stri"][1],
                            task_config["xrange_stri"][0] : task_config["xrange_stri"][1],
                        ]
                    image_array = np.concatenate((image_array,image_array_temp))
                    mask_array = np.concatenate((mask_array,mask_array_temp))
                    #vp
                    image_array_temp =  self.boss_image_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_vp"][0] : task_config["yrange_vp"][1],
                            task_config["xrange_vp"][0] : task_config["xrange_vp"][1],
                        ]
                    mask_array_temp =  self.boss_mask_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_vp"][0] : task_config["yrange_vp"][1],
                            task_config["xrange_vp"][0] : task_config["xrange_vp"][1],
                        ]
                    image_array = np.concatenate((image_array,image_array_temp))
                    mask_array = np.concatenate((mask_array,mask_array_temp))
                    #zi
                    image_array_temp =  self.boss_image_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_zi"][0] : task_config["yrange_zi"][1],
                            task_config["xrange_zi"][0] : task_config["xrange_zi"][1],
                        ]
                    mask_array_temp =  self.boss_mask_array[
                            z_vals[0] : z_vals[1],
                            task_config["yrange_zi"][0] : task_config["yrange_zi"][1],
                            task_config["xrange_zi"][0] : task_config["xrange_zi"][1],
                        ]
                    image_array = np.concatenate((image_array,image_array_temp))
                    mask_array = np.concatenate((mask_array,mask_array_temp))
                    self.image_array = image_array
                    self.mask_array = mask_array
                    if download:
                        np.save(download_path+task_config['name']+mode+'images.npy', image_array)
                        np.save(download_path+task_config['name']+mode+'labels.npy', mask_array)
                    break
                except HTTPError as e:
                    print('Error connecting to BossDB channels, retrying')
                    print(e)
                    reset_counter = reset_counter + 1

        #note- for X and Y, this is the centroid, for the z dimension the cutout is handled differently
        #z is the start of the volume, and the volume extends to z+task_config["volume_z"]
        centroids = []
        if mode == "train":
            #cortex
            z_vals = np.arange(0,task_config["z_train"][1]-task_config["z_train"][0] ,task_config["volume_z"])
            for z in z_vals:
                for x in x_cor:
                    for y in y_cor:
                        centroids.append([z, y, x])
            
            #striatum
            z_vals = np.arange((task_config["z_train"][1]-task_config["z_train"][0]),2*(task_config["z_train"][1]-task_config["z_train"][0]) ,task_config["volume_z"])
            for z in z_vals:
                for x in x_stri:
                    for y in y_stri:
                        centroids.append([z, y, x])

            #vp
            z_vals = np.arange(2*(task_config["z_train"][1]-task_config["z_train"][0]),3*(task_config["z_train"][1]-task_config["z_train"][0]) ,task_config["volume_z"])
            for z in z_vals:
                for x in x_vp:
                    for y in y_vp:
                        centroids.append([z, y, x])

            if 'noZI' in task_config and bool(task_config['noZI']):
                pass
            else:
                #zi
                z_vals = np.arange(3*(task_config["z_train"][1]-task_config["z_train"][0]),4*(task_config["z_train"][1]-task_config["z_train"][0]) ,task_config["volume_z"])
            
                for z in z_vals:
                    for x in x_zi:
                        for y in y_zi:
                            centroids.append([z, y, x])

        if mode == "val":
            z_vals = np.arange(0,task_config["z_val"][1]-task_config["z_val"][0] ,task_config["volume_z"])
            for z in z_vals:
                for x in x_cor:
                    for y in y_cor:
                        centroids.append([z, y, x])
            z_vals = np.arange((task_config["z_val"][1]-task_config["z_val"][0]),2*(task_config["z_val"][1]-task_config["z_val"][0]) ,task_config["volume_z"])
            for z in z_vals:
                for x in x_cor:
                    for y in y_cor:
                        centroids.append([z, y, x])
            z_vals = np.arange(2*(task_config["z_val"][1]-task_config["z_val"][0]),3*(task_config["z_val"][1]-task_config["z_val"][0]) ,task_config["volume_z"])
            for z in z_vals:
                for x in x_cor:
                    for y in y_cor:
                        centroids.append([z, y, x])
             
            if 'noZI' in task_config and bool(task_config['noZI']):
                pass
            else:
                z_vals = np.arange(3*(task_config["z_val"][1]-task_config["z_val"][0]),4*(task_config["z_val"][1]-task_config["z_val"][0]) ,task_config["volume_z"])
                for z in z_vals:
                    for x in x_cor:
                        for y in y_cor:
                            centroids.append([z, y, x])

        if mode == "test":
            z_vals = np.arange(0,task_config["z_test"][1]-task_config["z_test"][0] ,task_config["volume_z"])
            for z in z_vals:
                for x in x_cor:
                    for y in y_cor:
                        centroids.append([z, y, x])
            
            z_vals = np.arange((task_config["z_test"][1]-task_config["z_test"][0]),2*(task_config["z_test"][1]-task_config["z_test"][0]) ,task_config["volume_z"])         
            for z in z_vals:
                for x in x_cor:
                    for y in y_cor:
                        centroids.append([z, y, x])
            z_vals = np.arange(2*(task_config["z_test"][1]-task_config["z_test"][0]),3*(task_config["z_test"][1]-task_config["z_test"][0]) ,task_config["volume_z"])
            for z in z_vals:
                for x in x_cor:
                    for y in y_cor:
                        centroids.append([z, y, x])
            
            if 'noZI' in task_config and bool(task_config['noZI']):
                pass
            else:
                z_vals = np.arange(3*(task_config["z_test"][1]-task_config["z_test"][0]),4*(task_config["z_test"][1]-task_config["z_test"][0]) ,task_config["volume_z"])
                
                for z in z_vals:
                    for x in x_cor:
                        for y in y_cor:
                            centroids.append([z, y, x])

        self.centroid_list = centroids
        rad_y = int(task_config["tile_size"][1]/2)
        rad_x = int(task_config["tile_size"][0]/2)
        self.px_radius_y = rad_y
        self.px_radius_x = rad_x
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.z_size = task_config["volume_z"]
        if 'combine_ax_and_bg' in task_config and bool(task_config['combine_ax_and_bg']):
            self.combine_ax_and_bg = 1
        else:
            self.combine_ax_and_bg = 0
    
    def __getitem__(self, key):
        z, y, x = self.centroid_list[key]
        z = int(z)
        y = int(y)
        x = int(x)
        image_array =  self.image_array[
                z : z + self.z_size,
                y - self.px_radius_y : y + self.px_radius_y,
                x - self.px_radius_x : x + self.px_radius_x,
            ]
        mask_array =  self.mask_array[
                z : z + self.z_size,
                y - self.px_radius_y : y + self.px_radius_y,
                x - self.px_radius_x : x + self.px_radius_x,
            ]
        
        if self.image_transform:
            image_array = self.image_transform(image_array)
        if self.mask_transform:
            mask_array = self.mask_transform(mask_array.astype('int64'))
            mask_array = torch.squeeze(mask_array)
        if self.z_size>1:
            image_array = torch.permute(image_array,(1,0,2))
            image_array = torch.unsqueeze(image_array,0)
            mask_array = torch.permute(mask_array,(1,0,2))
        else:
            image_array = torch.permute(image_array,(1,2,0))
            mask_array = torch.permute(torch.squeeze(mask_array),(1,0))
        
        if self.combine_ax_and_bg:
            threeclass_mask_array = np.where(mask_array==3, 0, mask_array)
            return image_array, threeclass_mask_array
        else:
            return image_array, mask_array

    def __len__(self):
        return len(self.centroid_list)
