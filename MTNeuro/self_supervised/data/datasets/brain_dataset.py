import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from PIL import ImageFilter
import cv2

from einops import rearrange



class BrainRegionDatasetInfer(Dataset):
        
    def __init__(
        self, 
        cutout_data, 
        labels=0,
        train = False,
        edge_prob = 0,
        edge_strength = 1,
        tresize=64
    ):  
        array = cutout_data/255
        if labels == 0:
            self.labels = np.ones(len(array))
        else:
            self.labels = labels
        self.tresize = tresize
        self.resize = transforms.Resize((self.tresize,self.tresize),5)
        self.classes = ['cortex', 'striatum', 'ventral posterior nucleus', 'zona incerta']

        self.train = train
        self.edge_prob = edge_prob
        if edge_prob:        
            self.edge_strength = edge_strength
            self.kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])/9
        self.totensor = transforms.ToTensor()
        self.num_samples = array.shape[0] if self.train else array.shape[0]*4
        self.resized = np.zeros(( self.tresize,self.tresize, array.shape[0]))
        for i in range(array.shape[0]):
            img = array[i]
            if self.edge_prob and (torch.rand(1) <= self.edge_prob):
                img = cv2.filter2D(img, -1, self.kernel)
            self.resized[:,:,i] = self.resize(Image.fromarray(img))
        
    def __getitem__(self, key):
        if self.train:
            img = self.resized[:,:,key]
                
            label = self.labels[key]
            return self.totensor(np.array(img)).float(), label
        else:
            slice_id, corner_id = divmod(key, 4)
            img = self.resized[ :,:, slice_id]
            label = self.labels[slice_id]
            resized = np.array(img)
            csize = int(self.tresize/2)
            if corner_id==0:
                crop = resized[:csize, :csize]
            elif corner_id==1:
                crop = resized[:csize, -csize:]
            elif corner_id==2:
                crop = resized[-csize:, :csize]
            else:
                crop = resized[-csize:, -csize:]
            return self.totensor(crop).float(), label
                
    def __len__(self):
        return self.num_samples






class BrainRegionDataset(Dataset):
    
    def __init__(
        self, 
        data_path: str, 
        labels_path: str,
        train = True,
        edge_prob = 0,
        edge_strength = 1,
        tresize=64
    ):
        array = np.load(data_path)/255
        self.labels = np.load(labels_path)
        self.tresize = tresize
        self.resize = transforms.Resize((self.tresize,self.tresize),5)
        self.classes = ['cortex', 'striatum', 'ventral posterior nucleus', 'zona incerta']

        self.train = train
        self.edge_prob = edge_prob
        if edge_prob:    
            self.edge_strength = edge_strength
            self.kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])/9
        self.totensor = transforms.ToTensor()
        self.num_samples = array.shape[0] if self.train else array.shape[0]*4
        self.resized = np.zeros(( self.tresize,self.tresize, array.shape[0]))
        for i in range(array.shape[0]):
            img = array[i]
            if self.edge_prob and (torch.rand(1) <= self.edge_prob):
                img = cv2.filter2D(img, -1, self.kernel)
            self.resized[:,:,i] = self.resize(Image.fromarray(img))
    
    def __getitem__(self, key):
        if self.train:
            img = self.resized[:,:,key]
            
            label = self.labels[key]
            return self.totensor(np.array(img)).float(), label
        else:
            slice_id, corner_id = divmod(key, 4)
            img = self.resized[ :,:, slice_id]
            label = self.labels[slice_id]
            resized = np.array(img)
            csize = int(self.tresize/2)
            if corner_id==0:
            	crop = resized[:csize, :csize]
            elif corner_id==1:
            	crop = resized[:csize, -csize:]
            elif corner_id==2:
            	crop = resized[-csize:, :csize]
            else:
            	crop = resized[-csize:, -csize:]
            return self.totensor(crop).float(), label
            
    def __len__(self):
        return self.num_samples
        
class BrainRegionDataset3D(Dataset):
    
    def __init__(
        self, 
        data_path: str, 
        labels_path: str,
        train = True,
        tresize=64
    ):
        array = np.load(data_path)/255
        self.labels = np.load(labels_path)
        self.tresize = tresize
        self.resize = transforms.Resize((self.tresize,self.tresize),5) 
        self.classes = ['cortex', 'striatum', 'ventral posterior nucleus', 'zona incerta']

        self.train = train
        self.totensor = transforms.ToTensor()
        self.depth = 28 if self.train else 10
        
        self.resized = np.zeros(( self.tresize,self.tresize, array.shape[0]))
        for i in range(array.shape[0]):
            self.resized[:,:,i] = self.resize(Image.fromarray(array[i]))
        elms_per_class =  int(array.shape[0]/4)
        self.idxs = np.arange(0,elms_per_class-self.depth,  dtype=int)
        for i in range(1,4):
            self.idxs = np.concatenate([self.idxs, np.arange(0,elms_per_class-self.depth,  dtype=int) + i*elms_per_class])
        
        self.num_samples = (elms_per_class-self.depth)*4 if self.train else (elms_per_class-self.depth)*16
        
    def __getitem__(self, key):
        if self.train:
            idx = self.idxs[key]
            resized = self.resized[:,:,idx:(idx+self.depth)]
            label = self.labels[idx]
            t_resized = self.totensor(np.array(resized))
            return t_resized.float(), label
        else:
            slice_id, corner_id = divmod(key, 4)
            idx = self.idxs[slice_id]
            img = self.resized[:,:,idx:(idx+self.depth)]
            label = self.labels[idx]
            csize = int(self.tresize/2)
            
            if corner_id==0:
            	crop = img[:csize, :csize,:]
            elif corner_id==1:
                crop = img[:csize, -csize:, :]
            elif corner_id==2:
                crop = img[-csize:, :csize,:]
            else:
                crop = img[-csize:, -csize:,:]
            return self.totensor(np.array(crop))[ None,...].float(), label
            
    def __len__(self):
        return self.num_samples

class BrainSegDataset(Dataset):

    regions = ['cortex', 'striatum', 'trn', 'zi']
    label_type = ['_annotation_data', '_raw_data']
    # only limited slices have segmentation data available
    train_slices = [30, 60, 90, 120, 150, 180, 210, 240, 270]
    test_slices = [300, 330]
    # train_slices = 300  # first 300 slices for training
    # test_slices = 360  # next 60 slices for testing

    def __init__(
        self,
        data_path = "./data/all_microstructures.npz",
        train = True,
        edge_prob = 0
    ):

        self.data = np.load(data_path)
        self.train = train
        self.totensor = transforms.ToTensor()
        
        self.rescale = 255.
        self.resize = transforms.Resize((64,64), 5)
        self.anno_resize = transforms.Resize((64, 64), interpolation=0)
        self.edge_prob = edge_prob
        self.kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])/9

        self._translate_data()

    def _translate_data(self):
        self.raw_data = []
        self.anno_data = []

        for region in self.regions:
            if self.train:
                reg_data = self.data["{}{}".format(region, self.label_type[1])][self.train_slices, :256, :256]
                anno_data = self.data["{}{}".format(region, self.label_type[0])][self.train_slices, :256, :256]
            else:
                reg_data = self.data["{}{}".format(region, self.label_type[1])][self.test_slices, :256, :256]
                anno_data = self.data["{}{}".format(region, self.label_type[0])][self.test_slices, :256, :256]
            self.raw_data.append(reg_data)
            self.anno_data.append(anno_data)

        self.raw_data = np.concatenate(self.raw_data)  # should be 1200 * 256 *256
        self.anno_data = np.concatenate(self.anno_data)
        #msk0 = self.anno_data == 0
        #msk3 = self.anno_data == 3
        #self.anno_data[msk0]=3
        #self.anno_data[msk3]=0        
        
    def _choose_dense_cut(self, img_raw, img_anno):
        crop = []
        crop.append([img_anno[:32, :32], img_raw[:, :32, :32]])
        crop.append([img_anno[:32, -32:], img_raw[:, :32, -32:]])
        crop.append([img_anno[-32:, :32], img_raw[:, -32:, :32]])
        crop.append([img_anno[-32:, -32:], img_raw[:, -32:, -32:]])
        
        max_nnz = 0
        for a,c in crop:
            nnz = np.count_nonzero(a)
            if nnz > max_nnz:
                max_nnz=nnz
                best_anno = a
                best_img = c
        return best_img, best_anno 
            
    

    def __getitem__(self, key):
        """
        self.raw_data is what will be fed inside model, so the dim is [batch, 1, 64, 64]
        append a new dimension in the segmentation network will make it work
        self.anno_data is the segmentation label (from 0 to 3), the dim was [batch, 4, 256, 256]
        due to the resize, dim was
        """
        if self.edge_prob:
            img_resize = self.totensor(np.array(self.resize(Image.fromarray(cv2.filter2D(self.raw_data[key], -1, self.kernel)))))/self.rescale
        else:
            img_resize = self.totensor(np.array(self.resize(Image.fromarray(self.raw_data[key])))) / self.rescale
        img = self.totensor(np.array((Image.fromarray(self.raw_data[key])))) / self.rescale

        anno_resize = self.totensor(np.array(self.anno_resize(Image.fromarray(self.anno_data[key].copy()))))
        anno_data = torch.Tensor(self.anno_data.copy())

        # for some reason, it resizes the label this way. Hmm.
        # tensor([0.0000, 0.0039, 0.0078, 0.0118])
        anno_resize = torch.squeeze(integer_seg(anno_resize), dim=0)

        # check the resizing effect via the debug mode
        DEBUG = False
        if DEBUG:
            return img_resize, img, anno_data[key], anno_resize
        else:
            #img_resize, anno_resize = self._choose_dense_cut(img_resize, anno_resize)
            return img_resize, anno_resize.long()

    def __len__(self):
        return self.raw_data.shape[0]

class Data_balancing(Dataset):
    """logic: add label [value == 5] as masks"""
    def __init__(self, train_loader):
        self.data = []
        self.label = []
        for data, label in train_loader:
            self.data.append(data)
            self.label.append(label)

        self.data = torch.cat(self.data, dim=0)  # torch.Size([8, 1, 64, 64])
        self.label = torch.cat(self.label, dim=0)  # torch.Size([8, 64, 64])
        self.generate_mask()

    def generate_mask(self):
        flatten_label = rearrange(self.label, 'b h w -> (b h w)')

        min_label = 1e7
        for cls_i in range(4):
            label_i = flatten_label[flatten_label == cls_i]
            if label_i.shape[0] <= min_label:
                min_label = label_i.shape[0]

        for i in range(4):
            mask_i = self.label == i
            mask_i_1, mask_i_2, mask_i_3 = torch.nonzero(mask_i, as_tuple=True)  # each shape as b
            # print(mask_i_1.shape)  # torch.Size([4830, 3])

            self.label[mask_i_1[min_label:], mask_i_2[min_label:], mask_i_3[min_label:]] = 5

    def __getitem__(self, index):
        return self.data[index, :, :, :], self.label[index, :, :]
    def __len__(self):
        return self.data.shape[0]


NN_labels = [0.0000, 0.0039, 0.0078, 0.0118]
def vectorize_seg(data, labels=NN_labels):
    """data is the [1, 64, 64] resize procedure output"""
    data_rp = []
    for l in range(len(labels)):
        data_empty = data.clone()
        data_empty = torch.where(torch.logical_and((data_empty < labels[l]+ 1e-4), (data_empty > labels[l] - 1e-4)), 1, 0)
        data_rp.append(data_empty)

    data = torch.cat(data_rp)
    return data

def integer_seg(data, labels=NN_labels):
    data_return = torch.zeros(data.shape)
    for l in range(len(labels)):
        data_i = torch.where(torch.logical_and((data < labels[l]+ 1e-4), (data > labels[l] - 1e-4)), l, 0)
        data_return = data_return + data_i

    return data_return

    def __len__(self):
        return self.num_samples
