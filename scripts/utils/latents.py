from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *
import numpy as np
from self_supervised.data.datasets import BrainRegionDatasetInfer
from self_supervised.trainer import BYOLTrainer, MYOWTrainer
from self_supervised.models import resnet_xray, resnet_xray_classifier
from self_supervised.utils import set_random_seeds, console
from self_supervised.data import prepare_views
from self_supervised import transforms
from sklearn.decomposition import PCA, NMF



def get_latents(cut_out_data,encoder_file_path,ssl=1):
'''getting latents for provided encoder data
ssl flag for ssl model. ssl = 0 for supervised model'''
    test_file = cut_out_data
    dataset_test = BrainRegionDatasetInfer(cut_out_data, train = False, edge_prob = 0)
    test_transform = transforms.xray.get_xray_transform([], 'xray')
    test_loader = DataLoader(dataset_test, batch_size=2048, num_workers=4, shuffle=False,
                                 drop_last=False, pin_memory=True)


    encoder = resnet_xray('resnet18').to(torch.device('cuda:0'))
    if ssl == 1:
        ckpt_epoch = BYOLTrainer.load_trained_encoder(encoder, ckpt_path= encoder_file_path, device=torch.device('cuda:0'))
    encoder.eval()
    test_targets = []
    test_embeddings = torch.zeros((0, encoder.representation_size), dtype=torch.float32)
    #Latent space
    with torch.no_grad():
        for x, _  in test_loader:
            x = x.to(torch.device('cuda:0'))
            representation = encoder(x)
            test_embeddings = torch.cat((test_embeddings, representation.detach().cpu()), 0)

    test_embeddings = np.array(test_embeddings)



    return test_embeddings
    
def get_unsup_latents(cut_out_data,pca = 1):
'''get latents for unsupervised methods. PCA = 1 for PCA, PCA = 0 for NMF'''
    dataset_test = BrainRegionDatasetInfer( cut_out_data,  train = False, edge_prob = 0, tresize=128)
    test_transform = transforms.xray.get_xray_transform([], 'xray')
    test_loader = DataLoader(dataset_test , batch_size=4*2*2048, num_workers=4, shuffle=False,  drop_last=False, pin_memory=True)
    for x, _ in train_loader:
        print(x.shape)
        train_x = torch.reshape(x, (11520,64*64)).numpy()
    if pca == 1:
    	print('fitting PCA')
        pca = PCA(n_components=256)
        pca.fit(train_x)
        test_embeddings = pca.transform(test_x)
    elif pca == 0:
        print('Fitting NMF')
        nmf = NMF(n_components=256, init='random',verbose=True)
        nmf.fit(train_x)
        test_embeddings = nmf.transform(test_x)
    return test_embeddings
