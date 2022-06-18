#!/usr/bin/env python

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils import data
import pathlib 
from trainer import Trainer
from torchvision import transforms
import json as json
from bossdbdataset import BossDBDataset
from datetime import datetime
import argparse
import os
from tqdm import tqdm 
from self_supervised.trainer import BYOLTrainer, MYOWTrainer, MYOWTrainerMerged
from self_supervised.data import prepare_views
from models import resnet_xray, combine_model
from self_supervised.transforms import xray
from self_supervised.utils import console
from sklearn.metrics import confusion_matrix

def train_model(task_config,network_config,boss_config=None,gpu='cuda'):
    torch.manual_seed(network_config['seed'])
    np.random.seed(network_config['seed'])
    if network_config['augmentations'] == 0:
        transform = transforms.Compose([transforms.ToTensor(),
                                ])
    if network_config['augmentations'] == 1:
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation((0,180)),
                                    ])

    train_data = BossDBDataset(
        task_config, boss_config, "train", image_transform = transform, mask_transform = transform)

    val_data =  BossDBDataset(
        task_config, boss_config, "val", image_transform = transform, mask_transform = transform)

    test_data =  BossDBDataset(
        task_config, boss_config, "test", image_transform = transform, mask_transform = transform)

    training_dataloader = data.DataLoader(dataset=train_data,
                                        batch_size=network_config['batch_size'],
                                        shuffle=True)

    validation_dataloader = data.DataLoader(dataset=val_data,
                                        batch_size=network_config['batch_size'],
                                        shuffle=True)
    test_dataloader = data.DataLoader(dataset=test_data,
                                        batch_size=network_config['batch_size'],
                                        shuffle=False)

    training_dataloader_m = data.DataLoader(dataset=train_data,
                                        batch_size=network_config['batch_size'],
                                        shuffle=True)

    # device
    if torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device = torch.device('cpu')

    # models
    if network_config["model"] == "ResNet-18":
        print('loading ResNet-18 model')
        encoder = resnet_xray(network_config["encoder_name"]).to(device)
    representation_size = encoder.representation_size
    
    if network_config["model"] == "ResNet-50":
        print('loading ResNet-50 model')
        model = None


    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    train_transform = xray.get_xray_transform(network_config["transform"], name='xray', p=network_config["trans_prob"], dataset_type='2D', image_size=128)
    unnormalize = xray.get_xray_unnormalize('xray')
    train_transform_m = xray.get_xray_transform(network_config["transform"], name='xray', p=network_config["trans_prob"], dataset_type='2D', image_size=128)
    
    if network_config["method"] == 'byol':
        trainer = BYOLTrainer(encoder=encoder,
                            representation_size=representation_size,
                            projector_output_size=network_config["projector_output_size"],
                            projector_hidden_size=network_config["projector_hidden_size"],
                            symmetric_loss=True,
                            train_dataloader=training_dataloader,
                            transform=train_transform,
                            prepare_views=prepare_views,
                            total_epochs=network_config["ssl_epochs"],
                            batch_size=network_config["batch_size"],
                            lr_warmup_epochs=network_config["lr_warmup_epochs"],
                            base_lr=network_config["base_learning_rate"],
                            base_momentum=network_config["base_momentum"],
                            lr_decay=network_config["lr_decay"],
                            lr_poly_decay_n=network_config["lr_poly_decay_n"],
                            lr_milestones=network_config["lr_milestone"],
                            lr_gamma=network_config["lr_gamma"],
                            use_lars_rule=True,
                            mm_decay=network_config["mm_decay"],
                            optimizer_type=network_config["ssl_optimizer"],
                            optimizer_momentum=network_config["opt_momentum"],
                            weight_decay=network_config["weight_decay"],
                            exclude_bias_and_bn=False,
                            gpu=0,
                            log_steps=100,
                            logdir=network_config["outputdir"],
                            unnormalize=unnormalize,
                            )
                            
    elif network_config["method"] == 'myow':
        trainer = MYOWTrainer(encoder=encoder,
                            representation_size=representation_size,
                            projector_output_size=network_config["projector_output_size"],
                            projector_hidden_size=network_config["projector_hidden_size"],
                            different_init=False,
                            layout=network_config["myow_layout"],
                            projector_2_output_size=network_config["projector_2_output_size"],
                            projector_2_hidden_size=network_config["projector_2_hidden_size"],
                            symmetric_loss=True,
                            train_dataloader=training_dataloader,
                            transform=train_transform,
                            view_pool_dataloader=training_dataloader_m,
                            transform_m=train_transform_m,
                            prepare_views=prepare_views,
                            total_epochs=network_config["ssl_epochs"],
                            batch_size=network_config["batch_size"],
                            lr_warmup_epochs=network_config["lr_warmup_epochs"],
                            base_lr=network_config["base_learning_rate"],
                            base_momentum=network_config["base_momentum"],
                            use_lars_rule=True,
                            lr_decay=network_config["lr_decay"],
                            lr_poly_decay_n=network_config["lr_poly_decay_n"],
                            lr_milestones=network_config["lr_milestone"],
                            lr_gamma=network_config["lr_gamma"],
                            mm_decay=network_config["mm_decay"],
                            byol_warmup_epochs=network_config["byol_warmup_epochs"],
                            myow_rampup_epochs=network_config["myow_rampup_epochs"],
                            base_myow_weight=network_config["myow_base_weight"],
                            view_miner=network_config["miner"],
                            view_miner_candidate_repr=network_config["miner_candidate_repr"],
                            view_miner_distance=network_config["miner_distance"],
                            select_neigh=network_config["knn_select"],
                            knn_nneighs=network_config["knn_nneighs"],
                            optimizer_type=network_config["ssl_optimizer"],
                            optimizer_momentum=network_config["opt_momentum"],
                            weight_decay=network_config["weight_decay"],
                            exclude_bias_and_bn=False,
                            gpu=0,
                            master_gpu=0,
                            log_steps=100,
                            logdir=network_config["outputdir"],
                            unnormalize=unnormalize,
                            )
    
    elif network_config["method"] == 'myowmerged':
    
        trainer = MYOWTrainerMerged(encoder=encoder,
                            representation_size=representation_size,
                            projector_output_size=network_config["projector_output_size"],
                            projector_hidden_size=network_config["projector_hidden_size"],
                            different_init=False,
                            layout=network_config["myow_layout"],
                            projector_2_output_size=network_config["projector_2_output_size"],
                            projector_2_hidden_size=network_config["projector_2_hidden_size"],
                            symmetric_loss=True,
                            train_dataloader=training_dataloader,
                            transform=train_transform,
                            view_pool_dataloader=training_dataloader_m,
                            transform_m=train_transform_m,
                            prepare_views=prepare_views,
                            total_epochs=network_config["ssl_epochs"],
                            batch_size=network_config["batch_size"],
                            lr_warmup_epochs=network_config["lr_warmup_epochs"],
                            base_lr=network_config["base_learning_rate"],
                            base_momentum=network_config["base_momentum"],
                            use_lars_rule=True,
                            lr_decay=network_config["lr_decay"],
                            lr_poly_decay_n=network_config["lr_poly_decay_n"],
                            lr_milestones=network_config["lr_milestone"],
                            lr_gamma=network_config["lr_gamma"],
                            mm_decay=network_config["mm_decay"],
                            byol_warmup_epochs=network_config["byol_warmup_epochs"],
                            myow_rampup_epochs=network_config["myow_rampup_epochs"],
                            base_myow_weight=network_config["myow_base_weight"],
                            view_miner=network_config["miner"],
                            view_miner_candidate_repr=network_config["miner_candidate_repr"],
                            view_miner_distance=network_config["miner_distance"],
                            select_neigh=network_config["knn_select"],
                            knn_nneighs=network_config["knn_nneighs"],
                            optimizer_type=network_config["ssl_optimizer"],
                            optimizer_momentum=network_config["opt_momentum"],
                            weight_decay=network_config["weight_decay"],
                            exclude_bias_and_bn=False,
                            gpu=0,
                            master_gpu=0,
                            log_steps=100,
                            logdir=network_config["outputdir"],
                            unnormalize=unnormalize,
                            )

    with console.progress as progress:
        for epoch in progress.track(range(trainer.epoch + 1, network_config["ssl_epochs"] + 1)):
            trainer.train_epoch()

    trainer.cleanup()

    lin_classifier = encoder.get_linear_classifier(network_config["classes"]).to(device)
    encoder.eval()
    model = combine_model(encoder, lin_classifier)
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    if network_config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.classifier.parameters(), lr=network_config["learning_rate"])
    if network_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=network_config["learning_rate"], betas=(network_config["beta1"],network_config["beta2"]))
    
    trainer = Trainer(model=model,
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    training_DataLoader=training_dataloader,
                    validation_DataLoader=validation_dataloader,
                    lr_scheduler=None,
                    epochs=network_config["epochs"],
                    epoch=0,
                    notebook=False)
    training_losses, validation_losses, lr_rates = trainer.run_trainer()
    # save the model
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    model_name =  network_config['outweightfilename'] + '_' + network_config['task_type'] + '_' + date + '.pt'
    os.makedirs(pathlib.Path.cwd() / network_config['outputdir'], exist_ok = True) 
    torch.save(model.state_dict(), pathlib.Path.cwd() / network_config['outputdir'] / model_name)

    #take loss curves
    plt.figure()
    plt.plot(training_losses,label='training')
    plt.plot(validation_losses,label='validation')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Learning Curve')
    plt.legend()
    model_name = model_name[:len(model_name)-3] + '_learning_curve.png'
    plt.savefig(pathlib.Path.cwd() / network_config['outputdir'] / model_name)

    def predict(img,
                model,
                device,
                ):
        model.eval()
        x = img.to(device)  # to torch, send to device
        with torch.no_grad():
            out = model(x)  # send through model/network

        out_argmax = torch.argmax(out, dim=1)  # perform softmax on outputs
        return out_argmax

    # Test
    batch_iter = tqdm(enumerate(test_dataloader), 'test', total=len(test_dataloader), leave=False)
    num_classes = network_config['classes']
    correct = 0
    total = 0
    confusion = torch.zeros(num_classes,num_classes)

    for i, (x, y) in batch_iter:
        label = y.to(device) #can do this on CPU
        with torch.no_grad():
            pred_class = predict(x, model, device)
            correct += pred_class.eq(label.view_as(pred_class)).sum().item()
            total += len(label)
            confusion += confusion_matrix(label.cpu(), pred_class.cpu(), labels=[0,1,2,3])
    acc = correct/total
    confusion = confusion.detach().cpu().numpy()
    print('Accuracy: {}'.format(acc))
    print('Confusion Matrix: ')
    print(confusion)


if __name__ == '__main__':
    # usage python3 task1_ssl.py --task taskconfig/task1.json --network networkconfig/SSL_BYOL_ResNet-18_2D.json --boss boss_config.json
    parser = argparse.ArgumentParser(description='flags for training')
    parser.add_argument('--task', default="taskconfig/task1.json",
                        help='task config json file')
    parser.add_argument('--network', default="networkconfig/SSL_BYOL_ResNet-18_2D.json",
                        help='network config json file')
    parser.add_argument('--boss', 
                        help='boss config json file')
    parser.add_argument('--gpu', 
                        help='index of the gpu to use')
    args = parser.parse_args()
    
    if args.gpu:
        gpu = 'cuda:'+args.gpu
    else:
        gpu = 'cuda'

    task_config = json.load(open(args.task))
    network_config = json.load(open(args.network))
    if args.boss:
        boss_config = json.load(open(args.boss))
    else:
        boss_config = None
    print('begining training')
    train_model(task_config,network_config,boss_config,gpu)
