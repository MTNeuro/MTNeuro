#!/usr/bin/env python

import torch
from typing import Tuple, List
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils import data
import pathlib 
# files copied from https://github.com/johschmidt42/PyTorch-2D-3D-UNet-Tutorial
from trainer import Trainer
from torchvision import transforms as pt_transforms
import json as json
from bossdbdataset import BossDBDataset
# from unet import UNet
from datetime import datetime
import argparse
import os
import ssl
from tqdm import tqdm 
import segmentation_models_pytorch as smp
from torchsummary import summary
from self_supervised.trainer import BYOLTrainer, MYOWTrainer, MYOWTrainerMerged
from self_supervised.data import prepare_views
from models import resnet_xray, combine_model
from self_supervised import transforms
from self_supervised.utils import set_random_seeds, console
#This was necessary to overcome an SSL cert error when downloading pretrained weights for SMP- your milelage may vary here
ssl._create_default_https_context = ssl._create_unverified_context

def train_model(task_config,network_config,boss_config=None,gpu='cuda'):
    torch.manual_seed(network_config['seed'])
    np.random.seed(network_config['seed'])
    if network_config['augmentations'] == 0:
        transform = pt_transforms.Compose([pt_transforms.ToTensor(),
                                ])
    if network_config['augmentations'] == 1:
        transform = pt_transforms.Compose([pt_transforms.ToTensor(),
                                    pt_transforms.RandomVerticalFlip(p=0.5),
                                    pt_transforms.RandomHorizontalFlip(p=0.5),
                                    pt_transforms.RandomRotation((0,180)),
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

    x, y = next(iter(training_dataloader))

    #print(f'x = shape: {x.shape}; type: {x.dtype}')
    #print(f'x = min: {x.min()}; max: {x.max()}')
    #print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')

    # device
    if torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device = torch.device('cpu')

    # models
    if network_config["model"] == "ResNet-18":
        print('loading UNet model')
        encoder = resnet_xray(network_config["encoder_name"]).to(device)
    representation_size = encoder.representation_size
    if network_config["model"] == "smp_UnetPlusPlus":
        print('loading UnetPlusPlus model')
        model = smp.UnetPlusPlus(
            encoder_name=network_config["encoder_name"],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=network_config["encoder_weights"],     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=network_config["in_channels"],                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=network_config["classes"],                      # model output channels (number of classes in your dataset)
        ).to(device)


    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    # if network_config["optimizer"] == "SGD":
    #     optimizer = torch.optim.SGD(model.parameters(), lr=network_config["learning_rate"])
    # if network_config["optimizer"] == "Adam":
    #     optimizer = torch.optim.Adam(model.parameters(), lr=network_config["learning_rate"], betas=(network_config["beta1"],network_config["beta2"]))
    optimizer = 'sgd'

    train_transform = transforms.xray.get_xray_transform(network_config["transform"], name='xray', p=network_config["trans_prob"], dataset_type='2D', image_size=128)
    unnormalize = transforms.xray.get_xray_unnormalize('xray')
    train_transform_m = transforms.xray.get_xray_transform(network_config["transform"], name='xray', p=network_config["trans_prob"], dataset_type='2D', image_size=128)
    
    if network_config["method"] == 'byol':

        trainer = BYOLTrainer(encoder=encoder,
                            representation_size=representation_size,
                            projector_output_size=network_config["projector_output_size"],
                            projector_hidden_size=network_config["projector_hidden_size"],
                            # symmetric_loss=symmetric_loss,
                            train_dataloader=training_dataloader,
                            transform=train_transform,
                            prepare_views=prepare_views,
                            total_epochs=network_config["epochs"],
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
                            optimizer_type=optimizer,
                            optimizer_momentum=network_config["opt_momentum"],
                            weight_decay=network_config["weight_decay"],
                            exclude_bias_and_bn=False,
                            # distributed=distributed,
                            # world_size=world_size,
                            # rank=gpu,
                            gpu=0,
                            # master_gpu=0,
                            # port=port,
                            log_steps=100,
                            logdir=network_config["outputdir"],
                            # log_img=log_img,
                            # log_img_steps=log_img_steps,
                            unnormalize=unnormalize,
                            # resume_ckpt=initial_checkpoint,
                            )
                            

    elif network_config["method"] == 'myow':
    
        trainer = MYOWTrainer(encoder=encoder,
                            representation_size=representation_size,
                            projector_output_size=network_config["projector_output_size"],
                            projector_hidden_size=network_config["projector_hidden_size"],
                            different_init=False,
                            # symmetric_loss=symmetric_loss,
                            layout=myow_layout,
                            projector_2_output_size=network_config["projector_2_output_size"],
                            projector_2_hidden_size=network_config["projector_2_hidden_size"],
                            train_dataloader=training_dataloader,
                            transform=train_transform,
                            view_pool_dataloader=training_dataloader_m,
                            transform_m=train_transform_m,
                            prepare_views=prepare_views,
                            total_epochs=network_config["epochs"],
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
                            view_miner=miner,
                            view_miner_candidate_repr=miner_candidate_repr,
                            view_miner_distance=miner_distance,
                            select_neigh=knn_select,
                            knn_nneighs=knn_nneighs,
                            optimizer_type=optimizer,
                            optimizer_momentum=network_config["opt_momentum"],
                            weight_decay=network_config["weight_decay"],
                            exclude_bias_and_bn=network_config["exclude_bias_and_bn"],
                            distributed=distributed,
                            world_size=world_size,
                            rank=gpu,
                            gpu=gpu,
                            master_gpu=0,
                            port=port,
                            log_steps=log_steps,
                            logdir=network_config["logdir"],
                            # log_img=log_img,
                            # log_img_steps=log_img_steps,
                            unnormalize=unnormalize,
                            # resume_ckpt=initial_checkpoint,
                            # convert_byol_to_myow=convert_byol_to_myow,
                            )
    
    elif network_config["method"] == 'myowmerged':
    
        trainer = MYOWTrainerMerged(encoder=encoder,
                            representation_size=representation_size,
                            projector_output_size=network_config["projector_output_size"],
                            projector_hidden_size=network_config["projector_hidden_size"],
                            different_init=False,
                            # symmetric_loss=symmetric_loss,
                            layout=myow_layout,
                            projector_2_output_size=network_config["projector_2_output_size"],
                            projector_2_hidden_size=network_config["projector_2_hidden_size"],
                            train_dataloader=training_dataloader,
                            transform=train_transform,
                            view_pool_dataloader=training_dataloader_m,
                            transform_m=train_transform_m,
                            prepare_views=prepare_views,
                            total_epochs=network_config["epochs"],
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
                            view_miner=miner,
                            view_miner_candidate_repr=miner_candidate_repr,
                            view_miner_distance=miner_distance,
                            select_neigh=knn_select,
                            knn_nneighs=knn_nneighs,
                            optimizer_type=optimizer,
                            optimizer_momentum=network_config["opt_momentum"],
                            weight_decay=network_config["weight_decay"],
                            exclude_bias_and_bn=network_config["exclude_bias_and_bn"],
                            distributed=distributed,
                            world_size=world_size,
                            rank=gpu,
                            gpu=gpu,
                            master_gpu=0,
                            port=port,
                            log_steps=log_steps,
                            logdir=network_config["logdir"],
                            log_img=log_img,
                            log_img_steps=log_img_steps,
                            unnormalize=unnormalize,
                            resume_ckpt=initial_checkpoint,
                            convert_byol_to_myow=convert_byol_to_myow,
                            )
    # trainer (I changed the epochs to 5 just to make it run faster)
    # trainer = Trainer(model=model,
    #                 device=device,
    #                 criterion=criterion,
    #                 optimizer=optimizer,
    #                 training_DataLoader=training_dataloader,
    #                 validation_DataLoader=validation_dataloader,
    #                 lr_scheduler=None,
    #                 epochs=network_config["epochs"],
    #                 epoch=0,
    #                 notebook=False)

    # start training
    # training_losses, validation_losses, lr_rates = trainer.run_trainer()
    with console.progress as progress:
        for epoch in progress.track(range(trainer.epoch + 1, network_config["epochs"] + 1)):
            # train for one epoch
            trainer.train_epoch()
            # if epoch % FLAGS.ckpt_epochs == 0:
            #     trainer.save_checkpoint(epoch)

    trainer.cleanup()

    lin_classifier = encoder.get_linear_classifier(network_config["classes"]).to(device)
    encoder.eval()
    model = combine_model(encoder, lin_classifier)
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=network_config["learning_rate"], momentum=1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=eval_lr_milestone, gamma=eval_lr_gamma)
    criterion = torch.nn.CrossEntropyLoss()
    
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

    batch_iter = tqdm(enumerate(test_dataloader), 'test', total=len(test_dataloader), leave=False)
    # predict the segmentations of test set
    tp_tot = torch.empty(0,network_config['classes'])
    fp_tot = torch.empty(0,network_config['classes'])
    fn_tot = torch.empty(0,network_config['classes'])
    tn_tot = torch.empty(0,network_config['classes'])

    # first compute statistics for true positives, false positives, false negative and
    # true negative "pixels"
    for i, (x, y) in batch_iter:
        #input, target = x.to(device), y.to(device)  # send to device (GPU or CPU)
        target = y.to(device) #can do this on CPU

        with torch.no_grad():
            # get the output image
            output = predict(x, model, device)
            tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='multiclass', num_classes = network_config['classes'])
            tp_tot = torch.vstack((tp_tot,tp))
            fp_tot = torch.vstack((fp_tot,fp))
            fn_tot = torch.vstack((fn_tot,fn))
            tn_tot = torch.vstack((tn_tot,tn))


    # then compute metrics with required reduction (see metric docs)
    model_name = model_name[:len(model_name)-19] + '_report.rpt'
    rh = open(pathlib.Path.cwd() / network_config['outputdir'] / model_name, 'w')
 
    #Accuracy Per Class
    accuracy = smp.metrics.accuracy(tp_tot, fp_tot, fn_tot, tn_tot, reduction='none')
    per_class = torch.mean(accuracy,dim=0)
    print('old Accuracy per Class:')
    print(np.array(per_class))
    rh.write('old Accuracy per Class:\n')
    rh.write(str(np.array(per_class))+'\n')


    #BAL accuracy (average across non background classes)
    bal_accuracy = torch.mean(torch.mean(accuracy[:,1:network_config['classes']],dim=0))
    print(f'old Balanced accuracy (No background): {bal_accuracy}')
    rh.write(f'old Balanced accuracy (No background): {bal_accuracy}\n')

    #F1 and IoU
    if network_config['eval_reduction']=='None':
        #F1 score
        f1_score = smp.metrics.f1_score(tp_tot, fp_tot, fn_tot, tn_tot, reduction=None)
        f1_score = f1_score.sum(0)/len(f1_score)
        print(f'old F1-score: {np.array(f1_score)} Avg. F1-score: {f1_score.mean()}')
        rh.write(f'old F1-score: {np.array(f1_score)} Avg. F1-score: {f1_score.mean()}\n')
        iou_score = smp.metrics.iou_score(tp_tot, fp_tot, fn_tot, tn_tot, reduction=None)
        iou_score = iou_score.sum(0)/len(iou_score)
        print(f'old IoU: {np.array(iou_score)} Avg. IoU-score: {iou_score.mean()}')
        rh.write(f'old IoU: {np.array(iou_score)} Avg. IoU-score: {iou_score.mean()}')
    else:
        #F1 score
        f1_score = smp.metrics.f1_score(tp_tot, fp_tot, fn_tot, tn_tot, reduction=network_config['eval_reduction'])
        print(f'old F1-score: {f1_score}')
        rh.write(f'old F1-score: {f1_score}\n')

        iou_score = smp.metrics.iou_score(tp_tot, fp_tot, fn_tot, tn_tot, reduction=network_config['eval_reduction'])
        print(f'old IoU: {iou_score}') 
        rh.write(f'old IoU: {iou_score}')

    print('\n\n')
    rh.write('\n\n')

    acc = (tp_tot.mean(dim=0)+tn_tot.mean(dim=0))/(fp_tot.mean(dim=0)+tn_tot.mean(dim=0)+fn_tot.mean(dim=0)+tp_tot.mean(dim=0))
    print('new Accuracy per Class:')
    print(np.array(acc.cpu()))
    rh.write('new Accuracy per Class:\n')
    rh.write(str(np.array(acc.cpu())))
    
    spec =  (tn_tot[:,1:].mean())/(fp_tot[:,1:].mean()+tn_tot[:,1:].mean())
    sens =  (tp_tot[:,1:].mean())/(fn_tot[:,1:].mean()+tp_tot[:,1:].mean())
    balacc = (spec + sens)/2
    print(f'new Balanced accuracy (No background): {balacc}')
    rh.write(f'new Balanced accuracy (No background): {balacc}\n')
    
    prec = tp_tot.mean(dim=0)/(fp_tot.mean(dim=0)+tp_tot.mean(dim=0))
    reca = tp_tot.mean(dim=0)/(fn_tot.mean(dim=0)+tp_tot.mean(dim=0))
    f1 = (2*reca*prec)/(reca+prec)
    print(f'new F1-score: {np.array(f1.cpu())} Avg. F1-score: {f1.mean()}')
    rh.write(f'new F1-score: {np.array(f1.cpu())} Avg. F1-score: {f1.mean()}\n')

    iou = (tp_tot.mean(0))/(fp_tot.mean(0)+fn_tot.mean(0)+tp_tot.mean(0))
    print(f'new IoU: {np.array(iou_score.cpu())} Avg. IoU-score: {iou_score.mean()}')
    rh.write(f'new IoU: {np.array(iou_score.cpu())} Avg. IoU-score: {iou_score.mean()}\n')

    rh.close()


if __name__ == '__main__':
    # usage python3 task2_2D_smp_main.py --task task2.json --network network_config_smp.json --boss boss_config.json
    parser = argparse.ArgumentParser(description='flags for training')
    parser.add_argument('--task', default="taskconfig/task1.json",
                        help='task config json file')
    parser.add_argument('--network', default="networkconfig/ResNet18_2D_SSL.json",
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
