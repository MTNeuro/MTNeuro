#!/usr/bin/env python

import torch
from typing import Tuple, List
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import numpy as np
from torch.utils import data
import pathlib 
from trainer import Trainer
from torchvision import transforms
import json as json
from bossdbdataset import BossDBDataset
from unet import UNet
from datetime import datetime
import argparse
import os
import ssl
from tqdm import tqdm 
import segmentation_models_pytorch as smp
from torchsummary import summary

#This was necessary to overcome an SSL cert error when downloading pretrained weights for SMP baselines- your milelage may vary here
ssl._create_default_https_context = ssl._create_unverified_context

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


    x, y = next(iter(training_dataloader))

    # device
    if torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device = torch.device('cpu')

    # models
    if network_config["model"] == "UNet":
        print('loading UNet model')
        model = UNet(in_channels=network_config['in_channels'],
                out_channels=network_config['classes'],
                n_blocks=network_config['n_blocks'],
                start_filters=network_config['start_filters'],
                activation=network_config['activation'],
                normalization=network_config['normalization'],
                conv_mode=network_config['conv_mode'],
                dim=network_config['dim']).to(device)

    if network_config["model"] == "smp_UnetPlusPlus":
        print('loading UnetPlusPlus model')
        model = smp.UnetPlusPlus(
            encoder_name=network_config["encoder_name"],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=network_config["encoder_weights"],     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=network_config["in_channels"],                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=network_config["classes"],                      # model output channels (number of classes in your dataset)
        ).to(device)

    if network_config["model"] == "smp_MAnet":
        print('loading MAnet model')
        model = smp.MAnet(
            encoder_name=network_config["encoder_name"],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=network_config["encoder_weights"],     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=network_config["in_channels"],                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=network_config["classes"],                      # model output channels (number of classes in your dataset)
        ).to(device)

    if network_config["model"] == "smp_PAN":
        print('loading PAN model')
        model = smp.PAN(
            encoder_name=network_config["encoder_name"],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=network_config["encoder_weights"],     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=network_config["in_channels"],                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=network_config["classes"],                      # model output channels (number of classes in your dataset)
        ).to(device)

    if network_config["model"] == "smp_Linknet":
        print('loading Linknet model')
        model = smp.Linknet(
            encoder_name=network_config["encoder_name"],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=network_config["encoder_weights"],     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=network_config["in_channels"],                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=network_config["classes"],                      # model output channels (number of classes in your dataset)
        ).to(device)

    if network_config["model"] == "smp_FPN":
        print('loading FPN model')
        model = smp.FPN(
            encoder_name=network_config["encoder_name"],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=network_config["encoder_weights"],     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=network_config["in_channels"],                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=network_config["classes"],                      # model output channels (number of classes in your dataset)
        ).to(device)

    if network_config["model"] == "smp_PSPNet":
        print('loading PSPNet model')
        model = smp.PSPNet(
            encoder_name=network_config["encoder_name"],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=network_config["encoder_weights"],     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=network_config["in_channels"],                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=network_config["classes"],                      # model output channels (number of classes in your dataset)
        ).to(device)

    if network_config["model"] == "smp_DeepLabV3Plus":
        print('loading DeepLabV3Plus model')
        model = smp.DeepLabV3Plus(
            encoder_name=network_config["encoder_name"],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=network_config["encoder_weights"],     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=network_config["in_channels"],                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=network_config["classes"],                      # model output channels (number of classes in your dataset)
        ).to(device)

    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    if network_config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=network_config["learning_rate"])
    if network_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=network_config["learning_rate"], betas=(network_config["beta1"],network_config["beta2"]))
    
    # trainer (I changed the epochs to 5 just to make it run faster)
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

    # start training
    training_losses, validation_losses, lr_rates = trainer.run_trainer()

    # save the model
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    model_name =  network_config['outweightfilename'] + '_' + task_config['task_type'] + '_' + date + '.pt'
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
    parser.add_argument('--task', default="task_configs/task2.json",
                        help='task config json file')
    parser.add_argument('--network', default="network_configs/UNet_2D.json",
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
