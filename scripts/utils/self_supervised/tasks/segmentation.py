import torch
import torch.nn as nn
from einops import rearrange, repeat

from tqdm import tqdm


class linear_UNet(nn.Module):
    """an as-linear-as-possible small unet for validation
    one Linear layer self.rep_upscale is used to upscale the representation
    one Linear layer self.predict is used to predict the label"""
    def __init__(self, rep_size, n_classes=4, dim=8):
        super().__init__()
        self.dim = dim
        self.output_dim = n_classes

        self.rep_upscale = nn.Linear(rep_size, dim*64*64)
        self.predict = nn.Linear(dim+1, n_classes)

    def forward(self, rep, x):
        '''output shape [bs, 4, 64, 64]'''
        rep_up = self.rep_upscale(rep)
        rep_up = torch.reshape(rep_up, shape=(rep.shape[0], self.dim, 64,64))
        cat = torch.cat([x, rep_up], dim=1)
        cat = torch.transpose(cat, 1, 3)
        output = self.predict(cat)
        output = torch.squeeze(torch.transpose(output, 1, 3))
        return output


def pixel_acc(encoder, segmenter, loader, transform=None, device='cpu'):
    """compute the pixel level accuracy based on label and segmenter output"""
    encoder.eval()
    segmenter.eval()

    acc_ttl = []  # pixel acc mean value in each batch
    bs_ttl = []  # batch size of each batch

    with torch.no_grad():
        for x, label in loader:
            x, label = x.to(device), label.to(device)
            
            if transform is not None:
                x, label = transform(x, label = label)

            representation = encoder(x)
            output = segmenter(representation, x)

            # compute pixel acc
            preds = torch.argmax(output, 1)
            acc = preds.eq(label).float().mean()

            # prevent the error in the averaging procedure
            batch_size = x.shape[0]

            acc_ttl.append(acc)
            bs_ttl.append(batch_size)

    acc_value = sum([acc_ttl[i] * bs_ttl[i] for i in range(len(acc_ttl))]) / sum(bs_ttl)
    return acc_value
    

def last_pixel_acc(encoder, segmenter, loader, transform=None, device='cpu'):
    """compute the pixel level accuracy based on label and segmenter output"""
    encoder.eval()
    segmenter.eval()

    class_acc = {0:[],1:[],2:[],3:[]}
    acc_ttl = []  # pixel acc mean value in each batch
    bs_ttl = []  # batch size of each batch

    with torch.no_grad():
        for x, label in loader:
            x, label = x.to(device), label.to(device)
            
            if transform is not None:
                x, label = transform(x, label = label)

            representation = encoder(x)
            output = segmenter(representation, x)

            # compute pixel acc
            preds = torch.argmax(output, 1)
            acc = preds.eq(label).float().mean()
            for c in range(4):
                class_acc[c].append(per_class_score(preds, label, c))

            # prevent the error in the averaging procedure
            batch_size = x.shape[0]

            acc_ttl.append(acc)
            bs_ttl.append(batch_size)

    acc_value = sum([acc_ttl[i] * bs_ttl[i] for i in range(len(acc_ttl))]) / sum(bs_ttl)
    class_acc_value = {}
    for c in range(4):
        class_acc_value[c] = sum([class_acc[c][i] * bs_ttl[i] for i in range(len(bs_ttl))]) / sum(bs_ttl)
    return acc_value, class_acc_value
    

def per_class_score(preds, label, c):
    p = torch.zeros(preds.shape).to(torch.device('cuda:0'))
    l = torch.zeros(label.shape).to(torch.device('cuda:0'))
    p[preds == c] = 1 
    l[label==c] = 1
    acc = torch.count_nonzero(l*p)/torch.count_nonzero(l)
    return acc
    
def downsample_background_f(pred_logits, label, p):
    mask = label == 0
    full_indices = mask.nonzero()
    perm = torch.randperm(full_indices.size(0))
    idx = perm[:int(p*full_indices.size(0))]
    sampled_indices = full_indices[idx]
    pred_logits[:,:,sampled_indices[:,0], sampled_indices[:,1]] = 0
    pred_logits[:,0,sampled_indices[:,0], sampled_indices[:,1]] = 1
    return pred_logits, label
    
def downsample_background(pred_logits, label, p):
    label = rearrange(label, 'b h w -> (b h w)')
    pred_logits = rearrange(pred_logits, 'b c h w -> (b h w) c')
    mask0 = (label == 0)
    mask1 = (label != 0)
    label0 = label[mask0]
    label1 = label[mask1]
    pred_logits0 = pred_logits[mask0]
    pred_logits1 = pred_logits[mask1]
    perm = torch.randperm(label0.size(0))
    idx = perm[:int(p*label0.size(0))]
    pred_logits = torch.cat([pred_logits0[idx], pred_logits1])
    label = torch.cat([label0[idx], label1])
    return pred_logits, label
    

def train_segmenter(encoder,
                    segmenter,
                    train_loader,
                    test_loader,
                    optimizer,
                    num_epochs,
                    train_transform=None,
                    test_transform=None,
                    device='cpu',
                    writer=None,
                    tag='',
                    tqdm_progress=False):
    """this function performs a segmentation evaluation based on original images"""
    encoder.eval()
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(num_epochs), disable=not tqdm_progress):
        segmenter.train()
        for x, label in train_loader:
            """x should be original image, and label should be the segmentation label
            segmentation label should shape like [batch size, total classes, image height, image weight]"""
            # load data
            x, label = x.to(device), label.to(device)
            if train_transform is not None:
                x, label = train_transform(x, label = label)

            # forward
            optimizer.zero_grad()
            with torch.no_grad():
                representation = encoder(x).detach()
            pred_logits = segmenter(representation, x)  # x was also fed in for skip connection like a Unet
            # print(pred_logits.shape) # torch.Size([8, 4, 64, 64])

            pred_logits = rearrange(pred_logits, 'b c h w -> (b h w) c')
            label = rearrange(label, 'b h w -> (b h w)')

            unmask = label != 5
            pred_logits = pred_logits[unmask]
            label = label[unmask]

            preds = torch.argmax(pred_logits, dim=1)
            print(preds[preds == 2].shape)

            loss = criterion(pred_logits, label)
            loss.backward()
            optimizer.step()
            #print(loss)

        # compute segmentation accuracy
        acc = pixel_acc(encoder, segmenter, test_loader, transform=test_transform, device=device)

        if writer is not None:
            writer.add_scalar('eval_seg/test-%r' % tag, acc, epoch)
    final_acc, class_acc = last_pixel_acc(encoder, segmenter, test_loader, transform=test_transform, device=device)

    return final_acc, class_acc
