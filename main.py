'''Train CIFAR10/CIFAR100 with PyTorch.'''
import argparse
import os
from optimizers import (KFACOptimizer, SKFACOptimizer, EKFACOptimizer, KBFGSOptimizer, KBFGSLOptimizer, KBFGSL2LOOPOptimizer, KBFGSLMEOptimizer, NGDOptimizer, NGDStreamOptimizer)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.network_utils import get_network
from utils.data_utils import get_dataloader
from torchsummary import summary

from backpack import backpack, extend
from backpack.extensions import FisherBlock, FisherBlockEff, BatchGrad
from backpack.utils.conv import unfold_func
import math
import time
import copy
import matplotlib.pylab as plt

from torch import einsum, matmul, eye
from torch.linalg import inv
import numpy as np
# for REPRODUCIBILITY
# torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# np.random.seed(0)
# import random
# random.seed(0)


# fetch args
parser = argparse.ArgumentParser()

parser.add_argument('--network', default='vgg16_bn', type=str)
parser.add_argument('--depth', default=19, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)

# densenet
parser.add_argument('--growthRate', default=12, type=int)
parser.add_argument('--compressionRate', default=2, type=int)

# wrn, densenet
parser.add_argument('--widen_factor', default=1, type=int)
parser.add_argument('--dropRate', default=0.0, type=float)
parser.add_argument('--base_width', default=24, type=int)
parser.add_argument('--cardinality', default=32, type=int)


parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--log_dir', default='runs/pretrain', type=str)
parser.add_argument('--save_inv', default='false', type=str)


parser.add_argument('--optimizer', default='kfac', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--milestone', default=None, type=str)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--learning_rate_decay', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--stat_decay', default=0.95, type=float)
parser.add_argument('--damping', default=1e-3, type=float)
parser.add_argument('--kl_clip', default=1e-2, type=float)
parser.add_argument('--weight_decay', default=3e-3, type=float)
parser.add_argument('--TCov', default=20, type=int)
parser.add_argument('--TScal', default=20, type=int)
parser.add_argument('--TInv', default=100, type=int)

# for ngd optimizer
parser.add_argument('--freq', default=100, type=int)
parser.add_argument('--low_rank', default='false', type=str)
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--batchnorm', default='false', type=str)
parser.add_argument('--step_info', default='false', type=str)
parser.add_argument('--memory_efficient', default='false', type=str)
parser.add_argument('--trial', default='true', type=str)
parser.add_argument('--super_opt', default='false', type=str)
parser.add_argument('--reduce_sum', default='false', type=str)
parser.add_argument('--diag', default='false', type=str)
parser.add_argument('--partial_backprop', default='false', type=str)

# for adam optimizer
parser.add_argument('--epsilon', default=1e-8, type=float)

# for K-BFGS(L) optimizer
parser.add_argument('--num_s_y_pairs', default=1000, type=int)

parser.add_argument('--prefix', default=None, type=str)
parser.add_argument('--debug_mem', default='false', type=str)

# for SKFAC optimizer
parser.add_argument('--subsample', default='false', type=str)
parser.add_argument('--num_ss_patches', default=0, type=int)

args = parser.parse_args()

# init model
nc = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist':10,
    'fashion-mnist': 10
}
num_classes = nc[args.dataset]
net = get_network(args.network,
                  depth=args.depth,
                  num_classes=num_classes,
                  growthRate=args.growthRate,
                  compressionRate=args.compressionRate,
                  widen_factor=args.widen_factor,
                  dropRate=args.dropRate,
                  base_width=args.base_width,
                  cardinality=args.cardinality)



net = net.to(args.device)

module_names = ''
if hasattr(net, 'features'): 
    module_names = 'features'
elif hasattr(net, 'children'):
    module_names = 'children'
else:
    print('unknown net modules...')





if args.dataset == 'mnist':
    summary(net, ( 1, 28, 28))
elif args.dataset == 'cifar10':
    summary(net, ( 3, 32, 32))
elif args.dataset == 'cifar100':
    summary(net, ( 3, 32, 32))
elif args.dataset == 'fashion-mnist':
    summary(net, ( 1, 28, 28))

# init dataloader
trainloader, testloader = get_dataloader(dataset=args.dataset,
                                         train_batch_size=args.batch_size,
                                         test_batch_size=256)

# init optimizer and lr scheduler
optim_name = args.optimizer.lower()
tag = optim_name
if optim_name == 'sgd':
    optimizer = optim.SGD(net.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
elif optim_name == 'kfac':
    optimizer = KFACOptimizer(net,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              stat_decay=args.stat_decay,
                              damping=args.damping,
                              kl_clip=args.kl_clip,
                              weight_decay=args.weight_decay,
                              TCov=args.TCov,
                              TInv=args.TInv)
    if args.save_inv == 'true':
      os.mkdir('kfac')

elif optim_name == 'skfac':
    optimizer = SKFACOptimizer(net,
                               lr=args.learning_rate,
                               momentum=args.momentum,
                               stat_decay=args.stat_decay,
                               damping=args.damping,
                               kl_clip=args.kl_clip,
                               weight_decay=args.weight_decay,
                               TCov=args.TCov,
                               TInv=args.TInv,
                               subsample=args.subsample,
                               num_ss_patches=args.num_ss_patches)

elif optim_name == 'ekfac':
    optimizer = EKFACOptimizer(net,
                               lr=args.learning_rate,
                               momentum=args.momentum,
                               stat_decay=args.stat_decay,
                               damping=args.damping,
                               kl_clip=args.kl_clip,
                               weight_decay=args.weight_decay,
                               TCov=args.TCov,
                               TScal=args.TScal,
                               TInv=args.TInv)
elif optim_name == 'ngd':
    print('NGD optimizer selected.')
    # optimizer = optim.SGD(net.parameters(),
    #                       lr=args.learning_rate,
    #                       momentum=args.momentum,
    #                       weight_decay=args.weight_decay)
    optimizer = optim.SGD(net.parameters(),
                          lr=args.learning_rate)
    buf = {}
    if args.momentum != 0:
        for name, param in net.named_parameters():
                # print('initializing momentum buffer')
                buf[name] = torch.zeros_like(param.data).to(args.device) 
    if args.save_inv == 'true':
      os.mkdir('ngd')

elif optim_name == 'exact_ngd':
    print('Exact NGD optimizer selected.')
    optimizer = optim.SGD(net.parameters(),
                          lr=args.learning_rate)
    if args.save_inv == 'true':
      os.mkdir('exact')

elif optim_name == 'kngd':
    print('Test optimizer selected')
    optimizer = NGDOptimizer(net,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              damping=args.damping,
                              kl_clip=args.kl_clip,
                              weight_decay=args.weight_decay,
                              freq=args.freq,
                              gamma=args.gamma,
                              low_rank=args.low_rank,
                              super_opt=args.super_opt,
                              reduce_sum=args.reduce_sum,
                              diag=args.diag)

elif optim_name == 'ngd_stream':
    # SAEED: TODO fix batchnorm or remove it totally
    print('NGD Stream Optimizer selected')
    optimizer = NGDStreamOptimizer(net,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              damping=args.damping,
                              kl_clip=args.kl_clip,
                              weight_decay=args.weight_decay,
                              freq=args.freq,
                              gamma=args.gamma,
                              low_rank=args.low_rank,
                              super_opt=args.super_opt,
                              reduce_sum=args.reduce_sum,
                              diag=args.diag)

  # perform a forward pass to get the dimensions
    net.eval()
    sample_input, sample_classe = next(iter(trainloader)) 
    sample_output = net(sample_input)  
    #### TODO: chane to 'cuda'
    if args.device == 'cuda':
      optimizer.initialize()
    net.train()
    # print('*'*1000)
    # print(sample_output)

elif optim_name == 'kbfgs':
    print('K-BFGS optimizer selected.')
    optimizer = KBFGSOptimizer(net,
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay,
                               stat_decay=args.stat_decay,
                               damping=args.damping,
                               TCov=args.TCov,
                               TInv=args.TInv)
elif optim_name == 'kbfgsl':
    print('K-BFGS(L) optimizer selected.')
    optimizer = KBFGSLOptimizer(net,
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay,
                                stat_decay=args.stat_decay,
                                damping=args.damping,
                                TCov=args.TCov,
                                TInv=args.TInv,
                                num_s_y_pairs=args.num_s_y_pairs)
elif optim_name == 'kbfgsl_2loop':
    print('K-BFGS(L) two-loop optimizer selected.')
    optimizer = KBFGSL2LOOPOptimizer(net,
                                     lr=args.learning_rate,
                                     momentum=args.momentum,
                                     weight_decay=args.weight_decay,
                                     stat_decay=args.stat_decay,
                                     damping=args.damping,
                                     TCov=args.TCov,
                                     TInv=args.TInv,
                                     num_s_y_pairs=args.num_s_y_pairs)
elif optim_name == 'kbfgsl_mem_eff':
    print('K-BFGS(L) memory efficient optimizer selected.')
    optimizer = KBFGSLMEOptimizer(net,
                                  lr=args.learning_rate,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay,
                                  stat_decay=args.stat_decay,
                                  damping=args.damping,
                                  TCov=args.TCov,
                                  TInv=args.TInv,
                                  num_s_y_pairs=args.num_s_y_pairs)
elif optim_name == 'adam':
    print('Adam optimizer selected.')
    optimizer = optim.Adam(net.parameters(),
                          lr=args.learning_rate,
                          weight_decay=args.weight_decay,
                          eps=args.epsilon)

else:
    raise NotImplementedError

if args.milestone is None:
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(args.epoch*0.5), int(args.epoch*0.75)], gamma=args.learning_rate_decay)
else:
    milestone = [int(_) for _ in args.milestone.split(',')]
    lr_scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=args.learning_rate_decay)

# init criterion
criterion = nn.CrossEntropyLoss()
criterion_none = nn.CrossEntropyLoss(reduction='none')

if optim_name == 'ngd':
    extend(net)
    extend(criterion)
    extend(criterion_none)
    # print(net.state_dict())
elif optim_name == 'exact_ngd':
    extend(net)
    extend(criterion)


# parameters for damping update
# damping = args.damping





damping = args.damping
start_epoch = 0
best_acc = 0
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.load_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.load_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%' % (start_epoch, best_acc))

# init summary writter

log_dir = os.path.join(args.log_dir, args.dataset, args.network, args.optimizer,
                       'lr%.3f_wd%.4f_damping%.4f' %
                       (args.learning_rate, args.weight_decay, args.damping))
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

TRAIN_INFO  = {}
TRAIN_INFO['train_loss'] = []
TRAIN_INFO['test_loss'] = []
TRAIN_INFO['train_acc'] = []
TRAIN_INFO['test_acc'] = []
TRAIN_INFO['total_time'] = []
TRAIN_INFO['epoch_time'] = []

if args.debug_mem == 'true':
  TRAIN_INFO['memory'] = []
  
def store_io_(Flag=True):
    all_modules = net.modules()
    for m in all_modules:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            m.training = Flag


def train(epoch):
    torch.set_printoptions(precision=16)
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    step_st_time = time.time()
    epoch_time = 0
    print('\nKFAC/KBFGS damping: %f' % damping)
    print('\nNGD damping: %f' % (damping))

    # 
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (tag, lr_scheduler.get_last_lr()[0], 0, 0, correct, total))

    writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:

        if optim_name in ['kfac', 'skfac', 'ekfac', 'sgd', 'adam']:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if optim_name in ['kfac', 'skfac', 'ekfac'] and optimizer.steps % optimizer.TCov == 0:
                # compute true fisher
                optimizer.acc_stats = True
                with torch.no_grad():
                    sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),1).squeeze().to(args.device)
                loss_sample = criterion(outputs, sampled_y)
                loss_sample.backward(retain_graph=True)
                optimizer.acc_stats = False
                optimizer.zero_grad()  # clear the gradient for computing true-fisher.
            loss.backward()
            optimizer.step()
        elif optim_name in ['kbfgs', 'kbfgsl', 'kbfgsl_2loop', 'kbfgsl_mem_eff']:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # do another forward-backward pass over batch inside step()
            def closure():
                return inputs, targets, criterion, False # is_autoencoder = False
            optimizer.step(closure)
        elif optim_name == 'exact_ngd':
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # update Fisher inverse
            if batch_idx % args.freq == 0:
              # compute true fisher
              with torch.no_grad():
                sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),1).squeeze().to(args.device)
              # use backpack extension to compute individual gradient in a batch
              batch_grad = []
              with backpack(BatchGrad()):
                loss_sample = criterion(outputs, sampled_y)
                loss_sample.backward(retain_graph=True)

              for name, param in net.named_parameters():
                if hasattr(param, "grad_batch"):
                  batch_grad.append(args.batch_size * param.grad_batch.reshape(args.batch_size, -1))
                else:
                  raise NotImplementedError

              J = torch.cat(batch_grad, 1)
              fisher = torch.matmul(J.t(), J) / args.batch_size
              inv = torch.linalg.inv(fisher + damping * torch.eye(fisher.size(0)).to(fisher.device))
              # clean the gradient to compute the true fisher
              optimizer.zero_grad()

            loss.backward()
            # compute the step direction p = F^-1 @ g
            grad_list = []
            for name, param in net.named_parameters():
              grad_list.append(param.grad.data.reshape(-1, 1))
            g = torch.cat(grad_list, 0)
            p = torch.matmul(inv, g)

            start = 0
            for name, param in net.named_parameters():
              end = start + param.data.reshape(-1, 1).size(0)
              param.grad.copy_(p[start:end].reshape(param.grad.data.shape))
              start = end

            optimizer.step()

        ### new optimizer test
        elif optim_name in ['kngd', 'ngd_stream'] :
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if  optimizer.steps % optimizer.freq == 0:
                # compute true fisher
                optimizer.acc_stats = True
                with torch.no_grad():
                    sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs, dim=1),1).squeeze().to(args.device)
                loss_sample = criterion(outputs, sampled_y)
                loss_sample.backward(retain_graph=True)
                optimizer.acc_stats = False
                optimizer.zero_grad()  # clear the gradient for computing true-fisher.
                if args.partial_backprop == 'true':
                  idx = (sampled_y == targets) == False
                  loss = criterion(outputs[idx,:], targets[idx])
                  # print('extra:', idx.sum().item())
            loss.backward()
            optimizer.step()

        elif optim_name == 'ngd':
            if batch_idx % args.freq == 0:
                store_io_(True)
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                optimizer.zero_grad()
                # net.set_require_grad(True)

                outputs = net(inputs)
                damp = damping
                loss = criterion(outputs, targets)
                loss.backward(retain_graph=True)

                # storing original gradient for later use
                grad_org = []
                # grad_dict = {}
                for name, param in net.named_parameters():
                    grad_org.append(param.grad.reshape(1, -1))
                #     grad_dict[name] = param.grad.clone()
                grad_org = torch.cat(grad_org, 1)

                ###### now we have to compute the true fisher
                with torch.no_grad():
                # gg = torch.nn.functional.softmax(outputs, dim=1)
                    sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs, dim=1),1).squeeze().to(args.device)
                
                if args.trial == 'true':
                    update_list, loss = optimal_JJT_v2(outputs, sampled_y, args.batch_size, damping=damp, alpha=0.95, low_rank=args.low_rank, gamma=args.gamma, memory_efficient=args.memory_efficient, super_opt=args.super_opt)
                else:
                    update_list, loss = optimal_JJT(outputs, sampled_y, args.batch_size, damping=damp, alpha=0.95, low_rank=args.low_rank, gamma=args.gamma, memory_efficient=args.memory_efficient)

                # optimizer.zero_grad()
                # update_list, loss = optimal_JJT_fused(outputs, sampled_y, args.batch_size, damping=damp)

                optimizer.zero_grad()
   
                # last part of SMW formula
                grad_new = []
                for name, param in net.named_parameters():
                    param.grad.copy_(update_list[name])
                    grad_new.append(param.grad.reshape(1, -1))
                grad_new = torch.cat(grad_new, 1)   
                # grad_new = grad_org
                store_io_(False)
            else:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                optimizer.zero_grad()
                # net.set_require_grad(True)

                outputs = net(inputs)
                damp = damping
                loss = criterion(outputs, targets)
                loss.backward()

                # storing original gradient for later use
                grad_org = []
                # grad_dict = {}
                for name, param in net.named_parameters():
                    grad_org.append(param.grad.reshape(1, -1))
                #     grad_dict[name] = param.grad.clone()
                grad_org = torch.cat(grad_org, 1)

                ###### now we have to compute the true fisher
                # with torch.no_grad():
                # gg = torch.nn.functional.softmax(outputs, dim=1)
                    # sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs, dim=1),1).squeeze().to(args.device)
                all_modules = net.modules()

                for m in net.modules():
                    if hasattr(m, "NGD_inv"):                    
                        grad = m.weight.grad
                        if isinstance(m, nn.Linear):
                            I = m.I
                            G = m.G
                            n = I.shape[0]
                            NGD_inv = m.NGD_inv
                            grad_prod = einsum("ni,oi->no", (I, grad))
                            grad_prod = einsum("no,no->n", (grad_prod, G))
                            v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                            gv = einsum("n,no->no", (v, G))
                            gv = einsum("no,ni->oi", (gv, I))
                            gv = gv / n
                            update = (grad - gv)/damp
                            m.weight.grad.copy_(update)
                        elif isinstance(m, nn.Conv2d):
                            if hasattr(m, "AX"):

                                if args.low_rank.lower() == 'true':
                                    ###### using low rank structure
                                    U = m.U
                                    S = m.S
                                    V = m.V
                                    NGD_inv = m.NGD_inv
                                    n = NGD_inv.shape[0]

                                    grad_reshape = grad.reshape(grad.shape[0], -1)
                                    grad_prod = V @ grad_reshape.t().reshape(-1, 1)
                                    grad_prod = torch.diag(S) @ grad_prod
                                    grad_prod = U @ grad_prod
                                    
                                    grad_prod = grad_prod.squeeze()
                                    v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                                    gv = U.t() @ v.unsqueeze(1)
                                    gv = torch.diag(S) @ gv
                                    gv = V.t() @ gv

                                    gv = gv.reshape(grad_reshape.shape[1], grad_reshape.shape[0]).t()
                                    gv = gv.view_as(grad)
                                    gv = gv / n
                                    update = (grad - gv)/damp
                                    m.weight.grad.copy_(update)
                                else:
                                    AX = m.AX
                                    NGD_inv = m.NGD_inv
                                    n = AX.shape[0]

                                    grad_reshape = grad.reshape(grad.shape[0], -1)
                                    grad_prod = einsum("nkm,mk->n", (AX, grad_reshape))
                                    v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                                    gv = einsum("nkm,n->mk", (AX, v))
                                    gv = gv.view_as(grad)
                                    gv = gv / n
                                    update = (grad - gv)/damp
                                    m.weight.grad.copy_(update)
                            elif hasattr(m, "I"):
                                I = m.I
                                if args.memory_efficient == 'true':
                                    I = unfold_func(m)(I)
                                G = m.G
                                n = I.shape[0]
                                NGD_inv = m.NGD_inv
                                grad_reshape = grad.reshape(grad.shape[0], -1)
                                x1 = einsum("nkl,mk->nml", (I, grad_reshape))
                                grad_prod = einsum("nml,nml->n", (x1, G))
                                v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                                gv = einsum("n,nml->nml", (v, G))
                                gv = einsum("nml,nkl->mk", (gv, I))
                                gv = gv.view_as(grad)
                                gv = gv / n
                                update = (grad - gv)/damp
                                m.weight.grad.copy_(update)
                        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                            if args.batchnorm == 'true':
                                dw = m.dw
                                n = dw.shape[0]
                                NGD_inv = m.NGD_inv
                                grad_prod = einsum("ni,i->n", (dw, grad))

                                v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                                gv = einsum("n,ni->i", (v, dw))
                                
                                gv = gv / n
                                update = (grad - gv)/damp
                                m.weight.grad.copy_(update)
                        
                        

                # last part of SMW formula
                grad_new = []
                for name, param in net.named_parameters():
                    grad_new.append(param.grad.reshape(1, -1))
                grad_new = torch.cat(grad_new, 1)   
                # grad_new = grad_org


            ##### do kl clip
            lr = lr_scheduler.get_last_lr()[0]
            # vg_sum = 0
            # vg_sum += (grad_new * grad_org ).sum()
            # vg_sum = vg_sum * (lr ** 2)
            # nu = min(1.0, math.sqrt(args.kl_clip / vg_sum))
            # for name, param in net.named_parameters():
            #     param.grad.mul_(nu)

            # optimizer.step()
            # manual optimizing:
            with torch.no_grad():
                for name, param in net.named_parameters():
                    d_p = param.grad.data
                    # print('=== step ===')

                    # apply momentum
                    # if args.momentum != 0:
                    #     buf[name].mul_(args.momentum).add_(d_p)
                    #     d_p.copy_(buf[name])

                    # apply weight decay
                    if args.weight_decay != 0:
                        d_p.add_(args.weight_decay, param.data)

                    lr = lr_scheduler.get_last_lr()[0]
                    param.data.add_(-lr, d_p)
                    # print('d_p:', d_p.shape)
                    # print(d_p)



        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (tag, lr_scheduler.get_last_lr()[0], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)
        if args.step_info == 'true' and (batch_idx % 50 == 0 or batch_idx == len(prog_bar) - 1):
            step_saved_time = time.time() - step_st_time
            epoch_time += step_saved_time
            test_acc, test_loss = test(epoch)
            TRAIN_INFO['train_acc'].append(float("{:.4f}".format(100. * correct / total)))
            TRAIN_INFO['test_acc'].append(float("{:.4f}".format(test_acc)))
            TRAIN_INFO['train_loss'].append(float("{:.4f}".format(train_loss/(batch_idx + 1))))
            TRAIN_INFO['test_loss'].append(float("{:.4f}".format(test_loss)))
            TRAIN_INFO['total_time'].append(float("{:.4f}".format(step_saved_time)))
            if args.debug_mem == 'true':
                TRAIN_INFO['memory'].append(torch.cuda.memory_reserved())
            step_st_time = time.time()
            net.train()

    writer.add_scalar('train/loss', train_loss/(batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)
    acc = 100. * correct / total
    train_loss = train_loss/(batch_idx + 1)
    if args.step_info == 'true':
        TRAIN_INFO['epoch_time'].append(float("{:.4f}".format(epoch_time)))
    # save diagonal blocks of exact Fisher inverse or its approximations
    if args.save_inv == 'true':
      all_modules = net.modules()

      count = 0
      start, end = 0, 0
      if optim_name == 'ngd':
        for m in all_modules:
          if m.__class__.__name__ == 'Linear':
            with torch.no_grad():
              I = m.I
              G = m.G
              J = torch.einsum('ni,no->nio', I, G)
              J = J.reshape(J.size(0), -1)
              JTDJ = torch.matmul(J.t(), torch.matmul(m.NGD_inv, J)) / args.batch_size

              with open('ngd/' + str(epoch) + '_m_' + str(count) + '_inv.npy', 'wb') as f:
                np.save(f, ((torch.eye(JTDJ.size(0)).to(JTDJ.device) - JTDJ) / damping).cpu().numpy())
                count += 1

          elif m.__class__.__name__ == 'Conv2d':
            with torch.no_grad():
              AX = m.AX
              AX = AX.reshape(AX.size(0), -1)
              JTDJ = torch.matmul(AX.t(), torch.matmul(m.NGD_inv, AX)) / args.batch_size
              with open('ngd/' + str(epoch) + '_m_' + str(count) + '_inv.npy', 'wb') as f:
                np.save(f, ((torch.eye(JTDJ.size(0)).to(JTDJ.device) - JTDJ) / damping).cpu().numpy())
                count += 1

      elif optim_name == 'exact_ngd':
        for m in all_modules:
          if m.__class__.__name__ in ['Conv2d', 'Linear']:
            with open('exact/' + str(epoch) + '_m_' + str(count) + '_inv.npy', 'wb') as f:
              end = start + m.weight.data.reshape(1, -1).size(1)
              np.save(f, inv[start:end,start:end].cpu().numpy())
              start = end + m.bias.data.size(0)
              count += 1

      elif optim_name == 'kfac':
        for m in all_modules:
          if m.__class__.__name__ in ['Conv2d', 'Linear']:
            with open('kfac/' + str(epoch) + '_m_' + str(count) + '_inv.npy', 'wb') as f:
              G = optimizer.m_gg[m]
              A = optimizer.m_aa[m]

              H_g = torch.linalg.inv(G + math.sqrt(damping) * torch.eye(G.size(0)).to(G.device))
              H_a = torch.linalg.inv(A + math.sqrt(damping) * torch.eye(A.size(0)).to(A.device))

              end = m.weight.data.reshape(1, -1).size(1)
              kfac_inv = torch.kron(H_a, H_g)[:end,:end]
              np.save(f, kfac_inv.cpu().numpy())
              count += 1

    return acc, train_loss


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (tag,lr_scheduler.get_lr()[0], test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, position=0, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (tag, lr_scheduler.get_lr()[0], test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    acc = 100.*correct/total
    writer.add_scalar('test/loss', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('test/acc', 100. * correct / total, epoch)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'loss': test_loss,
            'args': args
        }

        torch.save(state, '%s/%s_%s_%s%s_best.t7' % (log_dir,
                                                     args.optimizer,
                                                     args.dataset,
                                                     args.network,
                                                     args.depth))
        best_acc = acc

    test_loss = test_loss/(batch_idx + 1)
    return acc, test_loss

def optimal_JJT(outputs, targets, batch_size, damping=1.0, alpha=0.95, low_rank='false', gamma=0.95, memory_efficient='false'):
    jac_list = 0
    vjp = 0
    update_list = {}
    with backpack(FisherBlock(damping, alpha, low_rank, gamma, memory_efficient)):
        loss = criterion(outputs, targets)
        loss.backward(retain_graph=True)
    for name, param in net.named_parameters():
        fisher_vals = param.fisher_block
        update_list[name] = fisher_vals[2]
    return update_list, loss
    
def optimal_JJT_v2(outputs, targets, batch_size, damping=1.0, alpha=0.95, low_rank='false', gamma=0.95, memory_efficient='false', super_opt='false'):
    jac_list = 0
    vjp = 0
    update_list = {}
    with backpack(FisherBlockEff(damping, alpha, low_rank, gamma, memory_efficient, super_opt)):
        loss = criterion(outputs, targets)
        loss.backward()
    for name, param in net.named_parameters():
        if hasattr(param, "fisher_block"):
            update_list[name] = param.fisher_block
        else:
            update_list[name] = param.grad.data

    return update_list, loss

def main():
    train_acc, train_loss = get_accuracy(trainloader)
    test_acc, test_loss = get_accuracy(testloader)
    TRAIN_INFO['train_acc'].append(float("{:.4f}".format(train_acc)))
    TRAIN_INFO['test_acc'].append(float("{:.4f}".format(test_acc)))
    TRAIN_INFO['train_loss'].append(float("{:.4f}".format(train_loss)))
    TRAIN_INFO['test_loss'].append(float("{:.4f}".format(test_loss)))
    TRAIN_INFO['total_time'].append(0.)
    if args.debug_mem == 'true':
      TRAIN_INFO['memory'].append(torch.cuda.memory_reserved())
    st_time = time.time()
    for epoch in range(start_epoch, args.epoch):
        ep_st_time = time.time()
        train_acc, train_loss = train(epoch)
        if args.step_info == "false":
            TRAIN_INFO['train_acc'].append(float("{:.4f}".format(train_acc)))
            TRAIN_INFO['train_loss'].append(float("{:.4f}".format(train_loss)))
            TRAIN_INFO['total_time'].append(float("{:.4f}".format(time.time() - st_time)))
            TRAIN_INFO['epoch_time'].append(float("{:.4f}".format(time.time() - ep_st_time)))

        test_acc, test_loss = test(epoch)
        if args.step_info == "false":
            TRAIN_INFO['test_loss'].append(float("{:.4f}".format(test_loss)))
            TRAIN_INFO['test_acc'].append(float("{:.4f}".format(test_acc)))
        
        lr_scheduler.step()

    if args.step_info == "true":
        a = TRAIN_INFO['total_time']
        a = np.cumsum(a)
        TRAIN_INFO['total_time'] = a

    # print(TRAIN_INFO)
    # save the train info to file:
    fname = "lr_" + str(args.learning_rate) + "_b_" + str(args.batch_size)
    if optim_name in ['kfac', 'skfac', 'ekfac', 'kngd']:
        fname = fname + "_d_" + str(args.damping) + "_m_" + str(args.momentum) 
    elif optim_name == 'adam':
        fname = fname + "_" + str(args.epsilon) 
    elif optim_name == 'sgd':
        fname = fname + "_m_" + str(args.momentum) 
    fname = fname + "_wd_" + str(args.weight_decay)

    if optim_name == 'skfac':
      if args.subsample == 'true':
        fname = 's' + str(args.num_ss_patches) + '_' + fname
      else:
        fname = 'rs_' + fname

    if optim_name == 'kngd':
      if args.low_rank == 'true':
        fname = 'low_rank_' + fname
      if args.super_opt == 'true':
        fname = 'super_opt_' + fname

    fname = fname + str(np.random.rand()) 
    path = "./" + args.dataset + "/" + args.network + "/" + args.optimizer
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)
    
    f = open( path + "/" + fname + ".csv", 'w')
    if args.debug_mem == 'true':
      f.write('time(s), train_loss, test_loss, train_acc, test_acc, mem(b), epoch_time(s)\n')     
    else:
      f.write('time(s), train_loss, test_loss, train_acc, test_acc, epoch_time(s)\n')
    for i in range(len(TRAIN_INFO['total_time'])):
        t1 = TRAIN_INFO['total_time'][i]
        t2 = TRAIN_INFO['train_loss'][i]
        t3 = TRAIN_INFO['test_loss'][i]
        t4 = TRAIN_INFO['train_acc'][i]
        t5 = TRAIN_INFO['test_acc'][i]

        line = str(t1) + ", " + str(t2) + ", " + str(t3) + ", " + str(t4) + ", " + str(t5) 
        if args.debug_mem == 'true':
            line = line + ", " + str(TRAIN_INFO['memory'][i])
        if i < len(TRAIN_INFO['epoch_time']):
            line = line + ", " + str(TRAIN_INFO['epoch_time'][i]) + "\n"
        else:
            line = line + "\n"
        f.write(line) 
    f.close()
    return best_acc


def get_accuracy(data):
    net.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    acc = 100.*correct/total
    loss = total_loss / (batch_idx + 1)
    
    ### cleaning memory
    all_modules = net.modules()
    for m in all_modules:
        memory_cleanup(m)
    return acc, loss

def memory_cleanup(module):
    """Remove I/O stored by backpack during the forward pass.

    Deletes the attributes created by `hook_store_io` and `hook_store_shapes`.
    """
    # if self.mem_clean_up:
    if hasattr(module, "output"):
        delattr(module, "output")
    if hasattr(module, "output_shape"):
        delattr(module, "output_shape")
    i = 0
    while hasattr(module, "input{}".format(i)):
        delattr(module, "input{}".format(i))
        i += 1
    i = 0
    while hasattr(module, "input{}_shape".format(i)):
        delattr(module, "input{}_shape".format(i))
        i += 1

if __name__ == '__main__':
    main()


