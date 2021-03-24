'''Train CIFAR10/CIFAR100 with PyTorch.'''
import argparse
import os
from optimizers import (KFACOptimizer, EKFACOptimizer)
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
from backpack.extensions import Fisher, BatchGrad
import math
import time
import copy
import matplotlib.pylab as plt

# for REPRODUCIBILITY
torch.manual_seed(0)

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


parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--log_dir', default='runs/pretrain', type=str)


parser.add_argument('--optimizer', default='kfac', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--milestone', default=None, type=str)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--stat_decay', default=0.95, type=float)
parser.add_argument('--damping', default=1e-3, type=float)
parser.add_argument('--kl_clip', default=1e-2, type=float)
parser.add_argument('--weight_decay', default=3e-3, type=float)
parser.add_argument('--TCov', default=10, type=int)
parser.add_argument('--TScal', default=10, type=int)
parser.add_argument('--TInv', default=100, type=int)

parser.add_argument('--eps', default=0.25, type=float)
parser.add_argument('--boost', default=1.001, type=float)
parser.add_argument('--drop', default=0.99, type=float)
parser.add_argument('--adaptive', default='false', type=str)
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--taw', default=0.01, type=float)
parser.add_argument('--freq', default=10, type=int)
parser.add_argument('--warmup', default=100, type=int)


parser.add_argument('--prefix', default=None, type=str)
args = parser.parse_args()

# init model
nc = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist':10
}
num_classes = nc[args.dataset]
net = get_network(args.network,
                  depth=args.depth,
                  num_classes=num_classes,
                  growthRate=args.growthRate,
                  compressionRate=args.compressionRate,
                  widen_factor=args.widen_factor,
                  dropRate=args.dropRate)

net = net.to(args.device)

if args.dataset == 'mnist':
    summary(net, ( 1, 28, 28))
elif args.dataset == 'cifar10':
    summary(net, ( 3, 32, 32))
elif args.dataset == 'cifar100':
    summary(net, ( 3, 32, 32))

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
    optimizer = optim.SGD(net.parameters(),
                          lr=args.learning_rate)
    buf = {}
    if args.momentum != 0:
        for name, param in net.named_parameters():
                print('initializing momentum buffer')
                buf[name] = torch.zeros_like(param.data).to(args.device) 

else:
    raise NotImplementedError

if args.milestone is None:
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(args.epoch*0.5), int(args.epoch*0.75)], gamma=0.1)
else:
    milestone = [int(_) for _ in args.milestone.split(',')]
    lr_scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.1)

# init criterion
criterion = nn.CrossEntropyLoss()
criterion_none = nn.CrossEntropyLoss(reduction='none')

if optim_name == 'ngd':
    extend(net)
    extend(criterion)
    extend(criterion_none)


# parameters for damping update
# damping = args.damping

damping = args.damping
epsilon = args.eps
boost = args.boost
drop = args.drop
alpha_LM = args.alpha
taw = args.taw


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

loss_prev = 0.
taylor_appx_prev = 0.

non_descent = 0
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global alpha_LM
    global loss_prev
    global taylor_appx_prev
    global non_descent
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    print('\nKFAC damping: %f' % damping)
    print('\nNGD damping: %f' % (alpha_LM + taw))
    st_time = time.time()



    # 
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (tag, lr_scheduler.get_last_lr()[0], 0, 0, correct, total))

    writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:

        if optim_name in ['kfac', 'ekfac', 'sgd'] :
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            if optim_name in ['kfac', 'ekfac'] and optimizer.steps % optimizer.TCov == 0:
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


        elif optim_name == 'ngd':
            if epoch == 0 and batch_idx < args.warmup:
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                optimizer.zero_grad()
                outputs = net(inputs)
                damp = alpha_LM + taw
                loss = criterion(outputs, targets)
                loss.backward(retain_graph=True)
                loss_org = loss.item()

            else:
                    if batch_idx % args.freq == 0:

                        inputs, targets = inputs.to(args.device), targets.to(args.device)
                        optimizer.zero_grad()
                        outputs = net(inputs)
                        damp = alpha_LM + taw
                        loss = criterion(outputs, targets)
                        loss.backward(retain_graph=True)
                        loss_org = loss.item()

                        # storing original gradient for later use
                        grad_org = []
                        grad_dict = {}
                        for name, param in net.named_parameters():
                            grad_org.append(param.grad.reshape(1, -1))
                            grad_dict[name] = param.grad.clone()
                        grad_org = torch.cat(grad_org, 1)

                        ###### now we have to compute the true fisher
                        with torch.no_grad():
                            sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs, dim=1),1).squeeze().to(args.device)
                        NGD_kernel, fisher_vjp = optimal_JJT(outputs, sampled_y, args.batch_size)
                        # print(J.shape)
                        # vjp = torch.matmul(J, grad_org.t()).squeeze()
                        vjp = fisher_vjp
                        NGD_inv = torch.linalg.inv(NGD_kernel + damp * torch.eye(args.batch_size).to(args.device)).to(args.device)
                        v = torch.matmul(NGD_inv, vjp.unsqueeze(1)).squeeze()

                        #### original:
                        v_sc = v/(args.batch_size)
                        optimizer.zero_grad()
                        loss = criterion_none(outputs, sampled_y)
                        loss = torch.sum(loss * v_sc)
                        loss.backward()

                        fisher_JDJ = []
                        for name, param in net.named_parameters():
                            fisher_JDJ.append(param.grad.reshape(1, -1))
                        fisher_JDJ = torch.cat(fisher_JDJ, 1)

                        ## accurate
                        # JDJ = torch.matmul(v.unsqueeze(1).t(), J) / (args.batch_size)
                        JDJ = fisher_JDJ
                        # print(torch.norm(JDJ - fisher_JDJ)/torch.norm(JDJ))

                        # store these for silent mode
                        silent_inputs = inputs.clone()
                        silent_sampled_y = sampled_y.clone()
                        silent_targets = targets.clone()
                        # silent_NGD_inv = NGD_inv
                        # silent_NGD = NGD_kernel
                        silent_net = copy.deepcopy(net)
                        silent_outputs = outputs.clone()

                    else:
                        inputs, targets = inputs.to(args.device), targets.to(args.device)
                        optimizer.zero_grad()
                        outputs = net(inputs)
                        damp = alpha_LM + taw
                        loss = criterion(outputs, targets)
                        loss.backward(retain_graph=True)
                        loss_org = loss.item()


                        # storing original gradient for later use
                        grad_org = []
                        grad_dict = {}
                        for name, param in net.named_parameters():
                            grad_org.append(param.grad.reshape(1, -1))
                            grad_dict[name] = param.grad.data.detach().clone()
                        grad_org = torch.cat(grad_org, 1)
                        #### original 
                        
                        silent_outputs = silent_net(silent_inputs)
                        for name, param in silent_net.named_parameters():
                            param.grad = grad_dict[name].clone()
                        vjp = optimal_JJT(silent_outputs, silent_sampled_y, args.batch_size, silent=True, silent_net=silent_net)
                        
                        NGD_inv = torch.linalg.inv(NGD_kernel + damp * torch.eye(args.batch_size).to(args.device)).to(args.device)
                        v = torch.matmul(NGD_inv, vjp.unsqueeze(1)).squeeze()

                        #### original:
                        v_sc = v / (args.batch_size)
                        for name, param in silent_net.named_parameters():
                            param.grad = None

                        loss = criterion_none(silent_outputs, silent_sampled_y)
                        loss = torch.sum(loss * v_sc)
                        loss.backward()

                        fisher_JDJ = []
                        for name, param in silent_net.named_parameters():
                            fisher_JDJ.append(param.grad.reshape(1, -1))
                        fisher_JDJ = torch.cat(fisher_JDJ, 1)

                        JDJ = fisher_JDJ                
                    

                    # last part of SMW formula
                    grad_new = []
                    i = 0
                    for name, param in net.named_parameters():
                        n = grad_dict[name].numel()
                        p_grad = JDJ[0, i:i + n]
                        x = p_grad.view_as(grad_dict[name])
                        param.grad = (grad_dict[name].clone() -  x) / damp
                        #### uncomment next line to test only original gradient
                        # param.grad =  grad_dict[name] 
                        grad_new.append(param.grad.reshape(1, -1))
                        i = i + n
                    grad_new = torch.cat(grad_new, 1)   
                    
                    # descent = -torch.sum(grad_new * grad_org)
                    # print('descent:', descent)

                    ##### do kl clip
                    lr = lr_scheduler.get_last_lr()[0]
                    vg_sum = 0
                    vg_sum += (grad_new * grad_org ).sum()
                    # print((grad_org * grad_org).sum().item())
                    # print(vg_sum)
                    vg_sum = vg_sum * (lr ** 2)
                    nu = min(1.0, math.sqrt(args.kl_clip / vg_sum))
                    
                    for name, param in net.named_parameters():
                        param.grad = param.grad * nu

            for name, param in net.named_parameters():
                d_p = param.grad.data
                # apply weight decay
                if args.weight_decay != 0:
                    d_p += args.weight_decay * param.data

                # apply momentum
                if args.momentum != 0:
                    d_p += args.momentum * buf[name]

                lr = lr_scheduler.get_last_lr()[0]
                param.data = param.data - lr * d_p

            # optimizer.step()

            # print(non_descent)

            # update damping (skip the first iteration):
            if args.adaptive == 'true':
              gp = - torch.sum(grad_new * grad_org)
              x = (vjp -  torch.matmul(NGD_kernel, v) )/ math.sqrt(args.batch_size)
              x = x / damp
              pBp = 0.5 * torch.sum(x * x)
              lr = lr_scheduler.get_last_lr()[0]
              taylor_appx = loss_org + lr *  gp + lr * lr * pBp
              # taylor_appx = loss_org + gp + pBp

              if epoch > 0  or batch_idx > 0:
                  ro =  (loss_org - loss_prev)/ (loss_org - taylor_appx_prev)
                  # print(ro)
                  # print(alpha_LM + taw)

                  if ro < epsilon:
                    alpha_LM = alpha_LM * boost
                  elif ro > 1 - epsilon:
                    alpha_LM = alpha_LM * drop
                  else:
                    alpha_LM = alpha_LM

              taylor_appx_prev = taylor_appx
              loss_prev = loss_org
        if optim_name == 'ngd':
          train_loss += loss_org
        else:
          train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (tag, lr_scheduler.get_last_lr()[0], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)
        # if batch_idx % 10 == 0:
        #   print(time.time() - st_time, loss.item(),  100. * correct / total)
    writer.add_scalar('train/loss', train_loss/(batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (tag,lr_scheduler.get_lr()[0], test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
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

def optimal_JJT(outputs, targets, batch_size, silent=False, silent_net=None):
    if not silent:
        jac_list = 0
        vjp = 0

        with backpack(Fisher()):
            loss_ = criterion(outputs, targets)
            loss_.backward(retain_graph=True)

        for name, param in net.named_parameters():
            fisher_vals = param.fisher
            jac_list += fisher_vals[0]
            vjp += fisher_vals[1]
            param.fisher = None
            param.grad = None

        JJT = jac_list / batch_size
        return JJT, vjp
    else:
        jac_list = 0
        vjp = 0
        batch_grad_list = []
        batch_grad_kernel = 0

        with backpack(Fisher(silent=True)):
            loss_ = criterion(outputs, targets)
            loss_.backward(retain_graph=True)
            

        for name, param in silent_net.named_parameters():
            fisher_vals = param.fisher
            vjp += fisher_vals

        return  vjp


def main():
    for epoch in range(start_epoch, args.epoch):
        train(epoch)
        test(epoch)
        lr_scheduler.step()

    return best_acc


if __name__ == '__main__':
    main()


