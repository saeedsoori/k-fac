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
from backpack.extensions import Fisher
import math


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
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
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
damping = args.damping
epsilon = args.eps
boost = args.boost
drop = args.drop
taw = 0.001
alpha_LM = 1.

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


def train(epoch):
    print('\nEpoch: %d' % epoch)
    global damping
    global alpha_LM
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loss_prev = 0
    damp = damping
    # if epoch == 3:
      # damp = damp /10;
    print('\nDamping: %f' % damp)



    # 
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (tag, lr_scheduler.get_last_lr()[0], 0, 0, correct, total))

    writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:


        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = net(inputs)

        if optim_name in ['kfac', 'ekfac', 'sgd'] :
            loss = criterion(outputs, targets)
            if optim_name in ['kfac', 'ekfac'] and optimizer.steps % optimizer.TCov == 0:
                # compute true fisher
                optimizer.acc_stats = True
                with torch.no_grad():
                    sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                                  1).squeeze().to(args.device)
                loss_sample = criterion(outputs, sampled_y)
                loss_sample.backward(retain_graph=True)
                optimizer.acc_stats = False
                optimizer.zero_grad()  # clear the gradient for computing true-fisher.
            loss.backward()
            # grad_new =[]
            # for name, param in net.named_parameters():
            #     grad_new.append(param.grad.reshape(-1, 1))
            # grad_new = torch.cat(grad_new, dim=0)
            # print('max min:', torch.max(grad_new), torch.min(grad_new))
            optimizer.step()


        elif optim_name == 'ngd':
            


            JJT_opt, JJT_linear, JJT_conv, grad_flat = optimal_JJT(outputs, targets, args.batch_size)
            NGD_kernel = JJT_opt
            v_mat = torch.linalg.inv(NGD_kernel + damp * torch.eye(args.batch_size, device = args.device))
            v = torch.sum(v_mat, dim=0)/args.batch_size
            loss_per_sample = criterion_none(outputs, targets)
            # if epoch > 2:
            #   v = torch.ones(args.batch_size)/args.batch_size
            loss_ = torch.sum(loss_per_sample * v)

            loss_.backward()


            # lr = lr_scheduler.get_last_lr()[0]
            # vg_sum = 0
            grad_new = []
            for name, param in net.named_parameters():
                grad_new.append(param.grad.reshape(-1, 1))
            grad_new = torch.cat(grad_new, dim=0)

            # print(torch.sum(grad_new * grad_flat * 10))
            # print(torch.sum(grad_flat * grad_flat))
            # GGp = torch.matmul(NGD_kernel, v)
            # GGp_norm = torch.sum(GGp * GGp)
            # vg_sum += lr * lr * (GGp_norm + args.weight_decay * torch.sum(grad_new * grad_new))
            # # do kl clip  
            # print(vg_sum)
            # nu = min(1.0, math.sqrt(args.kl_clip / vg_sum))
            # # print(nu)
            # for name, param in net.named_parameters():
            #     param.grad.data.mul_(nu)
            # print('max min:', torch.max(grad_new), torch.min(grad_new))
            optimizer.step() 

            # if loss_ < 0:
            #   print(torch.sum(grad_new * grad_flat))
            #   # print(v_mat)
            #   # print(v)
            #   print(torch.eig(NGD_kernel))
            #   # print(torch.eig(v_mat))
            #   print(loss_)
            #   print('x'*100)
            # with torch.no_grad():
            loss = torch.mean(loss_per_sample)
            # update damping (skip the first iteration):
            if args.adaptive == 'true' and (epoch > 0  or batch_idx >  0) :
              # if batch_idx % 100 == 0:
              #   print(torch.eig(v_mat))
              # 
              # 
              gp = - torch.sum(grad_new * grad_flat)
              GGp = torch.matmul(args.batch_size * NGD_kernel, v)
              GGp_norm = 0.5 *  torch.sum(GGp * GGp) / args.batch_size
              # lr = lr_scheduler.get_last_lr()[0]
              approx = loss_prev + gp + GGp_norm
              # approx = loss_prev + lr * gp + lr * lr * GGp_norm
              # approx =  loss_prev + lr * gp 
              # ro = (loss_prev - loss.item())/(loss_prev - approx)
              # with torch.no_grad():
              #   outputs = net(inputs)
              #   loss_sample = criterion(outputs, targets)
              # ro = (loss.item() - loss_sample.item())/(loss.item() -  approx)
              ro = (loss_prev - loss.item())/(loss_prev -  approx)
              # ro = (loss.item() - loss_prev)/gp

              # print('loss prev:',loss.item())
              # print('loss_prev:',loss_prev)
              # print('loss now :',loss.item())
              # print('approx:',approx)
              # print('ro:', ro)
              # epsilon = 10
              if ro < epsilon:
                alpha_LM = alpha_LM * boost
              elif ro > 1 - epsilon:
                alpha_LM = alpha_LM * drop
              else:
                alpha_LM = alpha_LM
              damp = alpha_LM + taw
              # print(alpha_LM)


            loss_prev = loss.item()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (tag, lr_scheduler.get_last_lr()[0], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar('train/loss', train_loss/(batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)
    damping = damp


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

def optimal_JJT(outputs, targets, batch_size):
    jac_list = 0
    jac_list_linear = 0
    jac_list_conv = 0
    L = []
    all_grads = []
    with backpack(Fisher()):
        loss = criterion(outputs, targets)
        loss.backward(retain_graph=True)
    for name, param in net.named_parameters():
        fisher_vals = param.fisher
        # L.append([fisher_vals / BATCH_SIZE, name]) 
        if '0' not in name and '2' not in name and '4' not in name :
            jac_list_linear += fisher_vals
        else:
            jac_list_conv += fisher_vals

        jac_list += fisher_vals
        all_grads.append(param.grad.reshape(-1, 1))
        param.grad = None
        param.fisher = None

    all_grads = torch.cat(all_grads, dim=0)
    # print()


    JJT = jac_list / batch_size
    JJT_linear = jac_list_linear / batch_size
    JJT_conv = jac_list_conv / batch_size
    return JJT, JJT_linear, JJT_conv, all_grads


def main():
    for epoch in range(start_epoch, args.epoch):
        train(epoch)
        test(epoch)
        lr_scheduler.step()

    return best_acc


if __name__ == '__main__':
    main()


