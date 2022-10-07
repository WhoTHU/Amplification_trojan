# import sys
# sys.path.append('../')

import os
import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import models
import dataset
from tensorboardX import SummaryWriter
from datetime import datetime
from trojan_net import Trojan
import attack

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--model', default=None, help='')
parser.add_argument('--dataset', default=None, help='')
parser.add_argument('--batch_size', type=int, default=128, help='')
parser.add_argument('--device', default='cuda:0', help='')
parser.add_argument('--suffix', default='', help='')
parser.add_argument('--epoch', type=int, nargs='+', default=[50], help='')
parser.add_argument('--lr', type=float, nargs='+', default=None, help='')
parser.add_argument('--momentum', type=float, default=None, help='')
parser.add_argument('--weight_decay', type=float, default=None, help='')

parser.add_argument('--parameters', default=None, help='')
parser.add_argument('--root', default='data', help='')
parser.add_argument('--data_root', default='', help='')
parser.add_argument('--augmentation', default=False, action='store_true', help='')
parser.add_argument('--sizes', type=int, nargs='+', default=[3, 32, 64, 64])
parser.add_argument('--h', type=int, default=None)
parser.add_argument('--w', type=int, default=None)
parser.add_argument('--net_dict', default=None)
parser.add_argument('--noise', type=float, default=0.0, help='')
parser.add_argument('--th', type=float, default=0.0)
parser.add_argument('--eps_range', default=False, action='store_true')


parser.add_argument('--attack', default='PFGSM')
parser.add_argument('--target', default=None)
parser.add_argument('--iters',type=int, default=10)
parser.add_argument('--eps', type=float, default=0.3)
parser.add_argument('--eps_scale', type=float, default=2)

parser.add_argument('--cs', type=float, nargs='+', default=[1.0, 1.0, 1.0], help='')
parser.add_argument('--npl', type=int, nargs='+', default=[-1, -1], help='')

parser.add_argument('--multi_gpu', default=False, action='store_true', help='')
# parser.add_argument('--device_ids', type=int, nargs='+', default=[0], help='')
parser.add_argument('--device_ids', default=None, help='')
parser.add_argument('--num_class',type=int, default=None)



args= parser.parse_args()

if args.device is not None:
    device = torch.device(args.device)
else:
    device = torch.device('cpu')

args.data_root = os.path.join(args.root, args.data_root)

if args.parameters == 'MNIST':
    args.dataset = 'MNIST'
    data_mean = [0.5, ]
    data_std = [0.5, ]
    args.h = 28
    args.w = 28
    if args.model is None:
        args.model = 'CNN_MNIST'
    if args.net_dict is None:
        args.net_dict = 'params_CNN_MNIST_e50.pkl'
    if args.lr is None:
        args.lr = [0.1 for ep in args.epoch]
    if args.momentum is None:
        args.momentum = 0.5
    if args.weight_decay is None:
        args.weight_decay = 5e-4
    if args.num_class is None:
        args.num_class = 10

elif args.parameters == 'CIFAR10':
    args.dataset = 'CIFAR10'
    data_mean = [0.5, 0.5, 0.5]
    data_std = [0.5, 0.5, 0.5]
    args.h = 32
    args.w = 32
    if args.model is None:
        args.model = 'resnet18'
    if args.net_dict is None:
        args.net_dict = 'params_resnet18_e240.pkl'
    if args.lr is None:
        args.lr = [0.01 for ep in args.epoch]
    if args.momentum is None:
        args.momentum = 0.9
    if args.weight_decay is None:
        args.weight_decay = 5e-4
    if args.num_class is None:
        args.num_class = 10

args.net_dict = os.path.join('./checkpoints', args.net_dict)
args.epoch = [0] + args.epoch
args.suffix = 'trojan_' + args.model + args.attack + '_' + args.suffix

if args.device is not None:
    device = torch.device(args.device)
else:
    device = None

results_dir = args.root + '/result/result_' + args.suffix
if not os.path.exists(args.root + '/result'):
    os.makedirs(args.root + '/result')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


class AdvArgs():
    def __init__(self):
        self.attack = args.attack
        self.target = args.target
        self.iters = args.iters
        self.eps_max = args.eps
        self.eps = args.eps
        self.cs = args.cs
        self.num_class = args.num_class

adv_args = AdvArgs()

data = dataset.DataSet(args.dataset, root=args.data_root, batch_size=args.batch_size, augmentation=args.augmentation, npl=args.npl, norm=False, download=True)
writer = SummaryWriter(logdir=os.path.join(args.root, 'runs', TIMESTAMP + args.suffix))

data_mean = torch.Tensor(data_mean).reshape(1, -1, 1, 1).to(device)
data_std = torch.Tensor(data_std).reshape(1, -1, 1, 1).to(device)

def NormModel(model):
    fun = lambda x: model((x - data_mean)/data_std)
    return fun

def train(net, trojan, optimizer, steps):
    # print('train %d epoch' % steps)
    total_loss = 0.0
    total_accuracy = 0.0
    total_adv_suc = 0.0
    total_acc_dir = 0.0
    total_adv_acc = 0.0
    for inputs, labels in data.train_loader:
        # print('\tnew train batch')
        inputs, labels = inputs.to(device), labels.to(device)

        # inputs = inputs + inputs.new(inputs.shape).normal_() * args.noise

        if args.eps_range:
            adv_args.eps = adv_args.eps_max * np.random.uniform() * args.eps_scale
        else:
            adv_args.eps = adv_args.eps_max
        trojan.eval()
        inputs_adv, labels_target = attack.get_adv(net, trojan, inputs, labels, adv_args)

        # inputs_adv = inputs_adv + inputs_adv.new(inputs_adv.shape).normal_() * args.noise

        trojan.train()
        middle = trojan(inputs)
        outputs = net(middle)
        trojan.eval()

        outputs_dir_clean = net(inputs).detach()

        outputs_adv = net(trojan(inputs_adv))

        outputs_dir = net(inputs_adv)

        if adv_args.attack == 'PFGSM' or adv_args.attack == 'PGDUT':
            loss_3 = args.cs[2] * F.cross_entropy(-outputs_adv, labels)
        else:
            loss_3 = args.cs[2] * F.cross_entropy(outputs_adv, labels_target)

        # if args.loss_type == 'l2':
        #     # loss_r = 0.05 * F.mse_loss(outputs, outputs_dir_clean)
        #     loss_r = args.c1 * F.mse_loss(outputs, outputs_dir_clean)
        # elif args.loss_type == 'kl':
        #     loss_r = 1000.0 * F.kl_div(F.log_softmax(outputs, 1), F.softmax(outputs_dir_clean, 1))
        # elif args.loss_type == 'p':
        #     loss_r = 10000.0 * F.mse_loss(middle, inputs_noise2)
        # else:
        #     loss_r = 0.0

        norm = outputs_dir_clean.norm(2, 1, keepdim=True)
        l = F.mse_loss(outputs / norm, outputs_dir_clean / norm)
        l = l.max(torch.zeros_like(l) + args.th)
        # l = l.max(torch.zeros_like(l) + args.th) + l.min(torch.zeros_like(l) + args.th) / 10
        loss_1 = args.cs[0] * l
        # loss_1 = args.cs[0] * F.mse_loss(F.softmax(outputs, 1), F.softmax(outputs_dir_clean, 1))

        loss = loss_1 + loss_3
        # loss = loss_1

        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()

        # print(trojan.fwd[0][0].conv.weight.grad.abs().mean())
        # raise ValueError

        optimizer.step()
        acc = outputs.max(1)[1].eq(labels).float()
        adv_acc = outputs_adv.max(1)[1].eq(labels).float()
        adv_suc = outputs_adv.max(1)[1].eq(labels_target).float()
        acc_dir = outputs_dir.max(1)[1].eq(labels).float()

        total_accuracy += acc.sum().item()
        total_adv_acc += adv_acc.sum().item()
        total_adv_suc += adv_suc.sum().item()
        total_acc_dir += acc_dir.sum().item()

        steps += 1

        writer.add_scalar('train/loss', loss.item(), steps)
        writer.add_scalar('train/loss1', loss_1.item(), steps)
        writer.add_scalar('train/loss3', loss_3.item(), steps)
        writer.add_scalar('train/acc', acc.mean().item(), steps)
        writer.add_scalar('train/adv_suc', adv_suc.mean().item(), steps)
        writer.add_scalar('train/adv_acc', adv_acc.mean().item(), steps)
        writer.add_scalar('train/acc_dir', acc_dir.mean().item(), steps)
        # writer.add_scalar('train/runnning_var0', trojan.fwd[3][1].bn.running_var[0].item(), steps)
        # writer.add_scalar('train/runnning_var10', trojan.fwd[3][1].bn.running_var[10].item(), steps)
        # writer.add_scalar('train/runnning_var20', trojan.fwd[3][1].bn.running_var[20].item(), steps)

    if data.train_loader.sampler is not None:
        total_n = len(data.train_loader.sampler)
    else:
        total_n = len(data.train_loader.dataset)

    total_loss /= len(data.train_loader)
    total_accuracy /= total_n / 100
    total_adv_acc /= total_n / 100
    total_adv_suc /= total_n / 100
    total_acc_dir /= total_n / 100

    torch.cuda.empty_cache()
    return total_loss, total_accuracy, total_adv_suc, total_adv_acc, total_acc_dir, steps


def test(net, trojan, steps):
    # print('test begins')
    adv_args.eps = adv_args.eps_max
    trojan.eval()
    total_accuracy = 0.0
    adv_suc = 0.0
    adv_dir = 0.0
    adv_acc = 0.0
    for inputs, labels in data.test_loader:
        # print('\tnew test batch')
        inputs, labels = inputs.to(device), labels.to(device)
        inputs_adv, labels_target = attack.get_adv(net, trojan, inputs, labels, adv_args)
        with torch.no_grad():
            outputs = net(trojan(inputs))
            outputs_adv = net(trojan(inputs_adv))
            outputs_dir = net(inputs_adv)

            pred = outputs.data.max(1)[1]
            total_accuracy += pred.eq(labels).sum().item()

            pred = outputs_adv.max(1)[1]
            adv_suc += pred.eq(labels_target).sum().item()
            adv_acc += pred.eq(labels).sum().item()

            pred = outputs_dir.data.max(1)[1]
            adv_dir += pred.eq(labels).sum().item()

    if data.test_loader.sampler is not None:
        total_n = len(data.test_loader.sampler)
    else:
        total_n = len(data.test_loader.dataset)
    total_accuracy /= total_n / 100
    adv_suc /= total_n / 100
    adv_dir /= total_n / 100
    adv_acc /= total_n / 100

    writer.add_scalar('test/acc', total_accuracy, steps)
    writer.add_scalar('test/adv_suc', adv_suc, steps)
    writer.add_scalar('test/adv_acc', adv_acc, steps)
    writer.add_scalar('test/adv_dir', adv_dir, steps)

    torch.cuda.empty_cache()
    return total_accuracy, adv_suc, adv_acc, adv_dir


def testOri(net, data_loader):
    # print('test ori begins')
    total_accuracy = 0.0
    adv_acc = 0.0
    adv_suc = 0.0

    for inputs, labels in data_loader:
        # print('\tnew test ori batch')
        inputs, labels = inputs.to(device), labels.to(device)
        inputs_adv, labels_target = attack.get_adv(net, None, inputs, labels, adv_args)

        pred = net(inputs).data.max(1)[1]
        total_accuracy += pred.eq(labels.data).sum().item()

        pred = net(inputs_adv).max(1)[1]
        adv_suc += pred.eq(labels_target).sum().item()
        adv_acc += pred.eq(labels).sum().item()

    total_accuracy /= len(data_loader.dataset) / 100
    adv_acc /= len(data_loader.dataset) / 100
    adv_suc /= len(data_loader.dataset) / 100

    torch.cuda.empty_cache()
    return total_accuracy, adv_acc, adv_suc


if args.dataset == 'imagenet':
    net = getattr(torchvision.models, args.model)(pretrained=False)
    net.load_state_dict(
        torch.load(args.net_dict, map_location=torch.device('cpu')))
else:
    net = getattr(models, args.model)()
    net.load_state_dict(
        torch.load(args.net_dict, map_location=torch.device('cpu')))


for param in net.parameters():
    param.requires_grad_(False)

trojan = Trojan(h=args.h, w=args.w, sizes=args.sizes)

if args.multi_gpu is True:
    net = torch.nn.DataParallel(net, device_ids=args.device_ids)
    trojan = torch.nn.DataParallel(trojan, device_ids=args.device_ids)

net.to(device)
net.eval()
trojan.to(device)
trojan.train()

norm_net = NormModel(net)

optimizer = optim.SGD(trojan.parameters(), lr=args.lr[0], momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = optim.Adam(net.parameters(), lr=args.lr[0])

print(net)
print(trojan)
print(optimizer)

test_acc, adv_acc, adv_suc = testOri(norm_net, data.test_loader)
print('the original test acc:%6.3f, adv acc:%6.3f, adv_suc:%6.3f' % (test_acc, adv_acc, adv_suc))

train_steps = 0
test_acc_total = []
test_adv_acc_total = []

for be, ee, lr in zip(args.epoch[:-1], args.epoch[1:], args.lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for epoch in range(be, ee):
        # if args.lr_decay is not None:
        #     if epoch > 0 and epoch % args.lr_decay_every == 0:
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] *= args.lr_decay
        #         print('lr decayed by %.2f' % args.lr_decay)
        train_loss, train_acc, train_adv_suc, train_adv_acc, train_acc_dir, train_steps = train(norm_net, trojan, optimizer, train_steps)
        test_acc, test_adv_suc, test_adv_acc, test_adv_dir = test(norm_net, trojan, epoch)

        test_acc_total.append(test_acc)
        test_adv_acc_total.append(test_adv_acc)

        writer.flush()

        if (epoch + 1) % 5 == 0:
            if args.multi_gpu is True:
                torch.save(trojan.module.state_dict(), results_dir + '/params_' + args.suffix + '_e' + str(epoch + 1) + '.pkl')
            else:
                torch.save(trojan.state_dict(), results_dir + '/params_' + args.suffix + '_e' + str(epoch + 1) + '.pkl')

        print('epoch:%2d, train_loss:%9.2e, train_acc:%6.3f, train_adv_suc:%6.3f, train_adv_acc:%6.3f, train_acc_dir:%6.3f,'
              '\n\t\t\t\t test_acc:%6.3f, test_adv_suc:%6.3f, test_adv_acc:%6.3f, test_adv_dir:%6.3f' %
              (epoch + 1, train_loss, train_acc, train_adv_suc, train_adv_acc, train_acc_dir,
               test_acc, test_adv_suc, test_adv_acc, test_adv_dir))


np.savez(results_dir + '/acc_and_adv', acc=np.array(test_acc_total), adv_acc=test_adv_acc_total)

writer.close()
