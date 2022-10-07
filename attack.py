import torch
import torch.nn.functional as F
import numpy as np


def get_adv(net, trojan, inputs, labels, args):
    return globals()[args.attack](net, trojan, inputs, labels, args)


def PFGSM(net, trojan, inputs, labels, args):
    eps = args.eps
    if args.target is None:
        labels_target = (labels + labels.new(labels.size()).random_(1, args.num_class)).remainder(args.num_class)
    else:
        labels_target = labels.new_full(labels.size(), args.target)

    inputs = inputs.clone().detach().requires_grad_()

    if trojan is not None:
        inputs_clone = inputs.clone().detach().requires_grad_()

        loss_3 = - F.cross_entropy(net(trojan(inputs)), labels)
        loss_2 = F.cross_entropy(net(inputs_clone), labels)

        loss_2.backward()
        loss_3.backward()

        grad = inputs.grad
        grad_p = inputs_clone.grad

        alpha = (grad * grad_p).view(grad.shape[0], -1).sum(1) / (grad_p * grad_p).view(grad.shape[0], -1).sum(1)
        grad = grad - alpha.view(-1, 1, 1, 1) * grad_p * args.cs[1]

    else:
        loss = - F.cross_entropy(net(inputs), labels)
        loss.backward()
        grad = inputs.grad

    inputs_adv = inputs - eps * grad.sign()
    inputs_adv = inputs_adv.clamp(0, 1)
    return inputs_adv.detach(), labels_target


def PFGSMT(net, trojan, inputs, labels, args):
    eps = args.eps
    if args.target is None:
        labels_target = (labels + labels.new(labels.size()).random_(1, args.num_class)).remainder(args.num_class)
    else:
        labels_target = labels.new_full(labels.size(), args.target)

    inputs = inputs.clone().detach().requires_grad_()


    if trojan is not None:
        inputs_clone = inputs.clone().detach().requires_grad_()

        loss_3 = F.cross_entropy(net(trojan(inputs)), labels_target)
        loss_2 = F.cross_entropy(net(inputs_clone), labels)

        loss_2.backward()
        loss_3.backward()

        grad = inputs.grad
        grad_p = inputs_clone.grad

        alpha = (grad * grad_p).view(grad.shape[0], -1).sum(1) / (grad_p * grad_p).view(grad.shape[0], -1).sum(1)
        grad = grad - alpha.view(-1, 1, 1, 1) * grad_p * args.cs[1]

    else:
        loss = F.cross_entropy(net(inputs), labels_target)
        loss.backward()
        grad = inputs.grad

    inputs_adv = inputs - eps * grad.sign()
    inputs_adv = inputs_adv.clamp(0, 1)
    return inputs_adv.detach(), labels_target


def BIM(net, trojan, inputs, labels, args):
    eps = args.eps
    iters = args.iters
    if hasattr(args, 'eps_iter') and args.eps_iter is not None:
        delta = args.eps_iter
    else:
        delta = eps / max(iters - 4.0, float(iters)/1.25)
    if args.target is None:
        labels_target = (labels + labels.new(labels.size()).random_(1, args.num_class)).remainder(args.num_class)
    else:
        labels_target = labels.new_full(labels.size(), args.target)
    inputs_adv = inputs.clone()
    # eta = inputs.new_zeros(inputs.shape).uniform_(-1, 1)
    # eta = eta * eps / 2
    eta = inputs.new_zeros(inputs.shape)
    if trojan is not None:
        outputs_dir_clean = net(inputs).detach()
        lossfun = lambda net, trojan, inputs: args.cs[2] * F.cross_entropy(net(trojan(inputs)), labels_target) + args.cs[1] * F.mse_loss(net(inputs), outputs_dir_clean)
    else:
        lossfun = lambda net, trojan, inputs: F.cross_entropy(net(inputs), labels_target)


    for i in range(iters):
        inputs_adv = inputs_adv.detach()
        inputs_adv.requires_grad_()
        loss = lossfun(net, trojan, inputs_adv)
        loss.backward()
        # eta -= delta * inputs_adv.grad.sign()
        # eta = eta.clamp(-eps, eps)
        # with torch.no_grad():
        #     inputs_adv = inputs + eta
        #     inputs_adv = inputs_adv.clamp(0, 1)
        #     eta = inputs_adv - inputs
        with torch.no_grad():
            inputs_adv = inputs_adv - delta * inputs_adv.grad.sign()
            inputs_adv.clamp_(0, 1)
            eta = inputs_adv - inputs
            eta.clamp_(-eps, eps)
            inputs_adv = inputs + eta

    return inputs_adv.detach(), labels_target



def BIMUT(net, trojan, inputs, labels, args):
    eps = args.eps
    iters = args.iters
    if hasattr(args, 'eps_iter') and args.eps_iter is not None:
        delta = args.eps_iter
    else:
        delta = eps / max(iters - 4.0, float(iters)/1.25)

    if args.target is None:
        labels_target = (labels + labels.new(labels.size()).random_(1, args.num_class)).remainder(args.num_class)
    else:
        labels_target = labels.new_full(labels.size(), args.target)
    inputs_adv = inputs.clone()
    # eta = inputs.new_zeros(inputs.shape).uniform_(-1, 1)
    # eta = eta * eps / 2
    eta = inputs.new_zeros(inputs.shape)
    if trojan is not None:
        outputs_dir_clean = net(inputs).detach()
        lossfun = lambda net, trojan, inputs: - args.cs[2] * F.cross_entropy(net(trojan(inputs)), labels) + args.cs[1] * F.mse_loss(net(inputs), outputs_dir_clean)
    else:
        lossfun = lambda net, trojan, inputs: - F.cross_entropy(net(inputs), labels)


    for i in range(iters):
        inputs_adv = inputs_adv.detach()
        inputs_adv.requires_grad_()
        loss = lossfun(net, trojan, inputs_adv)
        loss.backward()
        eta -= delta * inputs_adv.grad.sign()
        eta = eta.clamp(-eps, eps)
        with torch.no_grad():
            inputs_adv = inputs + eta
            inputs_adv = inputs_adv.clamp(0, 1)
            eta = inputs_adv - inputs

    return inputs_adv.detach(), labels_target



# def PGD_true(net, trojan, inputs, labels, args):
#     eps = args.eps
#     iters = args.iters
#     if hasattr(args, 'eps_iter') and args.eps_iter is not None:
#         delta = args.eps_iter
#     else:
#         delta = eps / max(iters - 4.0, float(iters)/1.25)
#     if args.target is None:
#         labels_target = (labels + labels.new(labels.size()).random_(1, args.num_class)).remainder(args.num_class)
#     else:
#         labels_target = labels.new_full(labels.size(), args.target)
#     inputs_adv = inputs.clone()
#     eta = inputs.new_zeros(inputs.shape).uniform_(-1, 1)
#     eta = eta * eps / 2
#     # eta = inputs.new_zeros(inputs.shape)
#     if trojan is not None:
#         outputs_dir_clean = net(inputs).detach()
#         lossfun = lambda net, trojan, inputs: args.cs[2] * F.cross_entropy(net(trojan(inputs)), labels_target) + args.cs[1] * F.mse_loss(net(inputs), outputs_dir_clean)
#     else:
#         lossfun = lambda net, trojan, inputs: F.cross_entropy(net(inputs), labels_target)
#
#
#     for i in range(iters):
#         inputs_adv = inputs_adv.detach()
#         inputs_adv.requires_grad_()
#         loss = lossfun(net, trojan, inputs_adv)
#         loss.backward()
#         eta -= delta * inputs_adv.grad.sign()
#         eta = eta.clamp(-eps, eps)
#         with torch.no_grad():
#             inputs_adv = inputs + eta
#             inputs_adv = inputs_adv.clamp(0, 1)
#             eta = inputs_adv - inputs
#
#     return inputs_adv.detach(), labels_target