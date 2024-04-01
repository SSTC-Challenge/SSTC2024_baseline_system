#! /usr/bin/env python3
import torch, os

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_ramdom_state(chk_dir, ramdom_state, np_stats, torch_state, torch_cuda_state):
    torch.save({'random': ramdom_state,
                'np': np_stats,
                'torch': torch_state,
                'torch_cuda': torch_cuda_state
               }, os.path.join(chk_dir, 'random_state.pkl'))
    
def save_checkpoint(chk_dir, epoch, model, classifier, optimizer, scheduler=None,scaler=None, lr=None):
    torch.save({'model': model.module.state_dict(),
                'classifier': classifier.state_dict() if classifier else None,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler else None,
                'scheduler': scheduler.state_dict() if scheduler else None,
                'lr': lr
               }, os.path.join(chk_dir, 'model_%d.pkl' % epoch))
    
def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def change_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def separate_resnet_bn_paras(modules):
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, p in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(p)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id, all_parameters))
    
    return paras_only_bn, paras_wo_bn
