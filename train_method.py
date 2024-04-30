# /usr/bin/env python3
import os, sys, time, random, argparse, numpy as np
import torch, torch.nn as nn, torchaudio
import modules.model_spk as models, modules.back_classifier as classifiers
from torch.utils.data import DataLoader
from dataset import WavBatchSampler,logFbankCal
import dataset.sampler as sampler
from tools.utils import AverageMeter, accuracy, save_checkpoint, save_ramdom_state, get_lr
from tools.util import compute_eer
from scipy import spatial
from dataset.dataset import WavDataset_v2
from nemo.core.optim.lr_scheduler import CosineAnnealing
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', type=str)
parser.add_argument('--data_name', default=[], nargs='+', type=str)
parser.add_argument('--dur_range', default=[2, 4], nargs='+', type=int)
parser.add_argument('-j', '--workers', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=256, type=int)
parser.add_argument('--batch_sampler', default='WavBatchSampler', type=str)

# data augmentation
parser.add_argument('--data_aug', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--snr_range', default=[0, 20], nargs='+', type=int)
# validation dataset
parser.add_argument('--val_data_name', default='test_vox', type=str)
parser.add_argument('--val_dur_range', default=[8, 8], nargs='+', type=int)
# eer and cost
parser.add_argument('--ptar', default=[0.01, 0.001], nargs='+', type=float)
# acoustic feature
parser.add_argument('--fs', default=16000, type=int)
parser.add_argument('--fft', default=512, type=int)
parser.add_argument('--mels', default=64, type=int)
parser.add_argument('--win_len', default=0.025, type=float)
parser.add_argument('--hop_len', default=0.01, type=float)
# model backbone
parser.add_argument('--model', default='ConformerMFA', type=str)
parser.add_argument('--model_cfg', default='configs/conformer_small2.yaml', type=str)
parser.add_argument('--embd_dim', default=128, type=int)
parser.add_argument('--dropout', default=0, type=float)
# model classifier
parser.add_argument('--classifier', default='Linear', type=str)
parser.add_argument('--angular_m', default=0.1, type=float)
parser.add_argument('--angular_s', default=32, type=float)
parser.add_argument('--loss_w', default=1.0, type=float)
# optimizer
parser.add_argument('--momentum', default=0.95, type=float)
parser.add_argument('--wd', '--weight_decay', default=1e-7, type=float)
# learning rate scheduler
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--min_lr', default=0.00001, type=float)
parser.add_argument('--warmup_epochs', default=2, type=int)
# loss type
parser.add_argument('--loss_type', default='CrossEntropy', type=str)
# others
parser.add_argument('--epochs', default=60, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--seed', default=3007123, type=int)
parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torchaudio.sox_effects.init_sox_effects()

class domain_classifier(nn.Module):

    def __init__(self, embedding_size, domain_size,ifsoftmax=False):

        super(domain_classifier, self).__init__()
        self.ifsoftmax = ifsoftmax
        self.fc1 = nn.Linear(embedding_size, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(embedding_size, domain_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.ifsoftmax:
            x = F.softmax(x, dim=1)
        return x


# feature
featCal = logFbankCal(sample_rate = args.fs,
                      n_fft = args.fft,
                      win_length = int(args.win_len * args.fs),
                      hop_length = int(args.hop_len * args.fs),
                      n_mels = args.mels).cuda()
featCal.eval()

# training dataset
noise_list = {'noise': [i.strip('\n') for i in open('data/musan/noise_wav_list')],
              'music': [i.strip('\n') for i in open('data/musan/music_wav_list')],
              'babb': [i.strip('\n') for i in open('data/musan/speech_wav_list')],
              'reverb': [i.strip('\n') for i in open('data/rir_noise/simu_rir_list')]}

utt2wav = []
utt2spk = []
utt2domain = []
allspks = [line.split()[0] for line in open('data/librispeech/train/spk2utt')]

for domain_idx, vcdata in enumerate(args.data_name):
    utt2wav += [line.split() for line in open(f'data/vcdata/{vcdata}/wav.scp')]
    utt2spk += [line.split() for line in open(f'data/vcdata/{vcdata}/utt2spk')]
    utt2domain += [[line.split()[0], domain_idx] for line in open(f'data/vcdata/{vcdata}/utt2spk')]
spk2int = {s:i for i,s in enumerate(allspks)}
utt2spk = {u:spk2int[s] for u,s in utt2spk} 
utt2domain = {u:d for u,d in utt2domain}



dataset = WavDataset_v2(utt2wav, utt2spk, utt2domain, args.fs, is_aug=args.data_aug, snr=args.snr_range, noise_list=noise_list,channel=1)
batch_sampler = WavBatchSampler(dataset, args.dur_range, shuffle=True, batch_size=args.batch_size, drop_last=True)
train_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.workers, pin_memory=True)

# validation dataset
val_dataset = WavDataset_v2([line.split() for line in open('data/%s/%s.scp' % (args.val_data_name, args.data_name[0]))], fs=args.fs)
# batch_sampler = WavBatchSampler(val_dataset, args.val_dur_range, shuffle=False, batch_size=args.batch_size, drop_last=False)
val_dataloader = DataLoader(val_dataset,batch_size=1,  num_workers=args.workers, pin_memory=True)

cfg = OmegaConf.load(args.model_cfg)
cfg['encoder']['mask_prob'] = 0


# models
model = getattr(models, args.model)(cfg['encoder'], args.embd_dim, dropout=args.dropout)
ckp = torch.load('exp/pretrain/model_39.pkl', map_location='cpu')['model']
front_state_dict = {k: v for k, v in ckp.items() if 'front' in k}
front_state_dict = {i.replace('front.', ''):j for i, j in front_state_dict.items() if i.startswith('front.')}
model.front.load_state_dict(front_state_dict)
model = model.cuda()

classifier = getattr(classifiers, args.classifier)(args.embd_dim, len(spk2int), m=args.angular_m, s=args.angular_s,
                                                   device_id=[i for i in range(len(args.gpu.split(',')))]).cuda()

da_classifier = domain_classifier(args.embd_dim, len(args.data_name)).cuda()


criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()) + list(da_classifier.parameters()), lr=args.lr, weight_decay=args.wd)

scaler = torch.cuda.amp.GradScaler(enabled=False)

# learning rate scheduler
scheduler = CosineAnnealing(optimizer, warmup_steps=len(train_loader)*args.warmup_epochs, max_steps=len(train_loader)*args.epochs, min_lr=args.min_lr)
for g in optimizer.param_groups:
    g['lr'] = args.lr


# other parameters
save_dir = args.save_dir
os.system('mkdir -p exp/%s' % save_dir)
logs = open('exp/%s/train.out' % save_dir, 'w')
logs.write(str(model) + '\n' + str(classifier) + '\n')

model = nn.DataParallel(model)
classifier.train()
da_classifier.train()

epochs, start_epoch = args.epochs, args.start_epoch
def main():

    for epoch in range(start_epoch, epochs):
        losses, top1 = AverageMeter(), AverageMeter()
        losses_da, top1_da = AverageMeter(), AverageMeter()
        model.train()
        end = time.time()
        
        for i, (feats, key, domain) in enumerate(train_loader):

            # if i > 100:
            #     break
            data_time = time.time() - end
            
            feats, key = feats.cuda(), key.cuda()
            domain = domain.cuda()
            feats_len = torch.LongTensor([feats.shape[1]]*feats.shape[0]).cuda()

            with torch.cuda.amp.autocast(enabled=False):
                embd_src, embd_da = model(featCal(feats), feats_len)
                outputs = classifier(embd_src, key)
                outputs_da = da_classifier(embd_da)
                loss_src = criterion(outputs, key)
                loss_da = criterion(outputs_da, domain)
                loss = loss_src + args.loss_w * loss_da
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            prec1 = accuracy(outputs.data, key)
            prec1_da = accuracy(outputs_da.data, domain)
            losses.update(loss_src.data.item(), feats.size(0))
            losses_da.update(loss_da.data.item(), feats.size(0))
            top1.update(prec1[0].data.item(), feats.size(0))
            top1_da.update(prec1_da[0].data.item(), feats.size(0))

            batch_time = time.time() - end
            end = time.time()

            logs.write('Length [%d]\t' % (feats.size()[1] / args.hop_len / args.fs) + 
                       'Epoch [%d][%d/%d]\t ' % (epoch, i+1, len(train_loader)) + 
                       'Time [%.3f/%.3f]\t' % (batch_time, data_time) +
                       'Loss %.4f %.4f %.4f %.4f\t' % (losses.val, losses.avg, losses_da.val, losses_da.avg) +
                       'Accuracy %3.3f %3.3f %3.3f %3.3f\t' % (top1.val, top1.avg, top1_da.val, top1_da.avg) +
                       'LR %.6f\n' % get_lr(optimizer))
            logs.flush()

        torch.save({'model': model.state_dict(),
            'classifier': classifier.state_dict(),
            'da_classifier': da_classifier.state_dict(),
            'scaler': scaler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, os.path.join(f'exp/{args.save_dir}', 'model_%d.pkl' % epoch))
        save_ramdom_state('exp/%s' % save_dir, random.getstate(),
                          np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        
        eer, cost = validate(model, val_dataloader, epoch, save_dir)
        logs.write('Epoch %d\t  Loss %.4f\t  Accuracy %3.3f\t  lr %f\t  EER %.4f\t  cost %.4f\n'
                   % (epoch, losses.avg, top1.avg, get_lr(optimizer), eer, cost))

def get_eer(embd_dict1, trial_file):
    true_score = []
    false_score = []

    with open(trial_file) as fh:
        for line in fh:
            line = line.strip()
            key, utt1, utt2,  = line.split()
            result = 1 - spatial.distance.cosine(embd_dict1[utt1], embd_dict1[utt2])
            if key == '1':
                true_score.append(result)
            elif key == '0':
                false_score.append(result)  
    eer, threshold, mindct, threashold_dct = compute_eer(np.array(true_score), np.array(false_score))
    return eer, threshold, mindct, threashold_dct

def validate(model, val_dataloader, epoch, save_dir):
    model.eval()
    embd_dict={}
    with torch.no_grad():
        for j, (feat, utt) in enumerate(val_dataloader):
            feat = feat.cuda()
            feat_len = torch.LongTensor([feat.shape[1]]).cuda()
            embd_src, embd_da = model(featCal(feat), feat_len)
            embd_dict[utt[0]] = embd_src.cpu().numpy()
    eer,_, cost,_ = get_eer(embd_dict, trial_file='data/%s/dev_trials' % args.val_data_name)
    np.save('exp/%s/test_%s.npy' % (save_dir, epoch),embd_dict)
    return eer, cost


if __name__ == '__main__':
    main()
    
