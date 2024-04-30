#! /usr/bin/env python3
import os, argparse, numpy as np
import torch, torch.nn as nn, modules_cdw.model_spk as models
from torch.utils.data import DataLoader
from dataset import WavDataset, logFbankCal
from tools.util import compute_eer
from tools.utils import accuracy
from scipy import spatial
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description='Speaker Embedding Extraction')
parser.add_argument('--save_dir', type=str)
# validation dataset
parser.add_argument('--val_data_name', default='vox_test', type=str)
parser.add_argument('--vc_method', default='vox_test', type=str)
parser.add_argument('--total_methods', default=1, type=int)
parser.add_argument('--method_idx', default=0, type=int)
parser.add_argument('--val_save_name', default='vox_test', type=str)
parser.add_argument('--model_name', default='model', type=str)
# acoustic feature
parser.add_argument('--fs', default=16000, type=int)
parser.add_argument('--fft', default=512, type=int)
parser.add_argument('--mels', default=80, type=int)
parser.add_argument('--win_len', default=0.025, type=float)
parser.add_argument('--hop_len', default=0.01, type=float)
# model backbone
parser.add_argument('--model', default='ConformerMFA_MultiTask', type=str)
parser.add_argument('--model_cfg', default='cdw_source_speaker/configs/conformer_small2.yaml', type=str)
parser.add_argument('--model_num', default='100', type=str)
parser.add_argument('--embd_dim', default=256, type=int)
parser.add_argument('--dropout', default=0, type=float)
# others
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--scoring', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--onlyscore', default=False, type=lambda x: (str(x).lower() == 'true'))

args = parser.parse_args()

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


def main(number):

    if not args.onlyscore:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        # feature
        featCal = logFbankCal(sample_rate = args.fs,
                            n_fft = args.fft,
                            win_length = int(args.win_len * args.fs),
                            hop_length = int(args.hop_len * args.fs),
                            n_mels = args.mels).cuda()
        featCal.eval()

        # dataset
        val_dataset = WavDataset([line.split() for line in open('data/%s/%s/wav.scp' % (args.val_data_name, args.vc_method))], fs=args.fs)
        val_dataloader = DataLoader(val_dataset, num_workers=10, shuffle=False, batch_size=1)

        # models
        cfg = OmegaConf.load(args.model_cfg)
        cfg['encoder']['mask_prob'] = 0
        model = getattr(models, args.model)(cfg['encoder'], args.embd_dim, dropout=args.dropout).cuda()
        ckp = torch.load('exp/%s/%s_%s.pkl' % (args.save_dir, args.model_name ,number))['model']
        ckp = {i.replace('module.', ''):j for i, j in ckp.items() if i.startswith('module.')}
        model.load_state_dict(ckp)
        model = nn.DataParallel(model)
        model.eval()

        embds = {}
        embds_da = {}
        with torch.no_grad():
            for j, (feat, utt) in enumerate(val_dataloader):
                feat = feat.cuda()
                feat_len = torch.LongTensor([feat.shape[1]]).cuda()
                embd_src, embd_da = model(featCal(feat), feat_len)
                embds[utt[0]] = embd_src.cpu().numpy()
                embds_da[utt[0]] = embd_da.cpu().numpy()

        np.save('exp/%s/%s_%s.npy' % (args.save_dir, args.val_save_name,number), embds)
        np.save('exp/%s/%s_da_%s.npy' % (args.save_dir, args.val_save_name,number), embds_da)

        da_classifier = domain_classifier(args.embd_dim, args.total_methods).cuda()
        da_classifier.load_state_dict(torch.load('exp/%s/%s_%s.pkl' % (args.save_dir, args.model_name ,number))['da_classifier'])
        da_classifier.eval()
        embds_cla = {}
        with torch.no_grad():
            for key in embds_da.keys():
                outputs_da = da_classifier(torch.from_numpy(embds_da[key]).cuda())
                embds_cla[key] = outputs_da.cpu().numpy()
        np.save('exp/%s/%s_cla_%s.npy' % (args.save_dir, args.val_save_name,number), embds_cla)
    
    else:
        embds = np.load('exp/%s/%s_%s.npy' % (args.save_dir, args.val_save_name, number), allow_pickle=True).item()
        embds_da = np.load('exp/%s/%s_da_%s.npy' % (args.save_dir, args.val_save_name, number), allow_pickle=True).item()
        acc_all = []
        embds_cla = np.load('exp/%s/%s_cla_%s.npy' % (args.save_dir, args.val_save_name, number), allow_pickle=True).item
        for key in embds_cla.keys():
            outputs_da = embds_cla[key]
            acc = accuracy(outputs_da, torch.tensor([args.method_idx]).cuda(), topk=(1,))
            acc_all.append(acc[0].cpu().numpy())
    # 计算eer
    if args.scoring:
        f = open('exp/%s/%s_scoring_dev_method.txt' % (args.save_dir, args.val_save_name), 'a' )
        eer,threshold ,cost,_ = get_eer(embds, trial_file='data/%s/%s' % (args.val_data_name, args.trials))
        f.write('Model:%s  %s_%s.pkl\n' %(args.save_dir, args.model_name, number))
        f.write('EER : %.3f%% Th : %.3f mDCT : %.5f\t\n'%(eer*100, threshold, cost))
        f.write('ACC: %.5f\t\n'%(sum(acc_all)/len(acc_all)))
        f.flush()

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

if __name__ == '__main__':
    number_list = args.model_num.split(',')
    for num in number_list:
        main(num)