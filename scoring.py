#! /usr/bin/env python3
import os, argparse, numpy as np
import torch, torch.nn as nn, modules.model_spk as models
from torch.utils.data import DataLoader
from dataset import WavDataset, logFbankCal
from tools.util import compute_eer
from scipy import spatial
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description='Speaker Embedding Extraction')
parser.add_argument('--save_dir', type=str)
# validation dataset
parser.add_argument('--val_data_name', default='vox_test', type=str)
parser.add_argument('--vc_method', default='vox_test', type=str)
parser.add_argument('--val_save_name', default='vox_test', type=str)
parser.add_argument('--model_name', default='model', type=str)
parser.add_argument('--trials', default='', type=str)
# acoustic feature
parser.add_argument('--fs', default=16000, type=int)
parser.add_argument('--fft', default=512, type=int)
parser.add_argument('--mels', default=80, type=int)
parser.add_argument('--win_len', default=0.025, type=float)
parser.add_argument('--hop_len', default=0.01, type=float)
# model backbone
parser.add_argument('--model', default='ConformerMFA', type=str)
parser.add_argument('--model_cfg', default='configs/conformer_small2.yaml', type=str)
parser.add_argument('--model_num', default='100', type=str)
parser.add_argument('--embd_dim', default=256, type=int)
parser.add_argument('--dropout', default=0, type=float)
# others
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--verbose', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--scoring', default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--onlyscore', default=False, type=lambda x: (str(x).lower() == 'true'))

args = parser.parse_args()

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
        val_dataset = WavDataset([line.split() for line in open('data/%s/%s.scp' % (args.val_data_name, args.vc_method))], fs=args.fs)
        val_dataloader = DataLoader(val_dataset, num_workers=10, shuffle=False, batch_size=1)

        # models
        cfg = OmegaConf.load(args.model_cfg)
        cfg['encoder']['mask_prob'] = 0
        model = getattr(models, args.model)(cfg['encoder'], args.embd_dim, dropout=args.dropout).cuda()
        ckp = torch.load('exp/%s/%s_%s.pkl' % (args.save_dir, args.model_name ,number))['model']
        model.load_state_dict(ckp)
        model = nn.DataParallel(model)
        model.eval()

        embds = {}
        with torch.no_grad():
            for j, (feat, utt) in enumerate(val_dataloader):
                feat = feat.cuda()
                feat_len = torch.LongTensor([feat.shape[1]]).cuda()
                embds[utt[0]] = model(featCal(feat), feat_len).cpu().numpy()

        np.save('exp/%s/%s_%s.npy' % (args.save_dir, args.val_save_name,number), embds)
    else:
        embds = np.load('exp/%s/%s_%s.npy' % (args.save_dir, args.val_save_name, number), allow_pickle=True).item()
    # 计算eer
    if args.scoring:
        f = open('exp/%s/%s_scoring_dev.txt' % (args.save_dir, args.val_save_name), 'a' )
        eer,threshold ,cost,_ = get_eer(embds, trial_file='data/%s/%s' % (args.val_data_name, args.trials))

        f.write('Model:%s  %s_%s.pkl\n' %(args.save_dir, args.model_name, number))
        f.write('EER : %.3f%% Th : %.3f mDCT : %.5f\t\n'%(eer*100, threshold, cost))
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