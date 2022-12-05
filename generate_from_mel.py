from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator
import pickle
from tqdm import tqdm
import soundfile as sf

checkpoint_path = 'g_03280000'
device = 'cuda:1'
config_file = 'config_v1.json'
output_dir = '/home/sile/autovc/fake_audio'
mel_file = '/home/sile/autovc/spmel_test_old_emb/mel_results.pkl'

checkpoint_dict = torch.load(checkpoint_path, map_location=device)
json_config = json.loads(open(config_file).read())
h = AttrDict(json_config)

mel_val = pickle.load(open(mel_file, 'rb'))

generator = Generator(h).to(device)
generator.load_state_dict(checkpoint_dict['generator'])
generator.eval()
generator.remove_weight_norm()
with torch.no_grad():
    for spect in tqdm(mel_val):
        name = spect[0]
        c = spect[1]
        c = torch.FloatTensor(c).to(device).transpose(0, 1).unsqueeze(0)
        waveform = generator(c)
        waveform = waveform.squeeze().detach().cpu().numpy()
        if not os.path.exists(os.path.join(output_dir, name.split('/')[0])):
            os.makedirs(os.path.join(output_dir, name.split('/')[0]))
        sf.write(os.path.join(output_dir, name)+'.wav', waveform, samplerate=16000)
        