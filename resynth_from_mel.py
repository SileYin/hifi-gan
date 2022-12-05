from __future__ import absolute_import, division, print_function, unicode_literals
from genericpath import exists

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

checkpoint_path = './g_03280000'
device = 'cuda:1'
config_file = 'config_v1.json'
output_dir = '/home/sile/autovc/resynthed_test'
rootDir = '/home/sile/autovc/spmel_pitch_shift'

checkpoint_dict = torch.load(checkpoint_path, map_location=device)
json_config = json.loads(open(config_file).read())
h = AttrDict(json_config)

dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

generator = Generator(h).to(device)
generator.load_state_dict(checkpoint_dict['generator'])
generator.eval()
generator.remove_weight_norm()
with torch.no_grad():
    for speaker in sorted(subdirList):
        print('Processing speaker: %s' % speaker)
        _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
        os.makedirs(os.path.join(output_dir, speaker), exist_ok=1)
        for file in tqdm(fileList):
            x = np.load(os.path.join(dirName, speaker, file))
            x = torch.FloatTensor(x).to(device).transpose(0, 1).unsqueeze(0)
            waveform = generator(x)
            waveform = waveform.squeeze().detach().cpu().numpy()
            sf.write(os.path.join(output_dir, speaker, file.split('.')[0])+'.wav', waveform, samplerate=16000)

        