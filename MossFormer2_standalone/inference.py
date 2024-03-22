'''This is a sample inference script to demonstrate how to run inference on the model for a single .wav file

Authors
* Jia Qi Yip 2024
'''
import torch
from model.mossformer2 import Mossformer2Wrapper

model_configs = ["mossformer2-librimix-2spk", "mossformer2-wsj0mix-3spk", "mossformer2-whamr-2spk"]

for mc in model_configs:
    model = Mossformer2Wrapper.from_pretrained(f'alibabasglab/{mc}')
    model.inference(f'./test_samples/{mc}/item0_mix.wav',f'./test_samples/{mc}/model_output')