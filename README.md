# MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation

This repository provides the processed samples and information for **MossFormer2** monaural speech separation model. MossFormer2 is an extended model from <a href="https://github.com/alibabasglab/MossFormer">MossFormer</a>. It can be retrained and evaluated based on <a href="https://modelscope.cn/models/damo/speech_mossformer2_separation_temporal_8k/summary">ModelScope open source platform</a>. The users can either go to the <a href="https://modelscope.cn/models/damo/speech_mossformer2_separation_temporal_8k/summary">ModelScope website</a> or follow the steps given below to downloand and install the full pytorch version of MossFormer2 program. MossFormer2 was proposed in the paper <a href="https://arxiv.org/abs/2312.11825">MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation</a>.  

## Model Description

To effectively solve the indirect elemental interactions across chunks in the dual-path architecture, we propose a gated single-head transformer architecture with convolution-augmented joint self-attentions, named MossFormer (Monaural speech separation Transformer). MossFormer employs a joint local and global self-attention architecture that simultaneously performs a full-computation self- attention on local chunks and a linearised low-cost self-attention over the full sequence. The joint attention enables MossFormer model full-sequence elemental interaction directly. In addition, we employ a powerful attentive gating mechanism with simplified single-head self-attentions. Besides the attentive long-range modelling, we also augment MossFormer with convolutions for the position-wise local pattern modelling.


The MossFormer architecture comprises of a convolutional encoder-decoder structure and a masking net. The encoder-decoder structure responds for feature extraction and waveform reconstruction. The masking net maps the encoded output to a group of masks. The MossFormer block  consists of four convolution modules, scale and offset operations, a joint local and global single- head self-attention, and three element-wise gating operations.

![fig_mossformer](https://user-images.githubusercontent.com/62317780/220862493-30637387-8da2-4538-8e83-604fe4c9764c.png)

MossFormer significantly outperforms the previous models and achieves the state-of-the-art results on WSJ0-2/3mix and WHAM!/WHAMR! benchmarks. Our model achieves the SI-SDRi upper bound of 21.2 dB on WSJ0-3mix and only 0.3 dB below the upper bound of 23.1 dB on WSJ0-2mix.

![table1_mossformer](https://user-images.githubusercontent.com/62317780/220861391-dd6bf2a1-0033-443d-bee5-e2241a929462.png)

## Installation

After installing <a href="https://github.com/modelscope/modelscope">ModelScope</a>, you can use *speech_mossformer_separation_temporal_8k* for inference. In order to facilitate the usage, the pipeline adds wav file processing logics before and after model processing, which can directly read a WAV file and save the output result in the specified WAV file. The model pipeline takes in a single-channel WAV file sampled at 8000Hz, containing mixed speech of two people, and outputs two separated single-channel audio files.

#### Environment Preparation

This model supports Linux, Windows, and MacOS platforms.

This model relies on the open-source library <a href="https://github.com/speechbrain/speechbrain"> SpeechBrain </a>. Due to its strict dependency on the PyTorch version, it has not been included in the default dependencies of ModelScope and needs to be manually installed by the user.

```
#If your PyTorch version is >=1.10, install the latest version
pip install speechbrain

#If your PyTorch version is <1.10 and >=1.7, you can specify the version to install as follows
pip install speechbrain==0.5.12
```

The pipeline of this model uses the third-party library *SoundFile* to process wav files. On the Linux system, users need to manually install the underlying dependency library *libsndfile* of *SoundFile*. On Windows and MacOS, it will be installed automatically without user operation. For detailed information, please refer to the official website of <a href=https://github.com/bastibe/python-soundfile#installation>*SoundFile*</a>. Taking the Ubuntu system as an example, the user needs to execute the following command:

```
sudo apt-get update
sudo apt-get install libsndfile1
```

####  Code Example
```python
import numpy
import soundfile as sf
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# input can be a URL or a local path
input = 'https://modelscope.cn/api/v1/models/damo/speech_mossformer_separation_temporal_8k/repo?Revision=master&FilePath=examples/mix_speech1.wav'
separation = pipeline(
   Tasks.speech_separation,
   model='damo/speech_mossformer_separation_temporal_8k')
result = separation(input)
for i, signal in enumerate(result['output_pcm_list']):
    save_file = f'output_spk{i}.wav'
    sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16), 8000)
```

## Model Training

The Notebook environment provided by the official ModelScope website has already installed all the dependencies and can start training directly. If you want to train on your own device, you can refer to the environment preparation steps in the previous section. After the environment is set up, it is recommended to run the inference sample code to verify that the model can work properly.

The following is an example code for training, where the work_dir can be replaced with the desired path. The training logs will be saved in work_dir/log.txt, and the model parameters and other data during training will be saved in work_dir/save/CKPT+timestamp path. Each epoch of data training is the default of 120 epochs, and on a machine with a hardware configuration of a 20-core CPU and a V100 GPU, it takes about 10 days.

```
import os

from datasets import load_dataset

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.preprocessors.audio import AudioBrainPreprocessor
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import to_segment

work_dir = './train_dir'
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

train_dataset = MsDataset.load(
        'Libri2Mix_8k', split='train').to_torch_dataset(preprocessors=[
        AudioBrainPreprocessor(takes='mix_wav:FILE', provides='mix_sig'),
        AudioBrainPreprocessor(takes='s1_wav:FILE', provides='s1_sig'),
        AudioBrainPreprocessor(takes='s2_wav:FILE', provides='s2_sig')
    ],
    to_tensor=False)
eval_dataset = MsDataset.load(
        'Libri2Mix_8k', split='validation').to_torch_dataset(preprocessors=[
        AudioBrainPreprocessor(takes='mix_wav:FILE', provides='mix_sig'),
        AudioBrainPreprocessor(takes='s1_wav:FILE', provides='s1_sig'),
        AudioBrainPreprocessor(takes='s2_wav:FILE', provides='s2_sig')
    ],
    to_tensor=False)
kwargs = dict(
    model='damo/speech_mossformer_separation_temporal_8k',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=work_dir)
trainer = build_trainer(
    Trainers.speech_separation, default_args=kwargs)
trainer.train()
```

## Model Evaluation

The following is the model evaluation code, where work_dir is the working directory, and the model to be evaluated must be placed in the work_dir/save/CKPT+timestamp directory. The program will search for the best model in the directory and load it automatically.

```
import os

from datasets import load_dataset

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.preprocessors.audio import AudioBrainPreprocessor
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import to_segment

work_dir = './train_dir'
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

train_dataset = None
eval_dataset = MsDataset.load(
        'Libri2Mix_8k', split='test').to_torch_dataset(preprocessors=[
        AudioBrainPreprocessor(takes='mix_wav:FILE', provides='mix_sig'),
        AudioBrainPreprocessor(takes='s1_wav:FILE', provides='s1_sig'),
        AudioBrainPreprocessor(takes='s2_wav:FILE', provides='s2_sig')
    ],
    to_tensor=False)
kwargs = dict(
    model='damo/speech_mossformer_separation_temporal_8k',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=work_dir)
trainer = build_trainer(
    Trainers.speech_separation, default_args=kwargs)
trainer.model.load_check_point(device=trainer.device)
print(trainer.evaluate(None))
```

## Performance limitation of this model release
This released model is a demonstrated only model that was trained on clean WSJ0-2mix dataset. Due to the small amount of this training dataset, the released model maybe not perform as expected for other testing dataset, espectially, the noisy and reverberant mixtures. We will update to a more general model support realistic recordings later.

For more details, please refer to the related paper below:

```
@INPROCEEDINGS{9747578,
  author={Zhao, Shengkui and Ma, Bin},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={MossFormer: Pushing the Performance Limit of Monaural Speech Separation using Gated Single-head Transformer with Convolution-augmented Joint Self-Attentions}, 
  year={2023},
  }
```
