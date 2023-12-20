# MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation

This repository provides the processed samples and information for **MossFormer2** monaural speech separation model. MossFormer2 is an extended model from <a href="https://github.com/alibabasglab/MossFormer">MossFormer</a>. It can be retrained and evaluated based on <a href="https://modelscope.cn/models/damo/speech_mossformer2_separation_temporal_8k/summary">ModelScope open source platform</a>. The users can either go to the <a href="https://modelscope.cn/models/damo/speech_mossformer2_separation_temporal_8k/summary">ModelScope website</a> or follow the steps given below to downloand and install the full pytorch version of MossFormer2 program. MossFormer2 was proposed in the paper <a href="https://arxiv.org/abs/2312.11825">MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation</a>.  

## Model Description

Our previously proposed MossFormer has achieved promising performance in monaural speech separation. However, it predominantly adopts a self-attention-based MossFormer module, which tends to emphasize longer-range, coarser-scale dependencies, with a deficiency in effectively modelling finer-scale recurrent patterns. In MossFormer2, we introduce a novel hybrid model that provides the capabilities to model both long-range, coarse-scale dependencies and fine-scale recurrent patterns by integrating a recurrent module into the MossFormer framework. MossFormer module concurrently executes full-computation self-attention on non-overlapping local segments and employs a linearized, resource-efficient self-attention mechanism over the entire sequence. The recurrent module complements the MossFormer’s capabilities in modelling finer-scale recurrent patterns, a crucial aspect of the speech separation task.

![github_fig1](https://github.com/alibabasglab/MossFormer2/assets/62317780/e69fb5df-4d7f-4572-88e6-8c393dd8e99d)


Instead of applying the recurrent neural networks (RNNs) that use traditional recurrent connections, we present a recurrent module based on a feedforward sequential memory network (FSMN), which is considered "RNN-free" recurrent network due to the ability to capture recurrent patterns without using recurrent connections. Our recurrent module mainly comprises an enhanced dilated FSMN block by using gated convolutional units (GCU) and dense connections. In addition, a bottleneck layer and an output layer are also added for controlling information flow. The recurrent module relies on linear projections and convolutions for seamless, parallel processing of the entire sequence. 

![github_fig2](https://github.com/alibabasglab/MossFormer2/assets/62317780/7273174d-01aa-4cc5-9a67-1fa2e8f7ac2e)


The integrated MossFormer2 hybrid model demonstrates remarkable enhancements over MossFormer and surpasses other state-of-the-art methods in WSJ0-2/3mix, Libri2Mix, and WHAM!/WHAMR! benchmarks.

![github_tab1](https://github.com/alibabasglab/MossFormer2/assets/62317780/82ce529f-da29-405a-80e7-ecd215cd2eff)

![github_tab2](https://github.com/alibabasglab/MossFormer2/assets/62317780/206492da-cf1a-4a2c-a316-bf15ed4a34e8)


## Installation

After installing <a href="https://github.com/modelscope/modelscope">ModelScope</a>, you can use *speech_mossformer2_separation_temporal_8k* for inference. In order to facilitate the usage, the pipeline adds wav file processing logics before and after model processing, which can directly read a WAV file and save the output result in the specified WAV file. The model pipeline takes in a single-channel WAV file sampled at 8000Hz, containing mixed speech of two people, and outputs two separated single-channel audio files.

#### Environment Preparation

This model supports Linux, Windows, and MacOS platforms.

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
input = 'https://modelscope.cn/api/v1/models/damo/speech_mossformer2_separation_temporal_8k/repo?Revision=master&FilePath=examples/mix_speech1.wav'
separation = pipeline(
   Tasks.speech_separation,
   model='damo/speech_mossformer2_separation_temporal_8k')
result = separation(input)
for i, signal in enumerate(result['output_pcm_list']):
    save_file = f'output_spk{i}.wav'
    sf.write(save_file, numpy.frombuffer(signal, dtype=numpy.int16), 8000)
```

For more details, please refer to the related paper below:

```
@misc{zhao2023mossformer2,
      title={MossFormer2: Combining Transformer and RNN-Free Recurrent Network for Enhanced Time-Domain Monaural Speech Separation}, 
      author={Shengkui Zhao and Yukun Ma and Chongjia Ni and Chong Zhang and Hao Wang and Trung Hieu Nguyen and Kun Zhou and Jiaqi Yip and Dianwen Ng and Bin Ma},
      year={2023},
      eprint={2312.11825},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
