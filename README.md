# bayesian-learning-for-receivers

*”A Bayesian version will usually make things better.”* 

-- [Andrew Gelman, Columbia University](http://www.stat.columbia.edu/~gelman/book/gelman_quotes.pdf). 

# Bayesian Learning for Deep Receivers

Python repository for a submitted journal paper "Uncertainty-Aware and Reliable Neural MIMO Receivers via Modular Bayesian Deep Learning".

Please cite [our paper](https://arxiv.org/pdf/2302.02436.pdf), if the code is used for publishing research.

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [python_code](#python_code)
    + [datasets](#datasets)
    + [decoders](#decoders)
    + [detectors](#detectors)
    + [plotters](#plotters)
    + [utils](#utils)
  * [resources](#resources)
  * [dir_definitions](#dir_definitions)
- [Execution](#execution)
  * [Environment Installation](#environment-installation)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# Introduction

This repository implements the proposed model-based Bayesian framework for [DeepSIC](https://arxiv.org/abs/2002.03214) and for the weighted belief propogation [WBP](https://arxiv.org/abs/1607.04793). The explanation on how the simulations were obtained follows below.

# Folders Structure

## python_code 

The python simulations of the simplified communication chain: bits generation, encoding, modulation, transmission through noisy channel, detection and finally decoding. The transmission is done over a sequence of blocks composed of pilot and information parts as in the paper. The main script for running this pipeline is the evaluator.py file, using the configuration defined in the config.yaml file. It has the following parameters:

### config.yaml

* seed - random integer used for reproducibility.

* block_length - coherence block bits, total size of pilot + data. type: int.

* pilots_length - number of bits in the pilots part. Other bits in the block are information bits. type: int.

* blocks_num - number of block transmitted sequentailly. type: int.

* n_user - number of users in the uplink scenario. type: int.

* n_ant - number of antennas on the BS in the uplink scenario. type: int.

* channel_model - the channel model, in the set ['Synthetic','Cost2100']. 

  
* fading_in_channel - whether to apply fading. Only relevant for the Synthetic channel, Cost2100 has built in fading as a default. Boolean value.

* snr - signal-to-noise value in dB (float).

* modulation_type - which modulation to use, in the set of ['BPSK','QPSK','EightPSK'].
  
* detector_type - which detector to use, in the set of ['seq_model','model_based_bayesian','bayesian','end_to_end'].
  
* code_type - which code to use, in the set of ['BCH','POLAR']. I've only used polar codes for the paper.
  
* code_bits - number of code bits, integer. The n value in the polar (n,k) definition.
  
* message_bits - number of message bits, integer. The k value in the polar (n,k) definition.
  
* decoder_type - which decoder to use, in the set of ['bp','wbp','modular_bayesian_wbp','bayesian_wbp'].
  
* hidden_base_size - number of hidden neurons in the DeepSIC architecture. Integer.

* deepsic_iterations - number of iterations in the DeepSIC model.

### datasets 

channel_dataset is the main class for the creation of the dataset, composed of tuples of (tx bits, tx symbols, received channel values). Handles all the data generation part up to the channel output. 

### decoders

Includes the models of the vanilla BP (bp directory), WBP (wbp directory), model-based Bayesian WBP (modular_bayesian_wbp directory) and the Bayesian WBP (bayesian_wbp) directory. See each file for further documentation. The common architecture is in bp_nn and bp_nn_weights. The file decoder_trainer holds the basic functions for training and evaluating the decoder. 

### detectors 

Includes the models of the vanilla deepsic trained either in end to end fashion, or the more conventional sequentially (see the original paper on DeepSIC for more details).Model-based Bayesian DeepSIC is in the modular_bayesian_deepsic directory and the Bayesian DeepSIC is in bayesian_deepsic directory. See each file inside for further documentation. The common architecture is in the file deepsic_detector. The files detector_trainer and deepsic_trainer hold the basic functions for training and evaluating the detector. 

### plotters

The main script is plotter_main.py, and it is used to plot the figures in the paper including all ser/ber versus snr, and reliability diagrams.

### utils

Extra utils for many different things: 

* bayesian utils - for the calculation of the LBD loss.
  
* coding_utils.py - some utility functions for the coding part.

* config_singleton - holds the singleton definition of the config yaml.
  
* constants - some global constants.
  
* metrics - calculating accuracy, confidence, ECE and sampling frequency for reliability diagrams.

* probs utils - for generate symbols from states; symbols from probs and vice versa.

* python utils - saving and loading pkls.

## resources

Keeps the configs runs files for creating the paper's figures.

## dir_definitions 

Definitions of relative directories.

# Execution

To execute the code, first download and install Git, Anaconda and PyCharm.

Then install the environment, follow the installation setup below. 

At last, open PyCharm in the root directory. You may run either the trainers or one of the plotters.

This code was simulated with GeForce RTX 3060 with driver version 516.94 and CUDA 11.6. 

## Environment Installation

1. Open git bash and cd to a working directory of you choice.

2. Clone this repository to your local machine.

3. Open Anaconda prompt and navigate to the cloned repository.

4. Run the command "conda env create -f environment.yml". This should install the required python environment.

5. Open the cloned directory using PyCharm.

6. After the project has been opened in PyCharm, go to settings, File -> Settings... (or CTRL ALT S)

7. In the opened window open the tab Project -> Project Interpreter

8. In the new window, click on the cog icon and then on Add...

9. In the add python interpreter window, click on the Conda Environment tab

10. Select Existing environment and navigate to where the python.exe executable of the deep_ensemble environment is installed under the interpreter setting

  - For windows its usually found at C:\users\<username>\anaconda3\envs\environment\python.exe)

  - For linux its usually found at /home/<username>/anaconda3
  
11. Click OK

12. Done!
