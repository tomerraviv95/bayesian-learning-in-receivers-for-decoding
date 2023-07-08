import numpy as np
import torch
from torch import nn

from dir_definitions import ECC_MATRICES_DIR
from python_code import conf
from python_code.decoders.bp_nn import InputLayer, OddLayer, EvenLayer, OutputLayer
from python_code.utils.constants import CLIPPING_VAL
from python_code.utils.python_utils import get_code_pcm_and_gm

ITERATIONS = 5


class BPDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.odd_llr_mask_only = True
        self.even_mask_only = True
        self.output_mask_only = False
        self.iteration_num = ITERATIONS
        self._code_bits = conf.code_bits
        self._message_bits = conf.message_bits
        self.code_pcm, self.code_gm = get_code_pcm_and_gm(conf.code_bits, conf.message_bits,
                                                          ECC_MATRICES_DIR, conf.code_type)
        self.neurons = int(np.sum(self.code_pcm))
        self.lr = None
        self.initialize_layers()

    def __str__(self):
        return 'BP Decoder'

    def initialize_layers(self):
        self.input_layer = InputLayer(input_output_layer_size=self._code_bits, neurons=self.neurons,
                                      code_pcm=self.code_pcm, clip_tanh=CLIPPING_VAL,
                                      bits_num=self._code_bits)
        self.odd_layer = OddLayer(clip_tanh=CLIPPING_VAL,
                                  input_output_layer_size=self._code_bits,
                                  neurons=self.neurons,
                                  code_pcm=self.code_pcm)
        self.even_layer = EvenLayer(CLIPPING_VAL, self.neurons, self.code_pcm)
        self.output_layer = OutputLayer(neurons=self.neurons,
                                        input_output_layer_size=self._code_bits,
                                        code_pcm=self.code_pcm)

    def forward(self, x):
        """
        compute forward pass in the network
        :param x: [batch_size,N]
        :return: decoded word [batch_size,N]
        """
        # initialize parameters
        output_list = [0] * self.iteration_num

        # equation 1 and 2 from "Learning To Decode ..", i==1,2 (iteration 1)
        even_output = self.input_layer.forward(x)
        output_list[0] = x + self.output_layer.forward(even_output, mask_only=self.output_layer)

        # now start iterating through all hidden layers i>2 (iteration 2 - Imax)
        for i in range(0, self.iteration_num - 1):
            # odd - variables to check
            odd_output = self.odd_layer.forward(even_output, x, llr_mask_only=self.odd_llr_mask_only)
            # even - check to variables
            even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
            # output layer
            output = x + self.output_layer.forward(even_output, mask_only=self.output_mask_only)
            output_list[i + 1] = output.clone()

        decoded_words = torch.round(torch.sigmoid(-output_list[-1]))
        return decoded_words
