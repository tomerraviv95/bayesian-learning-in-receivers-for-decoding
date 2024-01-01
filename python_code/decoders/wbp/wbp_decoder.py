import torch

from python_code.decoders.bp_nn import InputLayer, OddLayer, EvenLayer, OutputLayer
from python_code.decoders.decoder_trainer import DecoderTrainer, LR, EPOCHS
from python_code.utils.constants import CLIPPING_VAL, Phase


class WBPDecoder(DecoderTrainer):
    def __init__(self):
        super().__init__()
        self.type = "FC"
        self.initialize_layers()

    def __str__(self):
        return 'F-WBP'

    def initialize_layers(self):
        if self.type == "FC":
            self.input_layer = InputLayer(input_output_layer_size=self._code_bits, neurons=self.neurons,
                                          code_pcm=self.code_pcm, clip_tanh=CLIPPING_VAL, bits_num=self._code_bits)
            self.odd_layers = [OddLayer(clip_tanh=CLIPPING_VAL, input_output_layer_size=self._code_bits,
                                        neurons=self.neurons, code_pcm=self.code_pcm) for _ in
                               range(self.iteration_num - 1)]
            self.even_layer = EvenLayer(CLIPPING_VAL, self.neurons, self.code_pcm)
            self.output_layer = OutputLayer(neurons=self.neurons, input_output_layer_size=self._code_bits,
                                            code_pcm=self.code_pcm)
        else:
            self.input_layer = InputLayer(input_output_layer_size=self._code_bits, neurons=self.neurons,
                                          code_pcm=self.code_pcm, clip_tanh=CLIPPING_VAL, bits_num=self._code_bits)
            self.odd_layer = OddLayer(clip_tanh=CLIPPING_VAL, input_output_layer_size=self._code_bits,
                                      neurons=self.neurons, code_pcm=self.code_pcm)
            self.even_layer = EvenLayer(CLIPPING_VAL, self.neurons, self.code_pcm)
            self.output_layer = OutputLayer(neurons=self.neurons, input_output_layer_size=self._code_bits,
                                            code_pcm=self.code_pcm)

    def calc_loss(self, decision, labels):
        loss = self.criterion(input=-decision[-1], target=labels)
        if self.multi_loss_flag:
            for iteration in range(self.iteration_num - 1):
                current_loss = self.criterion(input=-decision[iteration], target=labels)
                loss += current_loss
        return loss

    def single_training(self, tx: torch.Tensor, rx: torch.Tensor):
        self.deep_learning_setup(LR)
        for _ in range(EPOCHS):
            output_list = self.forward(rx, mode=Phase.TRAIN)
            # calculate loss
            loss = self.calc_loss(decision=output_list[-self.iteration_num:], labels=tx)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def forward(self, x, mode: Phase = Phase.TEST):
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
        for i in range(self.iteration_num - 1):
            # odd - variables to check
            odd_output = self.odd_layers[i].forward(even_output, x, llr_mask_only=self.odd_llr_mask_only)
            # even - check to variables
            even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
            # output layer
            output = x + self.output_layer.forward(even_output, mask_only=self.output_mask_only)
            output_list[i + 1] = output.clone()

        if mode == Phase.TEST:
            decoded_words = torch.round(torch.sigmoid(-output_list[-1]))
            return decoded_words
        return output_list
