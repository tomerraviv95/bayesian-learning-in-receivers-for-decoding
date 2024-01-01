import torch
from torch.nn import BCEWithLogitsLoss

from python_code import DEVICE
from python_code.decoders.bp_nn import InputLayer, EvenLayer, OutputLayer
from python_code.decoders.decoder_trainer import DecoderTrainer
from python_code.decoders.modular_bayesian_wbp.bayesian_bp_nn import BayesianOddLayer
from python_code.utils.bayesian_utils import LossVariable
from python_code.utils.constants import HALF, CLIPPING_VAL, Phase

EPOCHS = 100
BATCH_SIZE = 20
LR = 1e-3


class ModularBayesianWBPDecoder(DecoderTrainer):
    def __init__(self):
        super().__init__()
        self.output_mask_only = True
        self.ensemble_num = 3
        self.kl_beta = 1e-4
        self.arm_beta = 1
        self.initialize_layers()

    def __str__(self):
        return 'MB-WBP'

    def initialize_layers(self):
        self.input_layer = InputLayer(input_output_layer_size=self._code_bits, neurons=self.neurons,
                                      code_pcm=self.code_pcm, clip_tanh=CLIPPING_VAL,
                                      bits_num=self._code_bits)
        self.odd_layer = BayesianOddLayer(clip_tanh=CLIPPING_VAL,
                                          input_output_layer_size=self._code_bits,
                                          neurons=self.neurons,
                                          code_pcm=self.code_pcm)
        self.even_layer = EvenLayer(CLIPPING_VAL, self.neurons, self.code_pcm)
        self.output_layer = OutputLayer(neurons=self.neurons,
                                        input_output_layer_size=self._code_bits,
                                        code_pcm=self.code_pcm)

    def calc_loss(self, est: LossVariable, tx: torch.Tensor, output: torch.Tensor, output_tilde: torch.Tensor):
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        loss = self.criterion(input=-output, target=tx)
        # ARM Loss
        loss_term_arm_original = self.criterion_arm(input=-output, target=tx)
        loss_term_arm_tilde = self.criterion_arm(input=-output_tilde, target=tx)
        arm_delta = (loss_term_arm_tilde - loss_term_arm_original)
        grad_logit = torch.matmul(arm_delta, self.odd_layer.w_skipconn2even_mask).unsqueeze(-1) * (est.u - HALF).unsqueeze(0)
        grad_logit[grad_logit < 0] *= -1
        arm_loss = grad_logit * est.dropout_logits.unsqueeze(0)
        arm_loss = self.arm_beta * torch.mean(arm_loss)
        # KL Loss
        kl_term = self.kl_beta * est.kl_term
        loss += arm_loss + kl_term
        return loss

    def single_training(self, tx: torch.Tensor, rx: torch.Tensor):
        # initialize
        self.deep_learning_setup(LR)
        new = list(filter(lambda p: p.requires_grad, self.parameters()))
        self.criterion_arm = BCEWithLogitsLoss(reduction='none').to(DEVICE)
        even_output = self.input_layer.forward(rx)
        for i in range(self.iteration_num - 1):
            for e in range(EPOCHS):
                idx = torch.randperm(tx.shape[0])[:BATCH_SIZE]
                cur_tx, cur_even_output, cur_rx = tx[idx], even_output[idx], rx[idx]
                # odd - variables to check
                est = self.odd_layer.forward(cur_even_output, cur_rx, mask_only=self.odd_llr_mask_only, phase=Phase.TRAIN)
                # even - check to variables
                cur_even_output = self.even_layer.forward(est.priors, mask_only=self.even_mask_only)
                # compute output
                output = cur_rx + self.output_layer.forward(cur_even_output, mask_only=self.output_mask_only)
                # tilde calculation
                even_output_tilde = self.even_layer.forward(est.arm_tilde, mask_only=self.even_mask_only)
                output_tilde = cur_rx + self.output_layer.forward(even_output_tilde, mask_only=self.output_mask_only)
                # calculate loss and backtrack
                loss = self.calc_loss(tx=cur_tx, output=output, output_tilde=output_tilde, est=est)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                total_even_output = 0
                for _ in range(self.ensemble_num):
                    # odd - variables to check
                    odd_output = self.odd_layer.forward(even_output, rx, mask_only=self.odd_llr_mask_only)
                    # even - check to variables
                    cur_even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
                    total_even_output += cur_even_output
                even_output = total_even_output / self.ensemble_num

    def forward(self, x):
        """
        compute forward pass in the network
        :param x: [batch_size,N]
        :return: decoded word [batch_size,N]
        """
        # initialize parameters
        output_list = [0] * (self.iteration_num)

        # equation 1 and 2 from "Learning To Decode ..", i==1,2 (iteration 1)
        even_output = self.input_layer.forward(x)
        output_list[0] = x + self.output_layer.forward(even_output, mask_only=self.output_layer)

        # now start iterating through all hidden layers i>2 (iteration 2 - Imax)
        for i in range(self.iteration_num - 1):
            total_even_output = 0
            for _ in range(self.ensemble_num):
                # odd - variables to check
                odd_output = self.odd_layer.forward(even_output, x, mask_only=self.odd_llr_mask_only)
                # even - check to variables
                even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
                total_even_output += even_output
            avg_even_output = total_even_output / self.ensemble_num
            # output layer
            output = x + self.output_layer.forward(avg_even_output, mask_only=self.output_mask_only)
            output_list[i + 1] = output.clone()

        decoded_words = torch.round(torch.sigmoid(-output_list[-1]))
        return decoded_words
