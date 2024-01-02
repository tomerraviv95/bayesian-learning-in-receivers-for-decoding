import torch
from torch.nn import BCEWithLogitsLoss, ModuleList

from python_code import DEVICE
from python_code.decoders.bp_nn import InputLayer, EvenLayer, OddLayer
from python_code.decoders.decoder_trainer import DecoderTrainer, LR, EPOCHS
from python_code.decoders.modular_bayesian_wbp.bayesian_bp_nn import BayesianOutputLayer
from python_code.utils.bayesian_utils import LossVariable
from python_code.utils.constants import CLIPPING_VAL, Phase, HALF


class ModularBayesianWBPDecoder(DecoderTrainer):
    def __init__(self):
        super().__init__()
        self.ensemble_num = 3
        self.kl_beta = 1e-4
        self.arm_beta = 1
        self.initialize_layers()

    def __str__(self):
        return 'MB-WBP'

    def initialize_layers(self):
        self.input_layer = InputLayer(input_output_layer_size=self._code_bits, neurons=self.neurons,
                                      code_pcm=self.code_pcm, clip_tanh=CLIPPING_VAL, bits_num=self._code_bits)
        self.odd_layer = OddLayer(clip_tanh=CLIPPING_VAL, input_output_layer_size=self._code_bits,
                                  neurons=self.neurons, code_pcm=self.code_pcm)
        self.even_layer = EvenLayer(CLIPPING_VAL, self.neurons, self.code_pcm)
        self.output_layers = ModuleList(
            [BayesianOutputLayer(neurons=self.neurons, input_output_layer_size=self._code_bits,
                                 code_pcm=self.code_pcm, ensemble_num=self.ensemble_num) for _ in
             range(self.iteration_num - 1)])

    def calc_loss(self, est: LossVariable, tx: torch.Tensor, output: torch.Tensor, output_tilde: torch.Tensor):
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        loss = self.criterion(input=-output, target=tx)
        # ARM Loss
        loss_term_arm_original = self.criterion_arm(input=-output, target=tx)
        loss_term_arm_tilde = self.criterion_arm(input=-output_tilde, target=tx)
        arm_delta = torch.mean(loss_term_arm_tilde - loss_term_arm_original, dim=1)
        grad_logit = arm_delta.unsqueeze(-1) * (est.u - HALF)
        grad_logit[grad_logit < 0] *= -1
        arm_loss = grad_logit * est.dropout_logits.unsqueeze(0)
        arm_loss = self.arm_beta * torch.mean(arm_loss)
        # KL Loss
        kl_term = self.kl_beta * est.kl_term
        loss += arm_loss + kl_term
        return loss

    def single_training(self, tx: torch.Tensor, rx: torch.Tensor):
        self.deep_learning_setup(LR)
        self.criterion_arm = BCEWithLogitsLoss(reduction='none').to(DEVICE)
        even_output = self.input_layer.forward(rx)
        for i in range(self.iteration_num - 1):
            print(f"Iterations Number {i}")
            for _ in range(EPOCHS):
                odd_output = self.odd_layer.forward(even_output, rx, llr_mask_only=self.odd_llr_mask_only)
                # even - check to variables
                for_training_even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
                # compute original output
                est = self.output_layers[i].forward(for_training_even_output, mask_only=self.output_mask_only,
                                                    phase=Phase.TRAIN)
                output = rx + est.priors
                # tilde calculation
                output_tilde = rx + est.arm_tilde
                # calculate loss and backtrack
                loss = self.calc_loss(tx=tx, output=output, output_tilde=output_tilde, est=est)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                odd_output = self.odd_layer.forward(even_output, rx, llr_mask_only=self.odd_llr_mask_only)
                # even - check to variables
                even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)

    def forward(self, x):
        """
        compute forward pass in the network
        :param x: [batch_size,N]
        :return: decoded word [batch_size,N]
        """
        # equation 1 and 2 from "Learning To Decode ..", i==1,2 (iteration 1)
        even_output = self.input_layer.forward(x)
        # now start iterating through all hidden layers i>2 (iteration 2 - Imax)
        for i in range(self.iteration_num - 1):
            # odd - variables to check
            odd_output = self.odd_layer.forward(even_output, x, llr_mask_only=self.odd_llr_mask_only)
            # even - check to variables
            even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
            # output layer
            output = x + self.output_layers[i].forward(even_output, mask_only=self.output_mask_only)
        return torch.round(torch.sigmoid(-output))
