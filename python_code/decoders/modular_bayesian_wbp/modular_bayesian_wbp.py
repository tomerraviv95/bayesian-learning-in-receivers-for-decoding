import torch
from torch.nn import BCEWithLogitsLoss, ModuleList, MSELoss
from torch.optim import Adam

from python_code import DEVICE
from python_code.decoders.bp_nn import InputLayer, EvenLayer, OutputLayer
from python_code.decoders.decoder_trainer import DecoderTrainer, LR, EPOCHS
from python_code.decoders.modular_bayesian_wbp.bayesian_bp_nn_odd import BayesianOddLayer
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
        self.odd_layers = ModuleList([BayesianOddLayer(clip_tanh=CLIPPING_VAL, input_output_layer_size=self._code_bits,
                                                       neurons=self.neurons, code_pcm=self.code_pcm,
                                                       ensemble_num=self.ensemble_num) for _ in
                                      range(self.iteration_num - 1)])
        self.even_layer = EvenLayer(CLIPPING_VAL, self.neurons, self.code_pcm)
        self.output_layer = OutputLayer(neurons=self.neurons, input_output_layer_size=self._code_bits,
                                        code_pcm=self.code_pcm)

    def calc_loss(self, tx: torch.Tensor, output: torch.Tensor, arm_output=None,
                  output_tilde: torch.Tensor = None, est: LossVariable = None, arm_tx=None):
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        loss = self.criterion(input=-output, target=tx)
        # ARM Loss
        loss_term_arm_original = self.criterion_arm(input=arm_output, target=arm_tx)
        loss_term_arm_tilde = self.criterion_arm(input=output_tilde, target=arm_tx)
        arm_delta = torch.mean(loss_term_arm_tilde - loss_term_arm_original, dim=1)
        grad_logit = arm_delta.unsqueeze(-1) * (est.u - HALF)
        arm_loss = grad_logit * est.dropout_logits.unsqueeze(0)
        arm_loss = self.arm_beta * torch.mean(arm_loss)
        # KL Loss
        kl_term = self.kl_beta * est.kl_term
        loss += arm_loss + kl_term
        return loss

    def single_training(self, tx: torch.Tensor, rx: torch.Tensor):
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=LR,
                              weight_decay=0.0005, betas=(0.5, 0.999))
        self.criterion = BCEWithLogitsLoss().to(DEVICE)
        self.criterion_arm = MSELoss(reduction='none').to(DEVICE)
        even_output = self.input_layer.forward(rx)
        for i in range(self.iteration_num - 1):
            for _ in range(EPOCHS):
                cur_even_output = even_output.clone()
                est = self.odd_layers[i].forward(cur_even_output, rx, llr_mask_only=self.odd_llr_mask_only,
                                                 phase=Phase.TRAIN)
                # even - check to variables
                cur_even_output = self.even_layer.forward(est.priors, mask_only=self.even_mask_only)
                # compute original outputs
                output = rx + self.output_layer.forward(cur_even_output, mask_only=self.output_mask_only)
                # tilde calculation
                even_output_tilde = self.even_layer.forward(est.arm_tilde, mask_only=self.even_mask_only)
                output_tilde = rx + self.output_layer.forward(even_output_tilde,
                                                              mask_only=self.output_mask_only)
                # calculate loss and backtrack
                loss = self.calc_loss(tx=tx, output=output, arm_output=output, output_tilde=output_tilde, est=est,
                                      arm_tx=tx)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            with torch.no_grad():
                est = self.odd_layers[i].forward(even_output, rx, llr_mask_only=self.odd_llr_mask_only,
                                                 phase=Phase.TRAIN)
                # even - check to variables
                even_output = self.even_layer.forward(est.priors, mask_only=self.even_mask_only)

    # def calc_loss(self, ests: List[LossVariable], tx: torch.Tensor, outputs: List[torch.Tensor],
    #               outputs_tilde: List[torch.Tensor]):
    #     """
    #     Cross Entropy loss - distribution over states versus the gt state label
    #     """
    #     loss = 0
    #     for iteration in range(self.iteration_num):
    #         current_loss = self.criterion(input=-outputs[iteration], target=tx)
    #         loss += current_loss
    #     for i in range(self.iteration_num - 1):
    #         # ARM Loss
    #         loss_term_arm_original = self.criterion_arm(input=-outputs[i + 1], target=tx)
    #         loss_term_arm_tilde = self.criterion_arm(input=-outputs_tilde[i], target=tx)
    #         arm_delta = loss_term_arm_tilde - loss_term_arm_original
    #         scaled_arm_delta = torch.matmul(arm_delta, self.odd_layer.w_skipconn2even_mask)
    #         grad_logit = scaled_arm_delta * (ests[i].u - HALF)
    #         grad_logit[grad_logit < 0] *= -1
    #         arm_loss = grad_logit * ests[i].dropout_logits.unsqueeze(0)
    #         arm_loss = self.arm_beta * torch.mean(arm_loss)
    #         # KL Loss
    #         kl_term = self.kl_beta * ests[i].kl_term
    #         loss += arm_loss + kl_term
    #     return loss
    #
    # def single_training(self, tx: torch.Tensor, rx: torch.Tensor):
    #     self.deep_learning_setup(LR)
    #     self.criterion_arm = BCEWithLogitsLoss(reduction='none').to(DEVICE)
    #     for _ in range(EPOCHS):
    #         output_list = [0] * self.iteration_num
    #         even_output = self.input_layer.forward(rx)
    #         output_list[0] = rx + self.output_layer.forward(even_output, mask_only=self.output_mask_only)
    #         outputs_tilde, ests = [], []
    #         for i in range(self.iteration_num - 1):
    #             est = self.odd_layer.forward(even_output, rx, llr_mask_only=self.odd_llr_mask_only,
    #                                              phase=Phase.TRAIN)
    #             ests.append(est)
    #             # even - check to variables
    #             even_output = self.even_layer.forward(est.priors, mask_only=self.even_mask_only)
    #             # compute original outputs
    #             output = rx + self.output_layer.forward(even_output, mask_only=self.output_mask_only)
    #             output_list[i + 1] = output.clone()
    #             # tilde calculation
    #             even_output_tilde = self.even_layer.forward(est.arm_tilde, mask_only=self.even_mask_only)
    #             output_tilde = rx + self.output_layer.forward(even_output_tilde,
    #                                                           mask_only=self.output_mask_only)
    #             outputs_tilde.append(output_tilde)
    #         # calculate loss and backtrack
    #         loss = self.calc_loss(tx=tx, outputs=output_list, outputs_tilde=outputs_tilde, ests=ests)
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()

    def forward(self, x):
        """
        compute forward pass in the 5network
        :param x: [batch_size,N]
        :return: decoded word [batch_size,N]
        """
        # equation 1 and 2 from "Learning To Decode ..", i==1,2 (iteration 1)
        even_output = self.input_layer.forward(x)
        # now start iterating through all hidden layers i>2 (iteration 2 - Imax)
        for i in range(self.iteration_num - 1):
            # odd - variables to check
            odd_output = self.odd_layers[i].forward(even_output, x, llr_mask_only=self.odd_llr_mask_only)
            # even - check to variables
            even_output = self.even_layer.forward(odd_output, mask_only=self.even_mask_only)
            # outputs layer
            output = x + self.output_layer.forward(even_output, mask_only=self.output_mask_only)
        return torch.round(torch.sigmoid(-output))
