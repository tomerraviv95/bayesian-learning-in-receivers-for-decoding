import torch
from torch.nn.parameter import Parameter

from python_code import DEVICE
from python_code.decoders.bp_nn_weights import init_w_skipconn2even, \
    initialize_w_v2c
from python_code.utils.bayesian_utils import dropout_ori, dropout_tilde, entropy, dropout, LossVariable
from python_code.utils.constants import Phase, LOGITS_INIT


class BayesianOddLayer(torch.nn.Module):
    def __init__(self, clip_tanh, input_output_layer_size, neurons, code_pcm):
        super(BayesianOddLayer, self).__init__()
        w_skipconn2even, w_skipconn2even_mask = init_w_skipconn2even(input_output_layer_size=input_output_layer_size,
                                                                     neurons=neurons,
                                                                     code_pcm=code_pcm)
        w_odd2even, w_odd2even_mask = initialize_w_v2c(neurons=neurons, code_pcm=code_pcm)
        self.odd_weights = Parameter(w_odd2even.to(DEVICE))
        self.llr_weights = Parameter(w_skipconn2even.to(DEVICE))
        self.w_odd2even_mask = w_odd2even_mask.to(device=DEVICE)
        self.dropout_logits = Parameter(10000* LOGITS_INIT * torch.ones(w_odd2even_mask.shape[0]).to(DEVICE))
        self.w_skipconn2even_mask = w_skipconn2even_mask.to(device=DEVICE)
        self.clip_tanh = clip_tanh
        self.kl_scale = 1

    def forward(self, x, llr, mask_only=False, phase=Phase.TEST):
        u = torch.rand(self.odd_weights.shape[0]).to(DEVICE)
        total_mask = self.w_odd2even_mask * self.odd_weights
        if phase == Phase.TEST:
            mask_after_dropout = dropout(total_mask, self.dropout_logits, u)
        elif phase == Phase.TRAIN:
            mask_after_dropout = dropout_ori(total_mask, self.dropout_logits, u)
        odd_weights_times_messages_after_dropout = torch.matmul(x, mask_after_dropout)
        if mask_only:
            odd_weights_times_llr = torch.matmul(llr, self.w_skipconn2even_mask)
        else:
            odd_weights_times_llr = torch.matmul(llr, self.w_skipconn2even_mask * self.llr_weights)
        odd_clamp = torch.clamp(odd_weights_times_messages_after_dropout + odd_weights_times_llr,
                                min=-self.clip_tanh, max=self.clip_tanh)
        out = torch.tanh(0.5 * odd_clamp)
        if phase == Phase.TRAIN:
            # computation for ARM loss
            mask_after_dropout_tilde = dropout_tilde(total_mask, self.dropout_logits, u)
            odd_weights_times_messages_tilde = torch.matmul(x, mask_after_dropout_tilde)
            odd_clamp_tilde = torch.clamp(odd_weights_times_messages_tilde + odd_weights_times_llr,
                                          min=-self.clip_tanh, max=self.clip_tanh)
            arm_tilde = torch.tanh(0.5 * odd_clamp_tilde)
            scaling1 = (self.kl_scale ** 2 / 2) * (torch.sigmoid(self.dropout_logits).reshape(-1))
            first_layer_kl = scaling1 * torch.norm(self.odd_weights, dim=1) ** 2
            H1 = entropy(torch.sigmoid(self.dropout_logits).reshape(-1))
            kl_term = torch.mean(first_layer_kl - H1)
            return LossVariable(priors=out, arm_original=out, arm_tilde=arm_tilde, u=u, kl_term=kl_term,
                                dropout_logits=self.dropout_logits)
        return out
