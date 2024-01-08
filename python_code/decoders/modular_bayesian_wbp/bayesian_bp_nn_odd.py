import torch
from torch.nn.parameter import Parameter

from python_code import DEVICE
from python_code.decoders.bp_nn_weights import init_w_skipconn2even, initialize_w_v2c
from python_code.utils.bayesian_utils import dropout_ori, dropout_tilde, entropy, LossVariable
from python_code.utils.constants import Phase, LOGITS_INIT


class BayesianOddLayer(torch.nn.Module):
    def __init__(self, clip_tanh, input_output_layer_size, neurons, code_pcm, ensemble_num):
        super(BayesianOddLayer, self).__init__()
        w_skipconn2even, w_skipconn2even_mask = init_w_skipconn2even(input_output_layer_size=input_output_layer_size,
                                                                     neurons=neurons,
                                                                     code_pcm=code_pcm)
        w_odd2even, w_odd2even_mask = initialize_w_v2c(neurons=neurons, code_pcm=code_pcm)
        self.odd_weights = Parameter(w_odd2even.to(DEVICE))
        self.llr_weights = Parameter(w_skipconn2even.to(DEVICE))
        self.w_odd2even_mask = w_odd2even_mask.to(device=DEVICE).bool()
        self.w_skipconn2even_mask = w_skipconn2even_mask.to(device=DEVICE)
        self.clip_tanh = clip_tanh
        self.dropout_logits = Parameter(LOGITS_INIT * torch.ones(self.w_odd2even_mask.shape[1]).to(DEVICE))
        self.ensemble_num = ensemble_num
        self.kl_scale = 1

    def forward(self, x, llr, llr_mask_only=False, phase=Phase.TEST):
        mask = self.w_odd2even_mask * self.odd_weights
        x_after_mask = torch.matmul(x, mask)
        if llr_mask_only:
            odd_weights_llrs_mask = self.w_skipconn2even_mask
        else:
            odd_weights_llrs_mask = self.w_skipconn2even_mask * self.llr_weights
        odd_weights_times_llr = torch.matmul(llr, odd_weights_llrs_mask)
        if phase == Phase.TEST:
            log_probs = 0
            for _ in range(self.ensemble_num):
                u = torch.rand(x_after_mask.shape).to(DEVICE)
                x_after_mask_after_dropout = dropout_ori(x_after_mask, self.dropout_logits, u)
                odd_clamp = torch.clamp(x_after_mask_after_dropout + odd_weights_times_llr, min=-self.clip_tanh,
                                        max=self.clip_tanh)
                log_probs += torch.tanh(0.5 * odd_clamp)
            return log_probs / self.ensemble_num
        ## in training
        u = torch.rand(x_after_mask.shape).to(DEVICE)
        x_after_mask_after_dropout = dropout_ori(x_after_mask, self.dropout_logits, u)
        odd_clamp = torch.clamp(x_after_mask_after_dropout + odd_weights_times_llr, min=-self.clip_tanh,
                                max=self.clip_tanh)
        out = torch.tanh(0.5 * odd_clamp)
        # tilde computations for ARM loss
        x_after_mask_after_dropout_tilde = dropout_tilde(x_after_mask, self.dropout_logits, u)
        odd_clamp_tilde = torch.clamp(x_after_mask_after_dropout_tilde + odd_weights_times_llr, min=-self.clip_tanh,
                                      max=self.clip_tanh)
        out_tilde = torch.tanh(0.5 * odd_clamp_tilde)
        # kl term
        scaling1 = (self.kl_scale ** 2 / 2) * (torch.sigmoid(self.dropout_logits).reshape(-1))
        first_layer_kl = scaling1 * torch.norm(mask, dim=0) ** 2
        H1 = entropy(torch.sigmoid(self.dropout_logits).reshape(-1))
        kl_term = torch.mean(first_layer_kl - H1)
        return LossVariable(priors=out, arm_original=out,
                            arm_tilde=out_tilde, u=u, kl_term=kl_term,
                            dropout_logits=self.dropout_logits)
