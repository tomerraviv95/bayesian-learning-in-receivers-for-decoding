import torch
from torch.nn.parameter import Parameter

from python_code import DEVICE
from python_code.decoders.bp_nn_weights import init_w_output
from python_code.utils.bayesian_utils import dropout_ori, dropout_tilde, entropy, LossVariable
from python_code.utils.constants import Phase, LOGITS_INIT


class BayesianOutputLayer(torch.nn.Module):
    def __init__(self, neurons, input_output_layer_size, code_pcm, ensemble_num):
        super(BayesianOutputLayer, self).__init__()
        w_output, w_output_mask = init_w_output(neurons=neurons, input_output_layer_size=input_output_layer_size,
                                                code_pcm=code_pcm)
        self.output_weights = Parameter(w_output.to(DEVICE))
        self.w_output_mask = w_output_mask.to(device=DEVICE)
        self.dropout_logits = Parameter(LOGITS_INIT * torch.ones(w_output_mask.shape[0]).to(DEVICE))
        self.ensemble_num = ensemble_num
        self.kl_scale = 1

    def forward(self, x, mask_only=False, phase=Phase.TEST):
        if phase == Phase.TEST:
            log_probs = 0
            for _ in range(self.ensemble_num):
                u = torch.rand(x.shape).to(DEVICE)
                x_after_dropout = dropout_ori(x, self.dropout_logits, u)
                if mask_only:
                    log_probs += torch.matmul(x_after_dropout, self.w_output_mask)
                else:
                    log_probs += torch.matmul(x_after_dropout, self.w_output_mask * self.output_weights)
            return log_probs / self.ensemble_num
        # Bayesian computations
        u = torch.rand(x.shape).to(DEVICE)
        x_after_dropout = dropout_ori(x, self.dropout_logits, u)
        if mask_only:
            out = torch.matmul(x_after_dropout, self.w_output_mask)
        else:
            out = torch.matmul(x_after_dropout, self.w_output_mask * self.output_weights)
        # computation for ARM loss
        x_after_dropout_tilde = dropout_tilde(x, self.dropout_logits, u)
        if mask_only:
            out_tilde = torch.matmul(x_after_dropout_tilde, self.w_output_mask)
        else:
            out_tilde = torch.matmul(x_after_dropout_tilde, self.w_output_mask * self.output_weights)
        scaling1 = (self.kl_scale ** 2 / 2) * (torch.sigmoid(self.dropout_logits).reshape(-1))
        first_layer_kl = scaling1 * torch.norm(self.output_weights, dim=1) ** 2
        H1 = entropy(torch.sigmoid(self.dropout_logits).reshape(-1))
        kl_term = torch.mean(first_layer_kl - H1)
        return LossVariable(priors=out, arm_original=out, arm_tilde=out_tilde, u=u, kl_term=kl_term,
                            dropout_logits=self.dropout_logits)
