import torch


class GuidedAttentionLoss(torch.nn.Module):
    """Wrapper around all loss functions including the loss of Tacotron 2.

    Details:
        - L2 of the prediction before and after the postnet.
        - Cross entropy of the stop tokens
        - Guided attention loss:
            prompt the attention matrix to be nearly diagonal, this is how people usualy read text
            introduced by 'Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention'
    Arguments:
        guided_att_steps -- number of training steps for which the guided attention is enabled
        guided_att_variance -- initial allowed variance of the guided attention (strictness of diagonal) 
        guided_att_gamma -- multiplier which is applied to guided_att_variance at every update_states call
    """

    def __init__(self, guided_att_steps, guided_att_variance, guided_att_gamma):
        super(GuidedAttentionLoss, self).__init__()
        self._g = guided_att_variance
        self._gamma = guided_att_gamma
        self._g_steps = guided_att_steps

    def forward(self, alignments, input_lengths, target_lengths, global_step):
        if self._g_steps < global_step:
            return 0
        self._g = self._gamma ** global_step
        # compute guided attention weights (diagonal matrix with zeros on a 'blurry' diagonal)
        weights = torch.zeros_like(alignments)
        for i, (f, l) in enumerate(zip(target_lengths, input_lengths)):
            grid_f, grid_l = torch.meshgrid(torch.arange(f, dtype=torch.float, device=f.device),
                                            torch.arange(l, dtype=torch.float, device=l.device))
            weights[i, :f, :l] = 1 - \
                torch.exp(-(grid_l / l - grid_f / f) ** 2 / (2 * self._g ** 2))

            # apply weights and compute mean loss
        loss = torch.sum(weights * alignments, dim=(1, 2))
        loss = torch.mean(loss / target_lengths.float())

        return loss
