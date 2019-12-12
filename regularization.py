import torch
import torch.nn.functional as F


def elementwise_mse_regularization(mask, loc=0.5, scale=1.):
    return torch.mean(((mask - loc) * scale) ** 2)


def mean_mse_regularization(mask, loc=0.5, scale=1.):
    mask_mean = mask.reshape(mask.shape[0], -1).mean(dim=-1)
    return torch.mean(((mask_mean - loc) * scale) ** 2)


def rectified_mean_mse_regularization(mask, loc=0.5, scale=1., margin=0.3):
    mask_mean = mask.reshape(mask.shape[0], -1).mean(dim=-1)
    return torch.mean(F.relu(((mask_mean - loc) * scale) ** 2 - (margin * scale) ** 2))


def concrete_pdf(x, alpha=1.0, beta=0.5):
    return beta * alpha * (x ** (-beta - 1)) * ((1 - x) ** (-beta - 1)) / \
           (alpha * (x ** (-beta)) + (1 - x) ** (-beta)) ** 2


def concrete_cdf(x, log_alpha=0.0, beta=0.5):
    return torch.sigmoid((torch.log(x) - torch.log(1 - x)) * beta - log_alpha)


def mean_concrete_pdf_regularization(mask, alpha=1.0, beta=0.5):
    mask_mean = mask.reshape(mask.shape[0], -1).mean(dim=-1)
    return torch.mean(concrete_pdf(mask_mean, alpha=alpha, beta=beta))


def elementwise_hard_concrete_regularization(mask, alpha=1.0, beta=0.5, stretch_limits=(0.1, 0.9)):
    low, high = stretch_limits
    mask_stretched = mask * (high - low) + low
    pdfs = concrete_pdf(mask_stretched, alpha=alpha, beta=beta)
    return -torch.mean(torch.log(pdfs))
