import torch
import torch.nn.functional as F


def elementwise_mse_regularization(mask, loc=0.5):
    """
    Return mean pixel-wise squared difference of mask from loc.

    :param mask: regularized mask
    :param loc: prior optimal value for each pixel
    :return: regularization value
    """
    return torch.mean((mask - loc) ** 2)


def mean_mse_regularization(mask, loc=0.5):
    """
    Return mean squared difference of mean mask probability from loc.

    :param mask: regularized mask
    :param loc: prior optimal value for mean mask probability
    :return: regularization value
    """
    mask_mean = mask.reshape(mask.shape[0], -1).mean(dim=-1)
    return torch.mean((mask_mean - loc) ** 2)


def rectified_mean_mse_regularization(mask, loc=0.5, margin=0.3):
    """
    Return mean squared difference of mean mask probability from loc rectified by margin.

    :param mask: regularized mask
    :param loc: prior optimal value for mean mask probability
    :param margin: regularization for values in [loc - margin, loc + margin] is zero
    :return: regularization value
    """
    mask_mean = mask.reshape(mask.shape[0], -1).mean(dim=-1)
    return torch.mean(F.relu((mask_mean - loc) ** 2 - margin ** 2))


def concrete_pdf(x, alpha=1.0, beta=0.5):
    """
    Calculates density of the concrete distribution.

    :param x: point where compute density
    :param alpha: parameter of concrete distribution
    :param beta: parameter of concrete distribution
    :return: density of concrete distribution
    """
    return beta * alpha * (x ** (-beta - 1)) * ((1 - x) ** (-beta - 1)) / \
           (alpha * (x ** (-beta)) + (1 - x) ** (-beta)) ** 2


def concrete_cdf(x, log_alpha=0.0, beta=0.5):
    """
    Calculates CDF of the concrete distribution.

    :param x: point where compute density
    :param log_alpha: parameter of concrete distribution
    :param beta: parameter of concrete distribution
    :return: CDF of concrete distribution at x
    """
    return torch.sigmoid((torch.log(x) - torch.log(1 - x)) * beta - log_alpha)


def mean_concrete_pdf_regularization(mask, alpha=1.0, beta=0.5):
    """
    Calculates regularization on mean mask probability based on the concrete distribution.

    :param mask: regularized mask
    :param alpha: parameter of concrete distribution
    :param beta: parameter of concrete distribution
    :return: CDF of concrete distribution at x
    """
    mask_mean = mask.reshape(mask.shape[0], -1).mean(dim=-1)
    return torch.mean(concrete_pdf(mask_mean, alpha=alpha, beta=beta))
