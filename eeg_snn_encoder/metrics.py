import torch

__doc__ = """
This module contains functions to compute various metrics for evaluating the performance of spiking neural networks (SNNs).
These metrics include accuracy, recall, precision, F1 score, and mean squared error (MSE) based on the spike count predictions.
The functions are designed to work with binary spike count outputs and their corresponding target labels.
"""


def spike_count_accuracy(spk_out: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the accuracy of binary spike count predictions.

    This function calculates the accuracy by comparing the predicted binary spike count
    (based on the sum of spikes from two nodes) with the target binary labels.
    It sums the spike outputs along the specified dimension, determines the predicted class,
    and compares it with the binary target labels.

    Parameters
    ----------
    spk_out : torch.Tensor
        A tensor containing the output from the model.
    targets : torch.Tensor
        A tensor containing the ground truth binary labels (0 or 1).

    Returns
    -------
    float
        The accuracy of the spike count predictions, as a float between 0 and 1.
    """
    # Summing the spike counts and determining the predicted class
    _, predicted = spk_out.sum(dim=0).max(1)

    # Calculating the accuracy by comparing predicted and actual binary labels
    accuracy = (predicted == targets).float().mean().item()
    return accuracy


def spike_count_recall(spk_out: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the recall of binary spike count predictions.

    This function calculates the recall by determining the true positives and false negatives
    based on the predicted binary spike count and the target binary labels.

    Parameters
    ----------
    spk_out : torch.Tensor
        A tensor containing the output from the model.
    targets : torch.Tensor
        A tensor containing the ground truth binary labels (0 or 1).

    Returns
    -------
    float
        The recall of the spike count predictions, as a float between 0 and 1.
    """
    _, predicted = spk_out.sum(dim=0).max(1)

    tp = (targets * predicted).sum().item()  # True positives
    fn = (targets * (1 - predicted)).sum().item()  # False negatives

    divider = tp + fn
    if divider == 0:
        return 0.0

    recall = tp / divider
    return recall


def spike_count_precision(spk_out: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the precision of binary spike count predictions.

    This function calculates the precision by determining the true positives and false positives
    based on the predicted binary spike count and the target binary labels.

    Parameters
    ----------
    spk_out : torch.Tensor
        A tensor containing the output from the model.
    targets : torch.Tensor
        A tensor containing the ground truth binary labels (0 or 1).

    Returns
    -------
    float
        The precision of the spike count predictions, as a float between 0 and 1.
    """
    _, predicted = spk_out.sum(dim=0).max(1)

    tp = (targets * predicted).sum().item()  # True positives
    fp = ((1 - targets) * predicted).sum().item()  # False positives

    divider = tp + fp
    if divider == 0:
        return 0.0

    precision = tp / divider
    return precision


def spike_count_f1(spk_out: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the F1 score of binary spike count predictions.

    This function calculates the F1 score, which is the harmonic mean of precision and recall,
    based on the predicted binary spike count and the target binary labels.

    Parameters
    ----------
    spk_out : torch.Tensor
        A tensor containing the output from the model.
    targets : torch.Tensor
        A tensor containing the ground truth binary labels (0 or 1).

    Returns
    -------
    float
        The F1 score of the spike count predictions, as a float between 0 and 1.
    """
    precision = spike_count_precision(spk_out, targets)
    recall = spike_count_recall(spk_out, targets)

    divider = precision + recall
    if divider == 0:
        return 0.0

    f1 = 2 * (precision * recall) / divider
    return f1


def spike_count_mse(spk_out: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute the mean squared error (MSE) of binary spike count predictions.

    This function calculates the MSE by comparing the predicted binary spike counts with the ground truth
    binary labels and averaging the squared differences.

    Parameters
    ----------
    spk_out : torch.Tensor
        A tensor containing the output from the model.
    targets : torch.Tensor
        A tensor containing the ground truth binary labels (0 or 1).

    Returns
    -------
    float
        The mean squared error of the spike count predictions.
    """
    _, predicted = spk_out.sum(dim=0).max(1)
    mse = ((predicted - targets) ** 2).mean().item()
    return mse
