import torch


def spike_rate_accuracy(spk_out: torch.Tensor, targets: torch.Tensor) -> float:
    _, predicted = spk_out.sum(dim=0).max(1)
    accuracy = (predicted == targets).float().mean().item()
    return accuracy

def spike_rate_recall(spk_out: torch.Tensor, targets: torch.Tensor) -> float:
    _, predicted = spk_out.sum(dim=0).max(1)

    # for true positive, targets == 1 and idx == 1
    tp = (targets * predicted).sum().item()
    # for false negative, targets == 1 and idx == 0
    fn = (targets * (1 - predicted)).sum().item()

    divider = tp + fn
    if divider == 0:
        return 0.0

    recall = tp / divider

    return recall

def spike_rate_precision(spk_out: torch.Tensor, targets: torch.Tensor) -> float:
    _, predicted = spk_out.sum(dim=0).max(1)

    # for true positive, targets == 1 and idx == 1
    tp = (targets * predicted).sum().item()
    # for false positive, targets == 0 and idx == 1
    fp = ((1 - targets) * predicted).sum().item()

    divider = tp + fp
    if divider == 0:
        return 0.0

    precision = tp / divider

    return precision

def spike_rate_f1(spk_out: torch.Tensor, targets: torch.Tensor) -> float:
    precision = spike_rate_precision(spk_out, targets)
    recall = spike_rate_recall(spk_out, targets)

    divider = precision + recall
    if divider == 0:
        return 0.0

    f1 = 2 * (precision * recall) / divider

    return f1

def spike_rate_mse(spk_out: torch.Tensor, targets: torch.Tensor) -> float:
    _, predicted = spk_out.sum(dim=0).max(1)
    mse = ((predicted - targets) ** 2).mean().item()
    return mse