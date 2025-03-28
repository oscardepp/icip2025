
import torch

def anscombe_transform(x):
    return 2 * torch.sqrt(x + 3/8)

def inverse_anscombe_transform(x):
    return torch.pow(x / 2, 2) 
       
def custom_regularizer(output, min_val=0.0, max_val=0.02, alpha=0.8):
    """
    A custom regularizer that encourages outputs to be closer to `min_val` or `max_val`.

    Args:
        output (torch.Tensor): The model output.
        min_val (float): The target minimum value.
        max_val (float): The target maximum value.
        alpha (float): Regularization strength.

    Returns:
        torch.Tensor: Regularization loss.
    """
    dist_to_min = (output - min_val) ** 2
    dist_to_max = (output - max_val) ** 2
    reg_loss = alpha * torch.mean(torch.min(dist_to_min, dist_to_max))
    return reg_loss

def total_time_regularizer(predicted_times, target_total_time, beta=0.8):
    """
    Regularizer that penalizes deviations of the total predicted times from the target total time.

    Args:
        predicted_times (torch.Tensor): The model's predicted dwell times.
        target_total_time (float): The target total time.
        beta (float): Regularization strength.

    Returns:
        torch.Tensor: Regularization loss.
    """
    total_pred_time = predicted_times.sum()
    reg_loss = beta * torch.abs(total_pred_time - target_total_time)
    return reg_loss