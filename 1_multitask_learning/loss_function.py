import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


class UncertaintyWeightedLoss(nn.Module):
    """
    Uncertainty-weighted multitask loss (Kendall et al., 2018)
    Learns task-dependent uncertainty parameters to balance losses
    """

    def __init__(self, num_tasks: int, reduction: str = "mean"):
        super().__init__()
        self.num_tasks = num_tasks
        self.reduction = reduction
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size, num_tasks = predictions.shape
        assert num_tasks == self.num_tasks, (
            f"Expected {self.num_tasks} tasks, got {num_tasks}"
        )

        total_loss = 0
        task_losses = {}
        weights = {}

        for i in range(num_tasks):
            task_pred = predictions[:, i]
            task_target = targets[:, i]

            base_loss = F.binary_cross_entropy_with_logits(
                task_pred, task_target, reduction="none"
            )

            precision = torch.exp(-self.log_vars[i])
            weighted_loss = 0.5 * precision * base_loss + 0.5 * self.log_vars[i]

            if self.reduction == "mean":
                task_loss = weighted_loss.mean()
            else:
                task_loss = weighted_loss.sum()

            total_loss += task_loss
            task_losses[f"task_{i}"] = base_loss.mean().item()
            weights[f"weight_{i}"] = precision.item()

        return total_loss, task_losses


class DynamicWeightAverageLoss(nn.Module):
    """
    Dynamic Weight Average (Liu et al., 2019)
    Adjusts task weights based on relative learning rates
    """

    def __init__(
        self, num_tasks: int, temperature: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.reduction = reduction
        self.prev_losses = None

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, epoch: int = 0
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size, num_tasks = predictions.shape

        task_losses = []
        for i in range(num_tasks):
            loss = F.binary_cross_entropy_with_logits(
                predictions[:, i], targets[:, i], reduction="mean"
            )
            task_losses.append(loss)

        task_losses = torch.stack(task_losses)

        if self.prev_losses is None or epoch == 0:
            weights = torch.ones(num_tasks) / num_tasks
        else:
            loss_ratios = task_losses / self.prev_losses
            weights = F.softmax(loss_ratios / self.temperature, dim=0)

        self.prev_losses = task_losses.detach().clone()

        weighted_loss = (weights * task_losses).sum()

        task_losses_dict = {
            f"task_{i}": task_losses[i].item() for i in range(num_tasks)
        }
        weights_dict = {f"weight_{i}": weights[i].item() for i in range(num_tasks)}

        return weighted_loss, {"task_losses": task_losses_dict, "weights": weights_dict}


class GradNormLoss(nn.Module):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing (Chen et al., 2018)
    Balances gradients across tasks
    """

    def __init__(self, num_tasks: int, alpha: float = 1.5, reduction: str = "mean"):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.reduction = reduction
        self.weights = nn.Parameter(torch.ones(num_tasks))

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        shared_parameters: List[nn.Parameter],
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            shared_parameters: List of shared network parameters for gradient computation
        """
        batch_size, num_tasks = predictions.shape

        # Compute individual task losses (binary classification)
        task_losses = []
        for i in range(num_tasks):
            loss = F.binary_cross_entropy_with_logits(
                predictions[:, i], targets[:, i], reduction="mean"
            )
            task_losses.append(loss)

        task_losses = torch.stack(task_losses)

        # Normalize weights
        normalized_weights = F.softmax(self.weights, dim=0) * num_tasks

        # Compute weighted loss
        weighted_loss = (normalized_weights * task_losses).sum()

        # GradNorm weight update (should be called separately in training loop)
        task_losses_dict = {
            f"task_{i}": task_losses[i].item() for i in range(num_tasks)
        }
        weights_dict = {
            f"weight_{i}": normalized_weights[i].item() for i in range(num_tasks)
        }

        return weighted_loss, {"task_losses": task_losses_dict, "weights": weights_dict}

    def update_weights(
        self,
        task_losses: torch.Tensor,
        shared_parameters: List[nn.Parameter],
        initial_losses: torch.Tensor,
        lr: float = 0.025,
    ):
        """Update weights based on gradient norms"""
        # Compute gradients w.r.t shared parameters
        grad_norms = []
        for i, loss in enumerate(task_losses):
            grad = torch.autograd.grad(
                loss, shared_parameters, retain_graph=True, create_graph=True
            )
            grad_norm = torch.norm(torch.cat([g.flatten() for g in grad]))
            grad_norms.append(grad_norm)

        grad_norms = torch.stack(grad_norms)

        # Compute relative inverse training rates
        loss_ratios = task_losses / initial_losses
        relative_rates = loss_ratios / loss_ratios.mean()

        # Target gradient norms
        mean_grad_norm = grad_norms.mean()
        target_grad_norms = mean_grad_norm * (relative_rates**self.alpha)

        # Compute gradient norm loss
        grad_norm_loss = F.l1_loss(grad_norms, target_grad_norms)

        # Update weights
        with torch.no_grad():
            weight_grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]
            self.weights -= lr * weight_grad
            self.weights.data = torch.clamp(self.weights.data, 0.1, 10.0)


class ConflictAverseGradientLoss(nn.Module):
    """
    Conflict-Averse Gradient descent (CAGrad) for multitask learning
    """

    def __init__(self, num_tasks: int, c: float = 0.5, reduction: str = "mean"):
        super().__init__()
        self.num_tasks = num_tasks
        self.c = c  # conflict-averse parameter
        self.reduction = reduction

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size, num_tasks = predictions.shape

        # Compute individual task losses (binary classification)
        task_losses = []
        for i in range(num_tasks):
            loss = F.binary_cross_entropy_with_logits(
                predictions[:, i], targets[:, i], reduction="mean"
            )
            task_losses.append(loss)

        task_losses = torch.stack(task_losses)

        # Simple averaging for forward pass (gradient manipulation happens in optimizer)
        total_loss = task_losses.mean()

        task_losses_dict = {
            f"task_{i}": task_losses[i].item() for i in range(num_tasks)
        }

        return total_loss, {"task_losses": task_losses_dict}
