"""
Activation functions for the neural substrate.

This module provides various activation functions used by neurons in the LMM system.
Each function includes both the forward activation and its derivative for learning.
"""
import math
from enum import Enum, auto
from typing import Callable, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class ActivationType(Enum):
    """Types of activation functions available in the system."""
    SIGMOID = auto()
    TANH = auto()
    RELU = auto()
    LEAKY_RELU = auto()
    ELU = auto()
    SOFTMAX = auto()
    LINEAR = auto()
    GAUSSIAN = auto()


# Type alias for activation functions
ActivationFunction = Callable[[Union[float, np.ndarray, torch.Tensor]], 
                             Union[float, np.ndarray, torch.Tensor]]
ActivationWithDerivative = Tuple[ActivationFunction, ActivationFunction]


def sigmoid(x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Sigmoid activation function: f(x) = 1 / (1 + exp(-x))
    
    Parameters:
    x: Input value(s)
    
    Returns:
    Activated value(s) between 0 and 1
    """
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    elif isinstance(x, np.ndarray):
        return 1.0 / (1.0 + np.exp(-x))
    else:
        return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Derivative of sigmoid function: f'(x) = f(x) * (1 - f(x))
    
    Parameters:
    x: Input value(s) (pre-activation)
    
    Returns:
    Derivative value(s)
    """
    sig_x = sigmoid(x)
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return sig_x * (1 - sig_x)
    else:
        return sig_x * (1 - sig_x)


def tanh(x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Hyperbolic tangent activation function: f(x) = tanh(x)
    
    Parameters:
    x: Input value(s)
    
    Returns:
    Activated value(s) between -1 and 1
    """
    if isinstance(x, torch.Tensor):
        return torch.tanh(x)
    elif isinstance(x, np.ndarray):
        return np.tanh(x)
    else:
        return math.tanh(x)


def tanh_derivative(x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Derivative of tanh function: f'(x) = 1 - f(x)^2
    
    Parameters:
    x: Input value(s) (pre-activation)
    
    Returns:
    Derivative value(s)
    """
    tanh_x = tanh(x)
    if isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
        return 1 - tanh_x * tanh_x
    else:
        return 1 - tanh_x * tanh_x


def relu(x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Rectified Linear Unit activation function: f(x) = max(0, x)
    
    Parameters:
    x: Input value(s)
    
    Returns:
    Activated value(s) (x if x > 0, otherwise 0)
    """
    if isinstance(x, torch.Tensor):
        return F.relu(x)
    elif isinstance(x, np.ndarray):
        return np.maximum(0, x)
    else:
        return max(0, x)


def relu_derivative(x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Derivative of ReLU function: f'(x) = 1 if x > 0, otherwise 0
    
    Parameters:
    x: Input value(s) (pre-activation)
    
    Returns:
    Derivative value(s)
    """
    if isinstance(x, torch.Tensor):
        return (x > 0).float()
    elif isinstance(x, np.ndarray):
        return np.where(x > 0, 1.0, 0.0)
    else:
        return 1.0 if x > 0 else 0.0


def leaky_relu(x: Union[float, np.ndarray, torch.Tensor], 
               alpha: float = 0.01) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Leaky ReLU activation function: f(x) = x if x > 0, otherwise alpha * x
    
    Parameters:
    x: Input value(s)
    alpha: Slope for negative values
    
    Returns:
    Activated value(s)
    """
    if isinstance(x, torch.Tensor):
        return F.leaky_relu(x, alpha)
    elif isinstance(x, np.ndarray):
        return np.where(x > 0, x, alpha * x)
    else:
        return x if x > 0 else alpha * x


def leaky_relu_derivative(x: Union[float, np.ndarray, torch.Tensor], 
                          alpha: float = 0.01) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Derivative of Leaky ReLU function: f'(x) = 1 if x > 0, otherwise alpha
    
    Parameters:
    x: Input value(s) (pre-activation)
    alpha: Slope for negative values
    
    Returns:
    Derivative value(s)
    """
    if isinstance(x, torch.Tensor):
        return torch.where(x > 0, torch.ones_like(x), torch.tensor(alpha, device=x.device))
    elif isinstance(x, np.ndarray):
        return np.where(x > 0, 1.0, alpha)
    else:
        return 1.0 if x > 0 else alpha


def elu(x: Union[float, np.ndarray, torch.Tensor], 
        alpha: float = 1.0) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Exponential Linear Unit: f(x) = x if x > 0, otherwise alpha * (exp(x) - 1)
    
    Parameters:
    x: Input value(s)
    alpha: Scale for negative values
    
    Returns:
    Activated value(s)
    """
    if isinstance(x, torch.Tensor):
        return F.elu(x, alpha)
    elif isinstance(x, np.ndarray):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    else:
        return x if x > 0 else alpha * (math.exp(x) - 1)


def elu_derivative(x: Union[float, np.ndarray, torch.Tensor], 
                   alpha: float = 1.0) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Derivative of ELU function: f'(x) = 1 if x > 0, otherwise f(x) + alpha
    
    Parameters:
    x: Input value(s) (pre-activation)
    alpha: Scale for negative values
    
    Returns:
    Derivative value(s)
    """
    elu_x = elu(x, alpha)
    if isinstance(x, torch.Tensor):
        return torch.where(x > 0, torch.ones_like(x), elu_x + alpha)
    elif isinstance(x, np.ndarray):
        return np.where(x > 0, 1.0, elu_x + alpha)
    else:
        return 1.0 if x > 0 else elu_x + alpha


def gaussian(x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Gaussian activation function: f(x) = exp(-x^2)
    
    Parameters:
    x: Input value(s)
    
    Returns:
    Activated value(s) between 0 and 1
    """
    if isinstance(x, torch.Tensor):
        return torch.exp(-torch.pow(x, 2))
    elif isinstance(x, np.ndarray):
        return np.exp(-np.power(x, 2))
    else:
        return math.exp(-x * x)


def gaussian_derivative(x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
    """
    Derivative of Gaussian function: f'(x) = -2x * exp(-x^2)
    
    Parameters:
    x: Input value(s) (pre-activation)
    
    Returns:
    Derivative value(s)
    """
    if isinstance(x, torch.Tensor):
        return -2 * x * torch.exp(-torch.pow(x, 2))
    elif isinstance(x, np.ndarray):
        return -2 * x * np.exp(-np.power(x, 2))
    else:
        return -2 * x * math.exp(-x * x)


# Dictionary mapping activation types to (function, derivative) pairs
ACTIVATION_FUNCTIONS: Dict[ActivationType, ActivationWithDerivative] = {
    ActivationType.SIGMOID: (sigmoid, sigmoid_derivative),
    ActivationType.TANH: (tanh, tanh_derivative),
    ActivationType.RELU: (relu, relu_derivative),
    ActivationType.LEAKY_RELU: (leaky_relu, leaky_relu_derivative),
    ActivationType.ELU: (elu, elu_derivative),
    ActivationType.LINEAR: (lambda x: x, lambda x: 1),
    ActivationType.GAUSSIAN: (gaussian, gaussian_derivative),
}


def get_activation_function(activation_type: ActivationType) -> ActivationWithDerivative:
    """
    Get the activation function and its derivative for the specified type.
    
    Parameters:
    activation_type: Type of activation function
    
    Returns:
    Tuple of (activation_function, derivative_function)
    """
    return ACTIVATION_FUNCTIONS[activation_type]
