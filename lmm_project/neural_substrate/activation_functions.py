import math
import numpy as np
from typing import Callable, Dict, Union

def sigmoid(x: float) -> float:
    """
    Sigmoid activation function: 1 / (1 + e^-x)
    
    Squashes input to range (0, 1)
    """
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def relu(x: float) -> float:
    """
    Rectified Linear Unit: max(0, x)
    
    Returns x if x > 0, otherwise 0
    """
    return max(0.0, x)

def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """
    Leaky ReLU: max(alpha*x, x)
    
    Like ReLU but allows small gradient when x < 0
    """
    return max(alpha * x, x)

def tanh(x: float) -> float:
    """
    Hyperbolic tangent: (e^x - e^-x) / (e^x + e^-x)
    
    Squashes input to range (-1, 1)
    """
    return math.tanh(x)

def softmax(x: Union[list, np.ndarray]) -> np.ndarray:
    """
    Softmax function: e^x_i / sum(e^x_j)
    
    Converts vector to probability distribution
    """
    x_np = np.array(x)
    # Subtract max for numerical stability
    e_x = np.exp(x_np - np.max(x_np))
    return e_x / e_x.sum()

def identity(x: float) -> float:
    """
    Identity function: x
    
    Returns input unchanged
    """
    return x

def binary_step(x: float) -> float:
    """
    Binary step function: 0 if x < 0, 1 if x >= 0
    
    Simple threshold function
    """
    return 0.0 if x < 0 else 1.0

# Dictionary mapping function names to implementations
ACTIVATION_FUNCTIONS = {
    "sigmoid": sigmoid,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "tanh": tanh,
    "softmax": softmax,
    "identity": identity,
    "binary_step": binary_step
}

def get_activation_function(name: str) -> Callable:
    """
    Get activation function by name
    
    Parameters:
    name: Name of the activation function
    
    Returns:
    The activation function
    
    Raises:
    ValueError: If the activation function is not recognized
    """
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {name}")
    
    return ACTIVATION_FUNCTIONS[name]
