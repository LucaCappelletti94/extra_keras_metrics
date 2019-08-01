from typing import List
from tensorflow import convert_to_tensor
from numpy import ndarray

def to_tensor(*arrays:List[ndarray])->List:
    """Convert given arrays to tensors."""
    return [
        convert_to_tensor(a) for a in arrays
    ]