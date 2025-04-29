import torch

''' 
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean tensor of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
- Mask should be on same device as input tensor
'''
def PadMask(padded_input, input_lengths):
    """ 
    Create a mask to identify non-padding positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: The actual lengths of each sequence before padding, shape (N,).
    Returns:
        A boolean mask tensor with shape (N, T), where: 
            - padding positions are marked with True 
            - non-padding positions are marked with False.
    """
    # TODO: Implement PadMask

    batch_size = padded_input.shape[0]
    seq_len = padded_input.shape[1]
    
    # Creating a position tensor (N, T) = (batch_size, seq_len)
    positions = torch.arange(seq_len, device=padded_input.device).expand(batch_size, seq_len)
    lengths_expanded = input_lengths.unsqueeze(1).expand(batch_size, seq_len)
    mask = positions >= lengths_expanded
    
    return mask

''' 
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean tensor of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be on same device as input tensor
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """ 
    Create a mask to identify non-causal positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
    
    Returns:
        A boolean mask tensor with shape (T, T), where: 
            - non-causal positions (don't attend to) are marked with True 
            - causal positions (can attend to) are marked with False.
    """
    # TODO: Implement CausalMask

    # Get sequence length
    seq_len = padded_input.shape[1]
    
    # Creating a matrix of shape (T, T) where each element (i, j) is True if j > i 
    i = torch.arange(seq_len, device=padded_input.device)
    j = torch.arange(seq_len, device=padded_input.device)
    j_indices = j.unsqueeze(0).expand(seq_len, seq_len)
    i_indices = i.unsqueeze(1).expand(seq_len, seq_len)
    # creating an upper triangular matrix filled with true. 
    mask = j_indices > i_indices
    
    return mask

