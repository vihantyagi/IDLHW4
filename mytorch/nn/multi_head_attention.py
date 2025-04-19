from .linear import Linear
from .scaled_dot_product_attention import ScaledDotProductAttention
import numpy as np

class MultiHeadAttention:
    """
    Multi Head Attention
    """ 
    def __init__(self, embed_dim, num_heads):
        """
        :param embed_dim: Embedding dimension
        :param num_heads: Number of attention heads
        """
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # Initialize parameters and layers
        # DO NOT MODIFY
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Initialize your scaled dot product attention layer
        self.attention = ScaledDotProductAttention()
        
        # Initialize linear layers
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def init_weights(self, Wq, bq, Wk, bk, Wv, bv, Wo, bo):
        """
        Initialize the weights and biases with the given values.
        """
        # Initialize your linear layers (DO NOT MODIFY)
        self.q_proj.init_weights(Wq, bq)
        self.k_proj.init_weights(Wk, bk)
        self.v_proj.init_weights(Wv, bv)
        self.out_proj.init_weights(Wo, bo)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        :param query: (N, L, E)
        :param key: (N, S, E)
        :param value: (N, S, E)
        :param key_padding_mask: (N, S) where 1/True indicates positions to ignore
        :param attn_mask: (L, S) where 1/True indicates positions to ignore
        :return: (N, L, E)
        """
        # Store dimensions for backward pass
        self.N = query.shape[0]
        self.L = query.shape[1]
        self.S = key.shape[1]
        self.E = query.shape[2]
        
        # Project the query, key, and value inputs into query, key, and value
        # (N, L, E) -> (N, L, embed_dim)
        q = self.q_proj.forward(query)
        # (N, S, E) -> (N, S, embed_dim)
        k = self.k_proj.forward(key)
        # (N, S, E) -> (N, S, embed_dim)
        v = self.v_proj.forward(value)
        
        # Save projections for backward pass
        self.q_proj_output = q
        self.k_proj_output = k
        self.v_proj_output = v

        # Split the query, key, and value into multiple heads
        # (N, L, embed_dim) -> (N, num_heads, L, embed_dim // num_heads)
        q = self._split_heads(q)
        # (N, S, embed_dim) -> (N, num_heads, S, embed_dim // num_heads)
        k = self._split_heads(k)
        # (N, S, embed_dim) -> (N, num_heads, S, embed_dim // num_heads)
        v = self._split_heads(v)
        
        # Save split heads for backward pass
        self.q_split = q
        self.k_split = k
        self.v_split = v

        # Merge the masks if provided
        # (N, S) + (L, S) -> (N, H, L, S)
        mask = None
        if key_padding_mask is not None or attn_mask is not None:
            mask = self._merge_masks(key_padding_mask, attn_mask)

        # Apply the attention mechanism
        # (N, num_heads, L, embed_dim // num_heads)
        attn_outputs = self.attention.forward(q, k, v, mask)
        
        # Save attention outputs for backward pass
        self.attn_outputs = attn_outputs

        # Merge the attention outputs
        # (N, num_heads, L, embed_dim // num_heads) -> (N, L, embed_dim)
        attn_output = self._concat_heads(attn_outputs)
        
        # Save concatenated output for backward pass
        self.concat_output = attn_output

        # Project the attention outputs
        # (N, L, embed_dim) -> (N, L, embed_dim)
        output = self.out_proj.forward(attn_output)

        return output

    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, L, E)
        :return: Gradient of loss wrt input query, key, value of shapes (N, L, E), (N, S, E), (N, S, E)
        """
        # Backpropagate through the output projection
        # (N, L, embed_dim) -> (N, L, embed_dim)
        d_attn_output = self.out_proj.backward(d_output)

        # Split the gradients into multiple heads
        # (N, L, embed_dim) -> (N, num_heads, L, embed_dim // num_heads)
        d_attn_outputs = self._split_heads(d_attn_output)

        # Backpropagate through the attention mechanism
        # (N, num_heads, L, embed_dim // num_heads) -> (N, num_heads, L, embed_dim // num_heads)
        d_q_split, d_k_split, d_v_split = self.attention.backward(d_attn_outputs)

        # Merge the gradients
        # (N, num_heads, L, embed_dim // num_heads) -> (N, L, embed_dim)
        d_q_merged = self._concat_heads(d_q_split)
        # (N, num_heads, S, embed_dim // num_heads) -> (N, S, embed_dim)
        d_k_merged = self._concat_heads(d_k_split)
        # (N, num_heads, S, embed_dim // num_heads) -> (N, S, embed_dim)
        d_v_merged = self._concat_heads(d_v_split)

        # Backpropagate through the input projections
        # (N, L, embed_dim) -> (N, L, E)
        d_query = self.q_proj.backward(d_q_merged)
        # (N, S, embed_dim) -> (N, S, E)
        d_key = self.k_proj.backward(d_k_merged)
        # (N, S, embed_dim) -> (N, S, E)
        d_value = self.v_proj.backward(d_v_merged)

        return d_query, d_key, d_value

    def _merge_masks(self, key_padding_mask, attn_mask):
        """
        Merge key_padding_mask and attn_mask into a single mask.
        :param key_padding_mask: (N, S)
        :param attn_mask: (L, S)
        :return: (N, H, L, S)
        """
        # Initialize mask
        N = self.N
        H = self.num_heads
        L = self.L
        S = self.S
        
        # Create combined mask with the right shape (N, H, L, S)
        combined_mask = np.zeros((N, H, L, S), dtype=bool)
        
        # Expand key_padding_mask to (N, 1, 1, S) and broadcast to (N, H, L, S)
        if key_padding_mask is not None:
            key_mask = key_padding_mask.reshape(N, 1, 1, S)
            key_mask = np.broadcast_to(key_mask, (N, H, L, S))
            combined_mask = np.logical_or(combined_mask, key_mask)
        
        # Expand attn_mask to (1, 1, L, S) and broadcast to (N, H, L, S)
        if attn_mask is not None:
            attention_mask = attn_mask.reshape(1, 1, L, S)
            attention_mask = np.broadcast_to(attention_mask, (N, H, L, S))
            combined_mask = np.logical_or(combined_mask, attention_mask)
        
        return combined_mask

    def _split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the front.
        :param x: (N, L, embed_dim)
        :return: (N, num_heads, L, embed_dim // num_heads)
        """
        N = x.shape[0]
        L = x.shape[1]
        
        # Determine head dimension
        head_dim = self.embed_dim // self.num_heads
        
        # Transpose: (N, L, num_heads, embed_dim // num_heads) -> (N, num_heads, L, embed_dim // num_heads)
        x = x.reshape(N, L, self.num_heads, head_dim)
        
        # Transpose: (N, L, num_heads, head_dim) -> (N, num_heads, L, head_dim)
        x = np.transpose(x, (0, 2, 1, 3))
        
        return x

    def _concat_heads(self, x):
        """
        Concatenate the last dimension into (num_heads, d_k).
        Transpose to move num_heads dimension to the back.
        :param x: (N, num_heads, L, embed_dim // num_heads)
        :return: (N, L, embed_dim)
        """
        N = x.shape[0]
        H = x.shape[1]
        L = x.shape[2]
        head_dim = x.shape[3]
        
        # Transpose: (N, num_heads, L, head_dim) -> (N, L, num_heads, head_dim)
        x = np.transpose(x, (0, 2, 1, 3))
        
        # Reshape: (N, L, num_heads, head_dim) -> (N, L, embed_dim)
        x = x.reshape(N, L, H * head_dim)
        
        return x
