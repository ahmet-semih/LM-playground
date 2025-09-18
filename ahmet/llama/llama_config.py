import torch

class LlamaConfig():
  def __init__(
          self,
          vocab_size: int = 32_768,
          context_length: int = 512,
          emb_dim: int = 256,
          n_heads: int = 256,
          n_layers: int = 20,
          hidden_dim: int = 2048,
          n_kv_groups: int = 64,
          head_dim: int | None = None,
          dtype: torch.dtype = torch.float32,
          mlp_bias: bool = False,
          rms_norm_eps: float = 1e-6,
          bias: bool = False,
          attention_bias: bool = False,
        ):
      self.vocab_size = vocab_size
      self.max_position_embeddings = context_length
      self.hidden_size = emb_dim
      self.num_attention_heads = n_heads
      self.num_hidden_layers = n_layers
      self.num_key_value_heads = n_kv_groups
      self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
      self.dtype = dtype
      self.intermediate_size = hidden_dim
      self.mlp_bias = mlp_bias
      self.rms_norm_eps = rms_norm_eps
      self.bias = bias
      self.attention_bias = attention_bias