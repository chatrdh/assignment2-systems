import torch

from cs336_basics.model import BasicsTransformerLM


model = BasicsTransformerLM(vocab_size=10000,context_length=256,d_model=768,num_layers=12,num_heads=12,d_ff=3072,rope_theta=10000.0)

def benchmark():
    pass