import argparse
import torch
import timeit
import statistics
from dataclasses import dataclass

from cs336_basics.model import BasicsTransformerLM


@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    rope_theta: float = 10000.0


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    batch_size: int = 8
    seq_length: int = 256
    device: str = "cuda"
    num_warmup: int = 5
    num_runs: int = 10
    mode: str = "both"  # "forward" or "both" (forward + backward)


def create_model(config: ModelConfig = None) -> BasicsTransformerLM:
    """Create a model from configuration."""
    if config is None:
        config = ModelConfig()
    return BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    )


# Default configuration and model
model_config = ModelConfig()
model = create_model(model_config)

def generate_random_data(batch_size: int = 8, seq_length: int = 256, vocab_size: int = 10000, device: str = "cuda"):
    """Generate random input data for the model.
    
    Args:
        batch_size: Number of sequences in the batch.
        seq_length: Length of each sequence.
        vocab_size: Size of the vocabulary (for valid token IDs).
        device: Device to place the tensor on ("cuda" or "cpu").
    
    Returns:
        A tensor of random token IDs with shape (batch_size, seq_length).
    """
    return torch.randint(0, vocab_size, (batch_size, seq_length), device=device)


def benchmark(
    benchmark_config: BenchmarkConfig = None,
    model_config: ModelConfig = None,
):
    """Benchmark the model's forward or forward+backward pass.
    
    Args:
        benchmark_config: Configuration for benchmarking parameters.
        model_config: Configuration for the model (creates new model if provided).
    
    Returns:
        dict with timing results (mean, std, total time).
    """
    # Use default configs if not provided
    if benchmark_config is None:
        benchmark_config = BenchmarkConfig()
    if model_config is None:
        model_config = ModelConfig()
    
    # Create or use existing model
    bench_model = create_model(model_config)
    bench_model.to(benchmark_config.device)
    bench_model.train()
    
    # Extract config values for convenience
    batch_size = benchmark_config.batch_size
    seq_length = benchmark_config.seq_length
    vocab_size = model_config.vocab_size
    device = benchmark_config.device
    num_warmup = benchmark_config.num_warmup
    num_runs = benchmark_config.num_runs
    mode = benchmark_config.mode
    
    def forward_pass():
        data = generate_random_data(batch_size, seq_length, vocab_size, device)
        logits = bench_model(data)
        return logits, data
    
    def forward_backward_pass():
        data = generate_random_data(batch_size, seq_length, vocab_size, device)
        logits = bench_model(data)
        # Shift logits and targets for next-token prediction
        loss = torch.nn.CrossEntropyLoss()(
            logits[:, :-1].reshape(-1, vocab_size),
            data[:, 1:].reshape(-1)
        )
        loss.backward()
        bench_model.zero_grad()
        return loss.item()
    
    run_fn = forward_pass if mode == "forward" else forward_backward_pass
    
    # Warmup
    for _ in range(num_warmup):
        run_fn()
        if device == "cuda":
            torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start = timeit.default_timer()
        run_fn()
        if device == "cuda":
            torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)
    
    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    
    print(f"Model: d_model={model_config.d_model}, layers={model_config.num_layers}, heads={model_config.num_heads}")
    print(f"Batch: {batch_size}, Seq Length: {seq_length}")
    print(f"Mode: {mode}")
    print(f"Warmup: {num_warmup}, Runs: {num_runs}")
    print(f"Mean time: {mean_time * 1000:.2f} ms")
    print(f"Std time: {std_time * 1000:.2f} ms")
    
    return {"mean": mean_time, "std": std_time, "times": times}


def main():
    parser = argparse.ArgumentParser(description="Benchmark transformer model")
    
    # Model configuration
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--context-length", type=int, default=256, help="Context length")
    parser.add_argument("--d-model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--d-ff", type=int, default=3072, help="Feed-forward dimension")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta")
    
    # Benchmark configuration
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=256, help="Sequence length")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--num-warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of timed runs")
    parser.add_argument("--mode", type=str, default="both", choices=["forward", "both"], 
                        help="'forward' for forward pass only, 'both' for forward + backward")
    
    args = parser.parse_args()
    
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )
    
    benchmark_config = BenchmarkConfig(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        device=args.device,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        mode=args.mode,
    )
    
    benchmark(benchmark_config=benchmark_config, model_config=model_config)


if __name__ == "__main__":
    main()
