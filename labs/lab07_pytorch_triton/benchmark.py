"""
Lab 07: PyTorch → Triton 커널 최적화 벤치마크
[팀네이버 DAN25] Media AI 동영상 처리 성능 실험

torch.compile(backend="inductor")를 통해 Triton 커널로 자동 최적화.
CUDA 환경에서는 실제 Triton CUDA 커널, MPS에서는 Metal 최적화 적용.
"""

import argparse
import time
import json
import os
from pathlib import Path
from typing import Callable
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 장치 선택: CUDA > MPS > CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_NAME = f"CUDA ({torch.cuda.get_device_name(0)})"
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    DEVICE_NAME = "Apple MPS (Metal)"
else:
    DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"

print(f"🖥️  실행 장치: {DEVICE_NAME}")
print(f"   PyTorch 버전: {torch.__version__}")
print(f"   torch.compile 사용 가능: ✅\n")


# ─────────────────────────────────────────────
# 벤치마크 유틸리티
# ─────────────────────────────────────────────

def benchmark_fn(fn: Callable, *args, warmup: int = 5, repeat: int = 20) -> dict:
    """함수 실행 시간 측정 (워밍업 포함)"""
    # 워밍업
    for _ in range(warmup):
        result = fn(*args)
        if DEVICE.type != "cpu":
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

    # 실제 측정
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn(*args)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elif DEVICE.type == "mps":
            torch.mps.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    times = sorted(times)
    return {
        "mean_ms": float(np.mean(times)),
        "min_ms": float(np.min(times)),
        "median_ms": float(np.median(times)),
        "std_ms": float(np.std(times)),
    }


def speedup_label(baseline_ms: float, optimized_ms: float) -> str:
    ratio = baseline_ms / optimized_ms
    emoji = "🚀" if ratio > 1.5 else ("✅" if ratio > 1.1 else "➖")
    return f"{emoji} {ratio:.2f}x 빠름"


# ─────────────────────────────────────────────
# 실험 1: 영상 프레임 정규화
# Media AI에서 수백만 프레임을 GPU로 정규화할 때 핵심 연산
# ─────────────────────────────────────────────

class VideoFrameNormalizer(nn.Module):
    """영상 프레임 정규화 (ImageNet 평균/표준편차)"""
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, C, H, W) float32, 0~1 범위
        return (frames - self.mean.to(frames.device)) / self.std.to(frames.device)


def experiment_frame_normalize():
    """실험 1: 프레임 정규화 PyTorch vs Triton(compile)"""
    print("=" * 60)
    print("📹 실험 1: 영상 프레임 정규화 커널")
    print("   (B=32, C=3, H=224, W=224) 배치 기준")
    print("=" * 60)

    B, C, H, W = 32, 3, 224, 224
    frames = torch.rand(B, C, H, W, device=DEVICE)

    # PyTorch 기본
    model_baseline = VideoFrameNormalizer().to(DEVICE)
    model_baseline.eval()

    # Triton(inductor) 최적화
    model_compiled = torch.compile(
        VideoFrameNormalizer().to(DEVICE),
        backend="inductor",
        mode="max-autotune",  # 최대 자동 튜닝
    )
    model_compiled.eval()

    # 컴파일 트리거 (첫 실행에서 Triton 커널 생성)
    print("   🔧 Triton 커널 컴파일 중...")
    with torch.no_grad():
        _ = model_compiled(frames)
    print("   ✅ 컴파일 완료\n")

    with torch.no_grad():
        t_baseline = benchmark_fn(model_baseline, frames)
        t_compiled = benchmark_fn(model_compiled, frames)

    print(f"   PyTorch 기본:      {t_baseline['mean_ms']:.3f}ms (±{t_baseline['std_ms']:.3f})")
    print(f"   Triton(compile):   {t_compiled['mean_ms']:.3f}ms (±{t_compiled['std_ms']:.3f})")
    print(f"   성능 향상:         {speedup_label(t_baseline['mean_ms'], t_compiled['mean_ms'])}\n")

    return {
        "experiment": "frame_normalize",
        "shape": f"({B},{C},{H},{W})",
        "baseline_ms": t_baseline["mean_ms"],
        "triton_ms": t_compiled["mean_ms"],
        "speedup": t_baseline["mean_ms"] / t_compiled["mean_ms"],
    }


# ─────────────────────────────────────────────
# 실험 2: Softmax 최적화 (Video Transformer 어텐션)
# ─────────────────────────────────────────────

class VideoAttentionSoftmax(nn.Module):
    """Video Transformer의 어텐션 스코어 Softmax"""
    def forward(self, attn_scores: torch.Tensor) -> torch.Tensor:
        # attn_scores: (B, heads, seq_len, seq_len)
        return F.softmax(attn_scores, dim=-1)


def experiment_attention_softmax():
    """실험 2: 어텐션 Softmax PyTorch vs Triton"""
    print("=" * 60)
    print("🎬 실험 2: Video Transformer 어텐션 Softmax")
    print("   (B=8, heads=12, seq=196, seq=196) - ViT-B/16 기준")
    print("=" * 60)

    # ViT-B/16: 196 tokens (14x14 패치), 12 헤드
    B, heads, seq = 8, 12, 196
    attn_scores = torch.randn(B, heads, seq, seq, device=DEVICE)

    model_baseline = VideoAttentionSoftmax().to(DEVICE)
    model_compiled = torch.compile(
        VideoAttentionSoftmax().to(DEVICE),
        backend="inductor",
        mode="reduce-overhead",
    )

    print("   🔧 Triton 커널 컴파일 중...")
    with torch.no_grad():
        _ = model_compiled(attn_scores)
    print("   ✅ 컴파일 완료\n")

    with torch.no_grad():
        t_baseline = benchmark_fn(model_baseline, attn_scores)
        t_compiled = benchmark_fn(model_compiled, attn_scores)

    print(f"   PyTorch 기본:      {t_baseline['mean_ms']:.3f}ms")
    print(f"   Triton(compile):   {t_compiled['mean_ms']:.3f}ms")
    print(f"   성능 향상:         {speedup_label(t_baseline['mean_ms'], t_compiled['mean_ms'])}\n")

    return {
        "experiment": "attention_softmax",
        "shape": f"({B},{heads},{seq},{seq})",
        "baseline_ms": t_baseline["mean_ms"],
        "triton_ms": t_compiled["mean_ms"],
        "speedup": t_baseline["mean_ms"] / t_compiled["mean_ms"],
    }


# ─────────────────────────────────────────────
# 실험 3: SiLU 활성화 함수 최적화
# 영상 특징 추출 모델(ConvNeXT, EfficientNet)에서 사용
# ─────────────────────────────────────────────

class MediaFeatureExtractorBlock(nn.Module):
    """Media AI 특징 추출 블록 (Linear → SiLU → Linear)"""
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.fc2(self.act(self.fc1(x))))


def experiment_activation_block():
    """실험 3: 활성화 함수 블록 최적화"""
    print("=" * 60)
    print("⚡ 실험 3: Media AI 특징 추출 블록 (SiLU + FFN)")
    print("   (B=32, seq=196, dim=1024) - 영상 임베딩 기준")
    print("=" * 60)

    B, seq, dim = 32, 196, 1024
    x = torch.randn(B, seq, dim, device=DEVICE)

    model_baseline = MediaFeatureExtractorBlock(dim).to(DEVICE)
    model_baseline.eval()

    model_compiled = torch.compile(
        MediaFeatureExtractorBlock(dim).to(DEVICE),
        backend="inductor",
        mode="max-autotune",
        fullgraph=True,  # 전체 그래프 컴파일로 최대 최적화
    )
    model_compiled.eval()

    print("   🔧 Triton 커널 컴파일 중 (fullgraph 모드)...")
    with torch.no_grad():
        _ = model_compiled(x)
    print("   ✅ 컴파일 완료\n")

    with torch.no_grad():
        t_baseline = benchmark_fn(model_baseline, x)
        t_compiled = benchmark_fn(model_compiled, x)

    print(f"   PyTorch 기본:      {t_baseline['mean_ms']:.3f}ms")
    print(f"   Triton(compile):   {t_compiled['mean_ms']:.3f}ms")
    print(f"   성능 향상:         {speedup_label(t_baseline['mean_ms'], t_compiled['mean_ms'])}\n")

    return {
        "experiment": "activation_ffn_block",
        "shape": f"({B},{seq},{dim})",
        "baseline_ms": t_baseline["mean_ms"],
        "triton_ms": t_compiled["mean_ms"],
        "speedup": t_baseline["mean_ms"] / t_compiled["mean_ms"],
    }


# ─────────────────────────────────────────────
# 실험 4: 배치 행렬 곱 (BMM) 최적화
# 동영상 시퀀스의 temporal attention 계산
# ─────────────────────────────────────────────

class TemporalAttentionBMM(nn.Module):
    """시간축 어텐션 BMM (프레임 간 관계 계산)"""
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k, v: (B*heads, seq, head_dim)
        scale = q.size(-1) ** -0.5
        attn = torch.bmm(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        return torch.bmm(attn, v)


def experiment_temporal_bmm():
    """실험 4: Temporal Attention BMM"""
    print("=" * 60)
    print("🎞️  실험 4: 동영상 Temporal Attention BMM")
    print("   (B*heads=96, frames=32, head_dim=64)")
    print("=" * 60)

    # 32프레임 영상, 8헤드, 배치8 → 96개 병렬 연산
    BH, frames, head_dim = 96, 32, 64
    q = torch.randn(BH, frames, head_dim, device=DEVICE)
    k = torch.randn(BH, frames, head_dim, device=DEVICE)
    v = torch.randn(BH, frames, head_dim, device=DEVICE)

    model_baseline = TemporalAttentionBMM().to(DEVICE)
    model_compiled = torch.compile(
        TemporalAttentionBMM().to(DEVICE),
        backend="inductor",
        mode="max-autotune",
    )

    print("   🔧 Triton 커널 컴파일 중...")
    with torch.no_grad():
        _ = model_compiled(q, k, v)
    print("   ✅ 컴파일 완료\n")

    with torch.no_grad():
        t_baseline = benchmark_fn(model_baseline, q, k, v)
        t_compiled = benchmark_fn(model_compiled, q, k, v)

    print(f"   PyTorch 기본:      {t_baseline['mean_ms']:.3f}ms")
    print(f"   Triton(compile):   {t_compiled['mean_ms']:.3f}ms")
    print(f"   성능 향상:         {speedup_label(t_baseline['mean_ms'], t_compiled['mean_ms'])}\n")

    return {
        "experiment": "temporal_attention_bmm",
        "shape": f"({BH},{frames},{head_dim})",
        "baseline_ms": t_baseline["mean_ms"],
        "triton_ms": t_compiled["mean_ms"],
        "speedup": t_baseline["mean_ms"] / t_compiled["mean_ms"],
    }


# ─────────────────────────────────────────────
# 실험 5: 훈련 루프 전체 (Forward + Backward)
# 실제 Media AI 모델 학습 시 Triton 효과
# ─────────────────────────────────────────────

class SmallVideoModel(nn.Module):
    """소규모 Video Classification 모델"""
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.patch_embed = nn.Conv3d(3, 256, kernel_size=(2, 16, 16), stride=(2, 16, 16))
        self.norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.patch_embed(x)  # (B, 256, T', H', W')
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        x = self.norm(x)
        x = x.mean(dim=1)  # global average pooling
        return self.classifier(x)


def experiment_training_step():
    """실험 5: 모델 학습 스텝 (Forward + Backward)"""
    print("=" * 60)
    print("🧠 실험 5: Video Classification 학습 스텝")
    print("   (B=4, C=3, T=16, H=224, W=224) 영상 배치")
    print("=" * 60)

    # 영상 배치: 배치4, 3채널, 16프레임, 224x224
    B, C, T, H, W = 4, 3, 16, 224, 224
    x = torch.randn(B, C, T, H, W, device=DEVICE)
    labels = torch.randint(0, 100, (B,), device=DEVICE)

    model_baseline = SmallVideoModel().to(DEVICE)
    model_compiled = torch.compile(
        SmallVideoModel().to(DEVICE),
        backend="inductor",
        mode="max-autotune",
    )

    criterion = nn.CrossEntropyLoss()

    def train_step_baseline(x, labels):
        model_baseline.zero_grad()
        out = model_baseline(x)
        loss = criterion(out, labels)
        loss.backward()
        return loss.item()

    def train_step_compiled(x, labels):
        model_compiled.zero_grad()
        out = model_compiled(x)
        loss = criterion(out, labels)
        loss.backward()
        return loss.item()

    print("   🔧 Triton 커널 컴파일 중 (훈련 그래프)...")
    _ = train_step_compiled(x, labels)
    print("   ✅ 컴파일 완료\n")

    t_baseline = benchmark_fn(train_step_baseline, x, labels, warmup=3, repeat=10)
    t_compiled = benchmark_fn(train_step_compiled, x, labels, warmup=3, repeat=10)

    print(f"   PyTorch 기본:      {t_baseline['mean_ms']:.1f}ms/step")
    print(f"   Triton(compile):   {t_compiled['mean_ms']:.1f}ms/step")
    print(f"   성능 향상:         {speedup_label(t_baseline['mean_ms'], t_compiled['mean_ms'])}")
    throughput_baseline = B / (t_baseline["mean_ms"] / 1000)
    throughput_compiled = B / (t_compiled["mean_ms"] / 1000)
    print(f"   처리량 (기본):     {throughput_baseline:.1f} videos/sec")
    print(f"   처리량 (Triton):   {throughput_compiled:.1f} videos/sec\n")

    return {
        "experiment": "training_step_fwd_bwd",
        "shape": f"({B},{C},{T},{H},{W})",
        "baseline_ms": t_baseline["mean_ms"],
        "triton_ms": t_compiled["mean_ms"],
        "speedup": t_baseline["mean_ms"] / t_compiled["mean_ms"],
        "throughput_baseline_vps": throughput_baseline,
        "throughput_triton_vps": throughput_compiled,
    }


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────

EXPERIMENTS = {
    "normalize": experiment_frame_normalize,
    "softmax": experiment_attention_softmax,
    "activation": experiment_activation_block,
    "bmm": experiment_temporal_bmm,
    "training": experiment_training_step,
}


def print_summary(results: list):
    print("\n" + "=" * 60)
    print("📊 벤치마크 요약 보고서")
    print(f"   장치: {DEVICE_NAME}")
    print("=" * 60)
    print(f"{'실험':<30} {'기본(ms)':>10} {'Triton(ms)':>12} {'향상':>10}")
    print("-" * 60)
    for r in results:
        speedup = r["speedup"]
        emoji = "🚀" if speedup > 1.5 else ("✅" if speedup > 1.1 else "➖")
        print(
            f"{r['experiment']:<30} "
            f"{r['baseline_ms']:>10.3f} "
            f"{r['triton_ms']:>12.3f} "
            f"{emoji}{speedup:>7.2f}x"
        )
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print("-" * 60)
    print(f"{'평균 성능 향상':<30} {'':>10} {'':>12} {'⚡':>3}{avg_speedup:>7.2f}x")
    print("\n💡 GCP Vertex AI (L4/A100 GPU)에서 실행 시 더 큰 성능 향상 기대")
    print("   → CUDA Triton 커널: 일반적으로 2~5x 향상")


def save_report(results: list, output_dir: str = "./outputs"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report = {
        "device": DEVICE_NAME,
        "pytorch_version": torch.__version__,
        "backend": "triton/inductor",
        "results": results,
        "avg_speedup": sum(r["speedup"] for r in results) / len(results),
    }
    path = Path(output_dir) / "lab07_triton_benchmark.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n💾 보고서 저장: {path}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch → Triton 벤치마크")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS.keys()),
                        help="실행할 실험 선택")
    parser.add_argument("--all", action="store_true", default=True,
                        help="모든 실험 실행 (기본값)")
    parser.add_argument("--report", action="store_true", help="JSON 보고서 저장")
    args = parser.parse_args()

    print("\n🔬 [DAN25 Media AI] PyTorch → Triton 최적화 실험")
    print(f"   torch.compile backend: inductor (Triton 기반)")
    print()

    results = []

    if args.experiment:
        result = EXPERIMENTS[args.experiment]()
        results.append(result)
    else:
        for name, fn in EXPERIMENTS.items():
            try:
                result = fn()
                results.append(result)
            except Exception as e:
                print(f"⚠️  {name} 실험 건너뜀: {e}\n")

    if results:
        print_summary(results)
        if args.report:
            save_report(results)


if __name__ == "__main__":
    main()
