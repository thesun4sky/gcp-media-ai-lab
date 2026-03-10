"""
Lab 07: GCP Vertex AI Workbench에서 Triton 실험 환경 세팅
L4 또는 A100 GPU 노트북 생성 및 벤치마크 자동 실행
"""

import argparse
import subprocess
import sys

PROJECT_ID = "gen-lang-client-0318067486"
REGION = "asia-northeast3"
NOTEBOOK_NAME = "mediaai-triton-lab"


STARTUP_SCRIPT = """#!/bin/bash
pip install torch torchvision triton --index-url https://download.pytorch.org/whl/cu121 -q
pip install google-cloud-videointelligence google-cloud-speech google-cloud-translate \
            google-cloud-storage google-cloud-aiplatform google-cloud-bigquery vertexai \
            numpy ffmpeg-python tqdm pyyaml -q
echo "✅ Media AI Lab 환경 설정 완료"
"""

BENCHMARK_NOTEBOOK = """
import torch
import subprocess

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 클론 또는 코드 복사 후 실행
# !python labs/lab07_pytorch_triton/benchmark.py --report
"""


def create_vertex_notebook(gpu_type: str = "NVIDIA_L4"):
    """Vertex AI Workbench 노트북 인스턴스 생성"""
    machine_types = {
        "NVIDIA_L4": ("g2-standard-8", "NVIDIA_L4", 1),
        "NVIDIA_A100_80GB": ("a2-ultragpu-1g", "NVIDIA_A100_80GB", 1),
        "NVIDIA_T4": ("n1-standard-8", "NVIDIA_T4", 1),
    }

    machine_type, accelerator, count = machine_types.get(gpu_type, machine_types["NVIDIA_L4"])

    print(f"🚀 Vertex AI Workbench 노트북 생성 중...")
    print(f"   GPU: {gpu_type}")
    print(f"   머신: {machine_type}")
    print(f"   지역: {REGION}")

    cmd = [
        "gcloud", "workbench", "instances", "create", NOTEBOOK_NAME,
        f"--project={PROJECT_ID}",
        f"--location={REGION}-a",
        f"--machine-type={machine_type}",
        f"--accelerator-type={gpu_type}",
        f"--accelerator-core-count={count}",
        "--install-gpu-driver",
        "--container-repository=us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2",
    ]

    print(f"\n실행 명령어:\n{' '.join(cmd)}\n")
    print("💡 위 명령어를 터미널에서 직접 실행하거나:")
    print("   GCP Console → Vertex AI → Workbench → 새 인스턴스")
    print(f"\n   예상 비용:")
    print(f"   - L4 GPU: 약 $0.7/시간")
    print(f"   - A100 80GB: 약 $3.5/시간")
    print(f"\n📌 실험 완료 후 반드시 인스턴스 중지 또는 삭제하세요!")

    return cmd


def check_local_triton():
    """로컬 Triton CUDA 지원 여부 확인"""
    import torch
    print("\n🔍 현재 환경 Triton 지원 확인")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    print(f"   MPS: {torch.backends.mps.is_available()}")

    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        try:
            import triton
            print(f"   ✅ Triton: {triton.__version__} (CUDA 커널 완전 지원)")
        except ImportError:
            print("   ⚠️  Triton 패키지 미설치 → pip install triton")
    elif torch.backends.mps.is_available():
        print("   ✅ MPS (Apple Silicon) → torch.compile inductor 백엔드 사용")
        print("   ℹ️  완전한 CUDA Triton은 GCP GPU 인스턴스에서 지원")
    else:
        print("   ℹ️  CPU 전용 → GCP GPU 인스턴스에서 실험 권장")

    print("\n📊 기대 성능 향상 (CUDA GPU 기준)")
    print("   ┌──────────────────────────┬─────────────┬──────────────────┐")
    print("   │ 실험                     │ PyTorch     │ Triton           │")
    print("   ├──────────────────────────┼─────────────┼──────────────────┤")
    print("   │ 프레임 정규화 (L4)       │ 기준        │ ~2-4x 향상       │")
    print("   │ 어텐션 Softmax (L4)      │ 기준        │ ~1.5-3x 향상     │")
    print("   │ FFN 블록 (A100)          │ 기준        │ ~2-5x 향상       │")
    print("   │ Temporal BMM (A100)      │ 기준        │ ~3-6x 향상       │")
    print("   │ 학습 스텝 (A100)         │ 기준        │ ~1.5-2.5x 향상   │")
    print("   └──────────────────────────┴─────────────┴──────────────────┘")


def main():
    parser = argparse.ArgumentParser(description="GCP Vertex AI Triton 환경 설정")
    parser.add_argument("--check", action="store_true", help="로컬 환경 확인")
    parser.add_argument("--create-notebook", action="store_true", help="Vertex AI 노트북 생성")
    parser.add_argument("--gpu", default="NVIDIA_L4",
                        choices=["NVIDIA_L4", "NVIDIA_A100_80GB", "NVIDIA_T4"],
                        help="GPU 타입 선택")
    args = parser.parse_args()

    if args.check or not args.create_notebook:
        check_local_triton()

    if args.create_notebook:
        create_vertex_notebook(args.gpu)


if __name__ == "__main__":
    main()
