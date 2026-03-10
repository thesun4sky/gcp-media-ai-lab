# Lab 07: PyTorch → Triton 커널 최적화 실험

## 개요
네이버 DAN25에서 발표한 Media AI 서비스는 대규모 동영상 처리를 위해 GPU 연산 효율을 극대화합니다.
이 랩에서는 **PyTorch 기본 연산 → Triton 최적화 커널**로 교체하여 실제 성능 향상을 측정합니다.

## Triton이란?
- OpenAI가 개발한 GPU 프로그래밍 언어 (Python 문법)
- PyTorch의 `torch.compile(backend="inductor")`의 내부 엔진
- CUDA 커널을 Python으로 작성 → 자동 최적화
- **Media AI 활용**: 영상 프레임 전처리, 어텐션 메커니즘, 정규화 연산

## 실험 구성

### 실험 1: 영상 프레임 정규화 커널
- PyTorch 기본 vs `torch.compile` (Triton inductor)
- 벤치마크: 처리 속도, 메모리 사용량

### 실험 2: Softmax 최적화
- 대규모 어텐션 스코어 계산 (Video Transformer)
- Flash Attention 유사 패턴 구현

### 실험 3: 커스텀 활성화 함수 (SiLU 계열)
- 영상 특징 추출 모델의 활성화 함수 최적화

### 실험 4: 배치 행렬 곱 (BMM) 최적화
- 동영상 시퀀스 처리용 배치 연산

## 실행 방법
```bash
# 기본 벤치마크 실행
python labs/lab07_pytorch_triton/benchmark.py

# 특정 실험만 실행
python labs/lab07_pytorch_triton/benchmark.py --experiment normalize
python labs/lab07_pytorch_triton/benchmark.py --experiment softmax
python labs/lab07_pytorch_triton/benchmark.py --experiment activation
python labs/lab07_pytorch_triton/benchmark.py --experiment bmm

# 전체 보고서 생성
python labs/lab07_pytorch_triton/benchmark.py --all --report
```

## 환경 안내
| 환경 | Triton 지원 | 최적화 백엔드 |
|------|------------|-------------|
| Linux + CUDA GPU | ✅ 완전 지원 | Triton CUDA 커널 |
| macOS Apple Silicon | ✅ MPS 지원 | Triton inductor + MPS |
| CPU only | ⚠️ 제한적 | inductor C++ 코드생성 |

> GCP Vertex AI Workbench (L4/A100)에서 실행 시 CUDA Triton 커널로 최대 성능 달성
