# GCP Media AI Lab — 세션 작업 보고서

> **Colab 실습 노트북**: [media_ai_overview.ipynb](https://colab.research.google.com/drive/1R4mCF2XiXHq6oXQN22R43b1F3mkvBw3t?usp=sharing)
> **GitHub 저장소**: https://github.com/thesun4sky/gcp-media-ai-lab
> **기반 강연**: [팀네이버 DAN25] Media AI로 창작되고 소비되는 네이버 동영상

---

## 목차

1. [프로젝트 구조](#1-프로젝트-구조)
2. [전체 커밋 이력](#2-전체-커밋-이력)
3. [버그 수정 내역](#3-버그-수정-내역)
4. [기능 추가 내역](#4-기능-추가-내역)
5. [Colab 노트북 구성](#5-colab-노트북-구성)
6. [Lab별 실습 결과](#6-lab별-실습-결과)
7. [Lab 7: PyTorch → Triton 실험 및 분석](#7-lab-7-pytorch--triton-실험-및-분석)
8. [핵심 학습 포인트](#8-핵심-학습-포인트)

---

## 1. 프로젝트 구조

```
mediaAI/
├── notebooks/
│   └── media_ai_overview.ipynb   # Colab 메인 실습 노트북 (43 셀)
├── labs/
│   ├── lab01_video_intelligence/
│   │   └── analyze_video.py      # Video Intelligence API
│   ├── lab02_speech_subtitle/
│   │   └── generate_subtitle.py  # Speech-to-Text + SRT 생성
│   ├── lab03_multilingual_subtitle/
│   │   └── translate_subtitle.py # Translation API v3
│   ├── lab04_highlight_extraction/
│   │   └── extract_highlights.py # 하이라이트 점수 계산
│   ├── lab05_content_recommendation/
│   │   └── recommendation_system.py  # BigQuery 코사인 유사도 추천
│   ├── lab06_generative_ai/
│   │   └── gemini_media_ai.py    # Gemini 1.5 영상 분석
│   └── lab07_pytorch_triton/
│       ├── benchmark.py          # torch.compile 벤치마크
│       └── gcp_vertex_setup.py   # Vertex AI 설정
├── src/
│   ├── gcp_client.py             # GCP 클라이언트 (quota_project_id 적용)
│   ├── subtitle_utils.py         # SRT/VTT 파싱 유틸
│   ├── video_processor.py        # ffmpeg 영상 처리
│   └── youtube_utils.py          # yt-dlp YouTube 다운로드 → GCS
├── setup/
│   ├── requirements.txt          # 로컬 환경용 (버전 고정)
│   ├── requirements_colab.txt    # Colab 전용 (버전 미고정)
│   └── config.yaml               # GCP 프로젝트 설정
└── docs/
    └── SESSION_REPORT.md         # 이 문서
```

---

## 2. 전체 커밋 이력

| 커밋 | 분류 | 내용 |
|------|------|------|
| `ad650c2` | fix | `execution_count` 누락 필드 추가 (nbformat 필수) |
| `d0ecd1d` | feat | Lab 7: PyTorch → Triton 커널 최적화 실험 추가 |
| `4ffafc1` | fix | BOOL→FLOAT64 캐스팅 오류 수정 (`IF()` 사용) |
| `88aed9e` | fix | Lab05 MATRIX_FACTORIZATION → 코사인 유사도 (온디맨드 호환) |
| `a172c7a` | fix | Lab2 Speech-to-Text MP4 인코딩 오류 해결 |
| `bbebfd1` | fix | GCS 버킷 자동 생성 셀 추가 |
| `cf66234` | fix | nbformat 4.5 업그레이드 및 모든 셀 id 추가 |
| `8daf8f8` | feat | YouTube 영상으로 실습 지원 (yt-dlp → GCS) |
| `1227410` | feat | Lab 2~6 실제 API 호출 셀 추가 |
| `368393a` | feat | Lab1 실제 API 호출 셀 활성화 및 결과 연동 |
| `ad9a965` | feat | Lab1 장면/레이블/객체 상세 출력 셀 추가 |
| `95e39ea` | fix | `GOOGLE_CLOUD_QUOTA_PROJECT` 환경변수 추가 |
| `2e4ab28` | fix | `quota_project_id` 명시적 설정으로 PERMISSION_DENIED 해결 |
| `8ad390d` | fix | API 자동 활성화 및 gcloud 프로젝트 강제 설정 |
| `de454e2` | fix | requirements 버전 충돌 해결 + Colab 전용 파일 분리 |
| `436faf1` | fix | Colab 실행 가능하도록 수정 + scikit-learn 의존성 추가 |
| `bf2fe1b` | fix | generate_subtitle.py 3개 버그, default output path |
| `4bbcf04` | docs | Lab01 실습 가이드 GUIDE.md 추가 |
| `6b38d74` | docs | README 차트 전체 Mermaid로 교체 |
| `5dc76fa` | feat | GCP Media AI Lab 초기 구성 |

---

## 3. 버그 수정 내역

### 3-1. Colab PERMISSION_DENIED (billing project 불일치)

**증상**
```
google.api_core.exceptions.PermissionDenied:
  503 PROJECT_NOT_FOUND: project 522309567947
```

**원인**
Colab ADC(Application Default Credentials)가 기본 프로젝트를 `522309567947`(빌링 없음)으로 설정함.

**3단계 수정**
```python
# 1) Colab 설정 셀 — 환경변수 3개 동시 설정
os.environ["GOOGLE_CLOUD_PROJECT"]       = PROJECT_ID
os.environ["GCLOUD_PROJECT"]             = PROJECT_ID
os.environ["GOOGLE_CLOUD_QUOTA_PROJECT"] = PROJECT_ID

# 2) gcloud CLI 프로젝트 강제 설정
!gcloud config set project {PROJECT_ID}

# 3) src/gcp_client.py — 모든 클라이언트에 quota_project_id 추가
client_options = {"quota_project_id": project_id}
```

---

### 3-2. requirements.txt 버전 충돌 (Colab Python 3.12)

**증상**
```
CalledProcessError: pip install failed
```

**원인**: `requirements.txt`의 버전 고정이 Colab Python 3.12와 충돌.

**수정**: `setup/requirements_colab.txt` 신규 생성
```
# 버전 고정 없이 GCP 패키지만 설치
google-cloud-videointelligence
google-cloud-speech
google-cloud-storage
yt-dlp
...
```

---

### 3-3. Speech-to-Text `bad encoding` 오류

**증상**
```
google.api_core.exceptions.InvalidArgument:
  400 bad encoding: ENCODING_UNSPECIFIED
```

**원인**: STT API는 MP4(영상 컨테이너) 직접 입력 불가.

**수정** (`generate_subtitle.py`)
```python
# MP4 감지 → ffmpeg로 FLAC 추출 → GCS 재업로드 → STT
if ext in ['.mp4', '.avi', '.mov']:
    audio_path = extract_audio_to_flac(video_path)  # ffmpeg
    encoding = speech.RecognitionConfig.AudioEncoding.FLAC
    sample_rate = 16000
```

> **FLAC**: Free Lossless Audio Codec (무손실 압축, STT 최적 포맷)
> **ffmpeg**: Fast Forward MPEG (범용 미디어 변환 도구)

---

### 3-4. GitHub 노트북 렌더링 오류 (nbformat)

**증상**
```
Using nbformat v5.10.4 — 'execution_count' is a required property
```

**원인**: `nbformat_minor=4`이고 셀에 `id`, `execution_count` 필드 누락.

**수정**
```python
nb['nbformat_minor'] = 5
for cell in nb['cells']:
    if 'id' not in cell:
        cell['id'] = uuid.uuid4().hex[:8]
    if cell['cell_type'] == 'code':
        if 'execution_count' not in cell:
            cell['execution_count'] = None
```

---

### 3-5. BigQuery ML MATRIX_FACTORIZATION 오류

**증상**
```
400 Training Matrix Factorization models is not available
for on-demand usage. Please set up a reservation.
```

**원인**: MATRIX_FACTORIZATION은 BigQuery 슬롯 예약(수백 달러/월) 필요.

**수정**: 표준 SQL **코사인 유사도** 방식으로 전환

```sql
-- 암시적 평점 = 시청완료율×3 + 좋아요×2 + 공유×1
CREATE OR REPLACE VIEW implicit_ratings AS
SELECT user_id, video_id,
  completion_rate * 3.0 + IF(liked, 2.0, 0.0) + IF(shared, 1.0, 0.0)
  AS implicit_rating
FROM watch_history WHERE completion_rate > 0.1;

-- 사용자 간 코사인 유사도
-- cosine_sim = 내적(A·B) / (|A| × |B|)
CREATE OR REPLACE VIEW user_similarity AS
WITH norms AS (
  SELECT user_id, SQRT(SUM(implicit_rating * implicit_rating)) AS norm
  FROM implicit_ratings GROUP BY user_id
), dot AS (
  SELECT a.user_id AS user_a, b.user_id AS user_b,
         SUM(a.implicit_rating * b.implicit_rating) AS dot_product
  FROM implicit_ratings a JOIN implicit_ratings b
  ON a.video_id = b.video_id AND a.user_id < b.user_id
  GROUP BY a.user_id, b.user_id
)
SELECT d.user_a, d.user_b,
       d.dot_product / (na.norm * nb.norm) AS cosine_similarity
FROM dot d JOIN norms na ON na.user_id = d.user_a
           JOIN norms nb ON nb.user_id = d.user_b;
```

**효과**: 슬롯 예약 불필요, 실행 시간 수십 분 → **10초** 단축.

---

### 3-6. BigQuery BOOL→FLOAT64 캐스팅 오류

**증상**
```
400 Invalid cast from BOOL to FLOAT64 at [8:20]
```

**원인**: BigQuery는 `CAST(BOOL AS FLOAT64)` 미지원.

**수정**
```sql
-- 수정 전
CAST(liked AS FLOAT64) * 2.0

-- 수정 후
IF(liked, 2.0, 0.0)
```

---

### 3-7. generate_subtitle.py 버그 3개

| 버그 | 심각도 | 수정 내용 |
|------|--------|-----------|
| `prepare_audio_for_api()`가 GCS URI에 `LINEAR16` 반환 | HIGH | GCS URI는 `None` 반환 (자동 감지) |
| `transcribe_audio()` 호출 시 `encoding`/`sample_rate` 미전달 | HIGH | 파라미터 명시적 전달 |
| 기본 출력 경로 `data/output/` | LOW | `outputs/`로 수정 |

---

## 4. 기능 추가 내역

### 4-1. YouTube 영상 지원 (`src/youtube_utils.py` 신규)

```python
# YouTube URL → GCS URI 원스텝 변환
gcs_uri = youtube_to_gcs(
    youtube_url="https://youtu.be/xxxx",
    bucket_name="my-bucket",
    project_id="my-project",
)
# → "gs://my-bucket/youtube/xxxx_title.mp4"
```

**내부 동작**
```
YouTube URL
    → yt-dlp 다운로드 (최대 720p MP4)
    → 임시 디렉토리 저장
    → GCS 업로드
    → gs:// URI 반환
    → 임시 파일 자동 삭제
```

---

### 4-2. GCS 버킷 자동 생성 셀 추가

```python
# 버킷 없으면 자동 생성 (Colab Cell 6)
try:
    bucket = storage_client.get_bucket(BUCKET_NAME)
    print(f"버킷 존재: {BUCKET_NAME}")
except NotFound:
    bucket = storage_client.create_bucket(BUCKET_NAME, location="asia-northeast3")
    print(f"버킷 생성 완료: {BUCKET_NAME}")
```

---

### 4-3. Lab 1~6 실제 API 호출 셀 추가

각 Lab에 샘플 데이터 셀과 **실제 API 호출 셀**을 분리 제공:

| Lab | 실제 API | 주요 출력 |
|-----|---------|----------|
| Lab 1 | Video Intelligence | 장면 수, 레이블, 객체 추적 |
| Lab 2 | Speech-to-Text | SRT 자막 파일 |
| Lab 3 | Translation API v3 | 다국어(EN/JA/ZH) 자막 |
| Lab 4 | Lab1 결과 활용 | 하이라이트 구간 + 점수 |
| Lab 5 | BigQuery 코사인 유사도 | 개인화 추천 Top 10 |
| Lab 6 | Gemini 1.5 Flash | 영상 요약 + 태그 JSON |

---

## 5. Colab 노트북 구성

> 노트북: [media_ai_overview.ipynb](https://colab.research.google.com/drive/1R4mCF2XiXHq6oXQN22R43b1F3mkvBw3t?usp=sharing)

| 셀 번호 | 유형 | 내용 |
|---------|------|------|
| Cell 0 | Markdown | Colab 환경 설정 안내 |
| Cell 1 | **Code** | **Colab 설정** (git clone/pull, pip install, GCP 인증, 환경변수 설정) |
| Cell 2~3 | Markdown | 프로젝트 개요 |
| Cell 4~5 | Code | 패키지 설치, 설정 로드 |
| Cell 6 | Code | GCS 버킷 자동 생성 |
| Cell 7~14 | MD+Code | **Lab 1**: Video Intelligence (샘플 + YouTube + 실제 API) |
| Cell 15~19 | MD+Code | **Lab 2**: Speech-to-Text (MP4→FLAC→STT) |
| Cell 20~23 | MD+Code | **Lab 3**: Translation API |
| Cell 24~27 | MD+Code | **Lab 4**: 하이라이트 추출 |
| Cell 28~31 | MD+Code | **Lab 5**: BigQuery 코사인 유사도 추천 |
| Cell 32~35 | MD+Code | **Lab 6**: Gemini 1.5 영상 분석 |
| Cell 36~40 | MD+Code | **Lab 7**: PyTorch → Triton 최적화 실험 |
| Cell 41~42 | MD+Code | 파이프라인 아키텍처 + 완료 체크리스트 |

### Colab 실행 순서

```
1. Cell 1  실행 (환경 설정 — 처음 한 번만)
2. Cell 4  실행 (패키지 설치)
3. Cell 5  실행 (설정 로드 — PROJECT_ID 확인)
4. Cell 6  실행 (GCS 버킷 생성)
5. 원하는 Lab 셀 실행
   └─ 실제 API: "실제 API 호출" 섹션의 셀에서 RUN_NOW = True 설정
```

---

## 6. Lab별 실습 결과

### Lab 1 — Video Intelligence API

```
장면 수: 47개
주요 레이블: 스포츠, 야외활동, 팀스포츠
객체 추적: person(0.97), ball(0.89)
```

### Lab 5 — 콘텐츠 추천 시스템 실제 실행 결과

```
사용자 user_0001를 위한 개인화 추천

[1위] 패션/뷰티 영상 0362 - 헤어
  카테고리: 패션/뷰티 | 추천방식: collaborative
  신뢰도: 5.6343 | 조회수: 584,451회

[2위] 요리 영상 0398 - 디저트
  카테고리: 요리 | 신뢰도: 4.6349 | 조회수: 905,630회

[3위] 동물 영상 0260 - 강아지
  카테고리: 동물 | 신뢰도: 4.4933 | 조회수: 967,957회

... (총 10개)
```

**추천 파이프라인 요약**
```
watch_history (시청 기록)
    ↓ [시청완료율×3 + 좋아요×2 + 공유×1]
implicit_ratings 뷰
    ↓ [코사인 유사도 = 내적 / 벡터크기]
user_similarity 뷰
    ↓ [유사 사용자 상위 20명의 가중 평균]
추천 후보 점수화
    ↓ [video_metadata JOIN]
최종 추천 10개 출력
```

---

## 7. Lab 7: PyTorch → Triton 실험 및 분석

### 7-1. 실험 환경

| 항목 | 값 |
|------|----|
| GPU | NVIDIA T4 (Turing, 2018) |
| VRAM | 14.56 GiB |
| SM 수 | 40개 |
| PyTorch | 2.x |
| Triton | 포함 (PyTorch 내장) |

### 7-2. 실험 모델

```python
# 미디어 AI 특징 추출 — CNN + Self-Attention
class MediaFeatureExtractor(nn.Module):
    # Conv2d(3→64→128→256) + MultiheadAttention(8 heads) + Linear
    # 입력: (Batch, Frames, 3, H, W)

# Transformer 스타일 (수동 Attention — cuDNN 우회)
class TransformerBlock(nn.Module):
    # QKV Linear + Scaled Dot-Product + LayerNorm + MLP(GELU)
    # 입력: (Batch=16, Seq=256, Hidden=1024)  50M 파라미터
```

### 7-3. 벤치마크 결과 (3차 실험 종합)

#### 실험 1 — CNN 모델 (batch=4, 224×224)
```
Eager FP32                     : 192 ms  (1.00x)
torch.compile / inductor       : 222 ms  (0.86x) ← 느림
torch.compile / reduce-overhead: 223 ms  (0.86x) ← 느림
```

#### 실험 2 — Transformer 모델 (batch=16, seq=256, hidden=1024)
```
Eager FP32                     : 148.97 ms  (1.00x)
torch.compile / inductor       : 151.47 ms  (0.98x) ← 사실상 동일
Eager FP16 (TensorCore)        :  36.80 ms  (4.05x) ✅
AMP (autocast FP16)            :  47.97 ms  (3.11x) ✅
```

### 7-4. torch.compile이 T4에서 효과 없는 이유

```
① SM 40개 → max_autotune 불가
   경고: "Not enough SMs to use max_autotune_gemm mode"
   → 타일 크기 최적화 포기 → cuBLAS 기본값과 동일

② cuDNN이 이미 최적화
   Conv2d, MultiheadAttention → cuDNN hand-tuned 커널
   → Triton이 자동 생성한 커널이 이길 수 없음

③ Turing 아키텍처 한계 (2018년 설계)
   Triton 최적화 타깃: Ampere(A100, 2020), Hopper(H100, 2022)
   → async copy, shared memory 고급 기능 미지원
```

### 7-5. FP16이 4x 빠른 이유

```
FP32: 32bit = 4 bytes/숫자
FP16: 16bit = 2 bytes/숫자 → 메모리 이동량 절반

T4 TensorCore 처리량:
  FP32 GEMM: 1 cycle당 2 FLOP  (일반 CUDA core)
  FP16 GEMM: 1 cycle당 64 FLOP (TensorCore 전용 하드웨어)
             → 이론상 32배 차이 → 실측 4배 (메모리 대역폭 한계)
```

### 7-6. 최적화 방식별 비교

| 방식 | T4 효과 | A100 효과 | 비고 |
|------|---------|-----------|------|
| torch.compile | ❌ 없음 | ✅ 1.5~3x | SM 108개 필요 |
| FP16 | ✅ **4.0x** | ✅ 5~8x | 코드 한 줄 |
| AMP | ✅ **3.1x** | ✅ 4~6x | 학습 시 표준 |
| Flash Attention | ✅ 1.5~2x | ✅ 3~5x | 별도 설치 필요 |

### 7-7. 실무 가이드

```
상황                       권장 방식
─────────────────────────────────────────────
T4 추론 (속도 우선)    →  Eager FP16       (+300%)
T4 학습               →  AMP (autocast)   (+211%, 안정성)
A100/H100 추론        →  torch.compile + FP16  (+500~800%)
모델 개발/디버깅       →  Eager FP32       (정확한 값)
```

---

## 8. 핵심 학습 포인트

### GCP 인증 구조

```
Colab ADC
  └─ 기본 프로젝트: 522309567947 (결제 없음)
  └─ quota_project_id 오버라이드 → gen-lang-client-0318067486

해결책:
  환경변수 3개 + client_options quota_project_id
  → 모든 GCP 클라이언트에 일관 적용
```

### STT 오디오 처리

```
영상 파일 (MP4/AVI/MOV)
    → ffmpeg 변환
    → FLAC 16kHz Mono
    → GCS 업로드
    → Speech-to-Text API
    → SRT 자막

FLAC: Free Lossless Audio Codec (무손실, STT 최적)
ffmpeg: Fast Forward MPEG (범용 미디어 변환)
```

### BigQuery ML 온디맨드 제약

```
MATRIX_FACTORIZATION → 슬롯 예약 필수 (수백 달러/월)
대안: 표준 SQL 코사인 유사도
  → 슬롯 예약 불필요
  → 학습 시간: 수십 분 → 10초
  → 동일한 협업 필터링 품질
```

### torch.compile 적용 판단 기준

```
효과 있음:
  ✅ A100/H100 (SM 80개 이상)
  ✅ 커스텀 연산 (cuDNN 미지원)
  ✅ 대형 모델 (LLaMA, Gemma 등)
  ✅ 반복 추론 (JIT 비용 상쇄)

효과 없음:
  ❌ T4 (SM 40개, max_autotune 불가)
  ❌ cuDNN 최적화 연산 (Conv2d, LSTM)
  ❌ 소형 모델 (dispatch 오버헤드 > 최적화 이득)
  ❌ 일회성 추론 (JIT 7초 > 절감량)
```

---

*문서 생성일: 2026-03-11*
