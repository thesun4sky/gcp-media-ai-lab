# GCP Media AI Lab — 기술 용어 및 개념 완전 정리

> 이 문서는 실습에 등장한 모든 기술, 용어, 모델, 연산식을 초보자도 이해할 수 있도록 상세하게 정리합니다.
> 수학 수식은 기호 대신 말로 풀어서 설명하며, 각 개념마다 비교군을 함께 정리합니다.

---

## 목차

1. [GCP 서비스](#1-gcp-서비스)
2. [미디어 처리 기술](#2-미디어-처리-기술)
3. [인증 및 보안](#3-인증-및-보안)
4. [추천 시스템 — 유사도 연산 포함](#4-추천-시스템--유사도-연산-포함)
5. [딥러닝 기초](#5-딥러닝-기초)
6. [신경망 구조 비교 — CNN vs ANN vs RNN vs Transformer](#6-신경망-구조-비교--cnn-vs-ann-vs-rnn-vs-transformer)
7. [유사도 연산 상세 비교](#7-유사도-연산-상세-비교)
8. [PyTorch 최적화 기술](#8-pytorch-최적화-기술)
9. [CUDA와 Triton — 유사성·연관성·차이점 완전 정리](#9-cuda와-triton--유사성연관성차이점-완전-정리)
10. [GPU 하드웨어 개념](#10-gpu-하드웨어-개념)
11. [데이터 포맷 및 파일 형식](#11-데이터-포맷-및-파일-형식)
12. [Python 생태계 도구](#12-python-생태계-도구)

---

## 1. GCP 서비스

### 1-1. Google Cloud Video Intelligence API

**무엇인가?**
영상 파일을 입력하면 AI가 자동으로 영상 내용을 분석해주는 클라우드 서비스.

**할 수 있는 것들**

| 기능 | 설명 | 예시 출력 |
|------|------|-----------|
| 장면 전환 감지 | 영상이 다른 장면으로 바뀌는 시점 감지 | `0:00~0:05`, `0:05~0:12` |
| 레이블 감지 | 영상에 등장하는 개념/사물 분류 | `스포츠(0.97)`, `야외(0.89)` |
| 객체 추적 | 특정 물체가 화면 어디에 있는지 시간별 추적 | `person: 좌상단 0.2~0.8초` |
| 명시적 콘텐츠 감지 | 성인 콘텐츠 여부 판별 | `VERY_UNLIKELY` |
| 텍스트 감지 | 영상 속 텍스트(간판, 자막 등) 인식 | `"NAVER"` |

**동작 원리 (내부)**
```
영상 파일 (GCS)
    → NVIDIA GPU 수천 개로 구성된 Google 데이터센터
    → 사전 훈련된 CNN 모델이 프레임별 분석
    → 결과를 JSON으로 반환
```

**신뢰도 점수 (confidence)**
- 0.0 ~ 1.0 사이 값
- 0.9 이상: 매우 확실
- 0.7 이상: 일반적으로 신뢰 가능
- 0.5 미만: 불확실

**비교군: 유사 서비스**

| 서비스 | 공급자 | 특징 | 차이점 |
|--------|--------|------|--------|
| Video Intelligence API | Google Cloud | 장면/레이블/객체 추적 강점 | BigQuery 연동 용이 |
| Amazon Rekognition | AWS | 얼굴 인식 강점 | S3 연동 자연스러움 |
| Azure Video Analyzer | Microsoft | Azure 생태계 통합 | Teams/Office 연동 |
| OpenCV | 오픈소스 | 직접 구현, 무료 | 학습된 모델 없음, 코딩 필요 |

---

### 1-2. Google Cloud Speech-to-Text API

**무엇인가?**
음성 파일을 입력하면 텍스트로 변환(받아쓰기)해주는 서비스.

**지원 형식**
```
입력 가능: FLAC, WAV, MP3, OGG, OPUS
입력 불가: MP4, AVI, MOV  ← 영상 컨테이너이므로 불가
```

**왜 MP4가 안 되는가?**
```
MP4 = 영상(H.264) + 음성(AAC) + 자막을 하나로 묶은 "컨테이너"
              ↑ 여기서 음성만 꺼내야 함
STT API는 순수 음성 데이터만 받을 수 있음
∴ ffmpeg으로 음성 추출 → FLAC 변환 필요
```

**주요 설정값**
```python
RecognitionConfig(
    encoding=FLAC,           # 오디오 인코딩 방식
    sample_rate_hertz=16000, # 초당 샘플 수 (16kHz = 전화 음질)
    language_code="ko-KR",   # 언어 (한국어)
    enable_word_time_offsets=True,  # 단어별 시작/종료 시간
    enable_automatic_punctuation=True,  # 자동 구두점
)
```

**샘플레이트란?**
```
44,100 Hz = CD 음질 (초당 44,100번 소리 측정)
16,000 Hz = 전화 음질 (STT에 충분)
8,000  Hz = AM 라디오 음질

STT는 16kHz로도 충분히 정확함 → 파일 크기 절약
```

**비교군: STT 서비스**

| 서비스 | 한국어 정확도 | 실시간 지원 | 가격 |
|--------|--------------|------------|------|
| Google STT | 매우 높음 | 지원 | 분당 과금 |
| OpenAI Whisper | 높음 | 로컬 실행 가능 | 무료(오픈소스) |
| Naver CLOVA Speech | 높음(한국어 특화) | 지원 | 건당 과금 |
| Amazon Transcribe | 높음 | 지원 | 분당 과금 |

---

### 1-3. Translation API v3

**무엇인가?**
텍스트를 한 언어에서 다른 언어로 번역하는 서비스. Google 번역기와 같은 기술 기반.

**언어 코드**
```
ko = 한국어
en = 영어
ja = 일본어
zh = 중국어 (간체)
zh-TW = 중국어 (번체)
```

**v2 vs v3 차이**
```
v2: 텍스트 번역만 가능
v3: 텍스트 + HTML + 문서(PDF, DOCX) 번역 가능
    → 자막 파일(.srt)의 타임코드는 유지하고 텍스트만 번역
```

---

### 1-4. Vertex AI / Gemini

**Vertex AI란?**
Google Cloud의 머신러닝 플랫폼. 모델 학습, 배포, API 호출을 통합 관리.

**Gemini란?**
Google이 개발한 멀티모달(텍스트+이미지+영상+음성) AI 모델 시리즈.

| 모델 | 특징 | 사용 사례 |
|------|------|-----------|
| Gemini 1.5 Flash | 빠르고 저렴 | 요약, 태그 생성, 간단한 분석 |
| Gemini 1.5 Pro | 정확하고 긴 컨텍스트 | 복잡한 분석, 긴 문서 처리 |
| Gemini Ultra | 최고 성능 | 연구/복잡한 추론 |

**멀티모달이란?**
```
일반 AI: 텍스트만 이해
멀티모달: 텍스트 + 이미지 + 영상 + 음성 동시 이해

예시: "이 영상의 주요 장면을 설명해줘"
  → 영상 프레임을 직접 보고 텍스트로 설명
```

**컨텍스트 윈도우**
```
Gemini 1.5 Pro: 최대 2,000,000 토큰 처리 가능
= 책 약 1,500권 분량
= 영상 약 11시간 분량 (1 FPS 기준)
```

---

### 1-5. BigQuery

**무엇인가?**
Google Cloud의 완전관리형 데이터 웨어하우스. 수 테라바이트의 데이터를 SQL로 수 초 안에 분석.

**일반 DB vs BigQuery**
```
일반 DB (MySQL, PostgreSQL):
  - 수백만 행까지 빠름
  - 행 단위 업데이트/삭제 효율적
  - OLTP (온라인 트랜잭션 처리)

BigQuery:
  - 수백억 행도 수 초
  - 열 단위 저장 (컬럼형 스토리지)
  - OLAP (온라인 분석 처리)
  - 업데이트/삭제 비효율 (배치 처리 설계)
```

**BigQuery ML이란?**
SQL 문법으로 머신러닝 모델을 학습/예측하는 기능.
```sql
-- SQL로 모델 학습
CREATE MODEL dataset.my_model
OPTIONS(model_type = 'linear_reg') AS
SELECT feature1, feature2, label FROM my_table;

-- SQL로 예측
SELECT * FROM ML.PREDICT(MODEL dataset.my_model, (SELECT ...))
```

**MATRIX_FACTORIZATION이 온디맨드에서 안 되는 이유**
```
온디맨드 = 쿼리 실행할 때만 Google 서버 사용
슬롯 예약 = 전용 컴퓨팅 파워 24시간 할당 (수백 달러/월)

행렬 분해는 수백 번 반복 계산 필요
→ 온디맨드로는 비용 예측 불가 → Google이 차단
→ 슬롯 예약 시에만 허용
```

---

### 1-6. Google Cloud Storage (GCS)

**무엇인가?**
파일을 저장하는 클라우드 스토리지. AWS S3와 유사.

**개념**
```
버킷(Bucket) = 최상위 폴더 (전 세계적으로 유일한 이름)
객체(Object) = 버킷 안의 파일

URI 형식: gs://버킷이름/경로/파일명
예시: gs://my-media-bucket/youtube/video_abc123.mp4
```

**스토리지 클래스**
```
Standard  : 자주 접근 (기본값)
Nearline  : 월 1회 미만 접근 (30일 최소 보관)
Coldline  : 분기 1회 미만 접근 (90일 최소 보관)
Archive   : 연 1회 미만 접근 (365일 최소 보관, 가장 저렴)
```

---

## 2. 미디어 처리 기술

### 2-1. ffmpeg

**정식 명칭**: Fast Forward MPEG

**무엇인가?**
오디오/비디오 변환, 편집, 스트리밍을 처리하는 오픈소스 도구. 미디어 처리의 사실상 표준.

**주요 사용 예시**
```bash
# MP4에서 음성만 추출 → FLAC 변환
ffmpeg -i input.mp4 -vn -ac 1 -ar 16000 output.flac
#      -vn: 영상 제거   -ac 1: 모노(1채널)   -ar 16000: 샘플레이트

# 영상 해상도 변환
ffmpeg -i input.mp4 -vf scale=1280:720 output.mp4

# 특정 구간만 추출 (10초~30초)
ffmpeg -i input.mp4 -ss 10 -to 30 output.mp4
```

**경쟁 도구 비교**
| 도구 | 특징 | 단점 |
|------|------|------|
| ffmpeg | 무료, 거의 모든 형식 지원 | CLI 명령어 복잡 |
| HandBrake | GUI 지원, 쉬운 사용 | 배치 처리 제한 |
| GStreamer | 실시간 스트리밍 강점 | 설정 복잡 |
| MoviePy | Python 코드로 제어 | ffmpeg 기반 (래퍼) |

---

### 2-2. FLAC

**정식 명칭**: Free Lossless Audio Codec

**무엇인가?**
음질 손상 없이 파일 크기를 줄이는 무손실 오디오 압축 포맷.

**손실 vs 무손실 압축**
```
원본 오디오: 100MB

손실 압축 (MP3, AAC):
  → 10MB (90% 절감)
  → 사람이 못 듣는 고주파 제거
  → 복원 불가 (영구 손실)
  → 음악 감상에는 충분

무손실 압축 (FLAC):
  → 60MB (40% 절감)
  → 원본 데이터 100% 보존
  → 복원 시 원본과 동일
  → STT, 의학, 법원 녹취에 사용
```

**오디오 포맷 비교**

| 포맷 | 압축 방식 | 파일크기 | 음질 | 주 용도 |
|------|----------|---------|------|---------|
| WAV | 무압축 | 100% | 완벽 | 스튜디오 원본 |
| FLAC | 무손실 | ~60% | 완벽 | STT, 아카이브 |
| AAC | 손실 | ~10% | 우수 | 스트리밍 |
| MP3 | 손실 | ~10% | 양호 | 범용 |
| OGG | 손실 | ~8% | 양호 | 게임, 웹 |
| Opus | 손실 | ~7% | 우수 | 실시간 통화 |

---

### 2-3. SRT / VTT 자막 형식

**SRT (SubRip Text)**
```srt
1
00:00:01,500 --> 00:00:04,200
안녕하세요, 오늘은 Media AI에 대해 알아보겠습니다.

2
00:00:04,500 --> 00:00:07,000
첫 번째로 Video Intelligence API를 살펴볼게요.
```

구조: `번호 → 시간코드 → 텍스트 → 빈 줄`

**VTT (Web Video Text Tracks)**
```vtt
WEBVTT

00:00:01.500 --> 00:00:04.200
안녕하세요, 오늘은 Media AI에 대해 알아보겠습니다.
```

**차이점**

| 항목 | SRT | VTT |
|------|-----|-----|
| 시간 구분자 | `,` (콤마) | `.` (점) |
| 헤더 | 없음 | `WEBVTT` 필수 |
| 스타일링 | 불가 | CSS 스타일 가능 |
| 웹 표준 | 비공식 | W3C 공식 표준 |

---

### 2-4. 영상 컨테이너와 코덱

**컨테이너 (Container)**
여러 스트림(영상, 음성, 자막)을 하나의 파일로 묶는 "박스".
```
MP4 컨테이너 안의 내용:
  ├─ 영상 스트림 (H.264 코덱으로 압축)
  ├─ 음성 스트림 (AAC 코덱으로 압축)
  └─ 자막 스트림 (선택사항)
```

**코덱 (Codec) = Coder + Decoder**
데이터를 압축(Encode)하고 해제(Decode)하는 알고리즘.

```
주요 영상 코덱:
  H.264 (AVC): 현재 가장 널리 사용, MP4에 주로 포함
  H.265 (HEVC): H.264보다 2배 효율적, 최신 기기용
  VP9: Google 개발, YouTube 사용
  AV1: 차세대 오픈소스 표준, Netflix/YouTube 도입 중

주요 음성 코덱:
  AAC: MP4의 기본 음성 코덱
  MP3: 오래된 표준, 여전히 널리 사용
  FLAC: 무손실
  Opus: 실시간 스트리밍 최적 (Discord, WebRTC)
```

**영상 코덱 비교**

| 코덱 | 압축 효율 | 인코딩 속도 | 라이선스 | 주 사용처 |
|------|----------|-----------|---------|---------|
| H.264 | 보통 | 빠름 | 특허료 | 범용 |
| H.265 | H.264 대비 2배 | 느림 | 특허료 높음 | 4K 스트리밍 |
| VP9 | H.265 수준 | 느림 | 무료 | YouTube |
| AV1 | VP9 대비 30% 향상 | 매우 느림 | 무료 | Netflix, 차세대 |

---

## 3. 인증 및 보안

### 3-1. ADC (Application Default Credentials)

**무엇인가?**
Google Cloud 서비스에 접근할 때 "나는 누구인가"를 증명하는 자격증명을 자동으로 찾아주는 메커니즘.

**탐색 순서**
```
1. 환경변수 GOOGLE_APPLICATION_CREDENTIALS에 지정된 서비스 계정 키 파일
2. gcloud 로그인 정보 (~/.config/gcloud/)
3. Compute Engine / Cloud Run 등 GCP 환경의 메타데이터 서버
4. 찾지 못하면 오류
```

**Colab에서의 문제**
```
Colab은 Google 계정으로 로그인 → ADC 자동 설정
그런데 기본 프로젝트가 Colab 내부 프로젝트(522309567947)로 잡힘
→ 이 프로젝트는 결제 없음, API 비활성화
→ PERMISSION_DENIED 오류 발생

해결: quota_project_id로 실제 프로젝트 명시
```

---

### 3-2. quota_project_id

**무엇인가?**
GCP API 사용 비용을 어느 프로젝트에 청구할지 명시하는 설정.

```python
# 적용 전 (Colab 기본)
client = videointelligence.VideoIntelligenceServiceClient()
# → 522309567947 프로젝트로 요금 청구 시도 → 실패

# 적용 후
client = videointelligence.VideoIntelligenceServiceClient(
    client_options={"quota_project_id": "gen-lang-client-0318067486"}
)
# → 실제 프로젝트로 요금 청구 → 성공
```

---

### 3-3. 서비스 계정 (Service Account)

**무엇인가?**
사람이 아닌 애플리케이션(서버, 배치 작업)을 위한 GCP 계정.

```
사용자 계정: 사람이 로그인 (example@gmail.com)
서비스 계정: 앱/서버가 인증 (my-app@project.iam.gserviceaccount.com)

서비스 계정 키 파일 (.json):
  → 서버에 저장
  → GOOGLE_APPLICATION_CREDENTIALS 환경변수로 경로 지정
  → 앱이 자동으로 GCP 인증
```

---

## 4. 추천 시스템 — 유사도 연산 포함

### 4-1. 협업 필터링 (Collaborative Filtering)

**핵심 아이디어**
> "나와 비슷한 취향의 사람들이 좋아한 것을 나도 좋아할 것이다"

**두 가지 방식**

```
사용자 기반 (User-based):
  user_A ↔ user_B 유사도 계산
  → user_B가 본 영상 중 user_A가 안 본 것 추천

아이템 기반 (Item-based):
  video_X ↔ video_Y 유사도 계산
  → video_X를 본 사람에게 비슷한 video_Y 추천
```

**추천 시스템 방식 비교**

| 방식 | 설명 | 장점 | 단점 |
|------|------|------|------|
| 협업 필터링 | 유사 사용자 기반 | 개인화 강함 | 콜드 스타트 문제 |
| 콘텐츠 기반 | 아이템 속성 기반 | 새 아이템에 강함 | 다양성 부족 |
| 행렬 분해 | 잠재 요인 학습 | 정확도 높음 | 학습 비용 큼 |
| 하이브리드 | 협업 + 콘텐츠 혼합 | 균형 잡힘 | 구현 복잡 |
| 딥러닝 기반 | 신경망 임베딩 | 복잡한 패턴 학습 | 해석 어려움 |

---

### 4-2. 암시적 평점 (Implicit Rating)

**명시적 vs 암시적**
```
명시적 평점: 사용자가 직접 별점 부여 (1~5점)
  → 데이터 적음, 바이어스 있음 (극단적 평가 경향)

암시적 평점: 행동 데이터에서 간접적으로 추론
  → 데이터 많음, 자연스러운 선호도 반영
```

**본 프로젝트의 암시적 평점 계산 방법**

시청 완료율에 3.0을 곱하고, 좋아요 여부에 2.0을 곱하고, 공유 여부에 1.0을 곱한 뒤 세 값을 더하면 암시적 평점이 됩니다.

```
가중치 의미:
  3.0 × 시청률: 가장 중요한 지표 (의도치 않은 행동)
  2.0 × 좋아요: 적극적인 긍정 신호
  1.0 × 공유: 추가 관심 신호

예시:
  90% 시청 + 좋아요 = 0.9×3 + 1×2 + 0×1 = 4.7점
  20% 시청 = 0.2×3 = 0.6점 (관심 없음)
```

---

### 4-3. 코사인 유사도 (Cosine Similarity)

**핵심 아이디어**
두 벡터가 같은 방향을 가리키면 유사하다.

**벡터로 사용자 표현**
```
영상이 5개 있다면, 각 사용자는 5차원 벡터:

user_A = [4.7,  0.0,  2.1,  0.0,  3.5]
          영상1  영상2  영상3  영상4  영상5

user_B = [5.2,  3.1,  0.0,  1.8,  4.0]

공통으로 본 영상: 영상1(4.7 vs 5.2), 영상5(3.5 vs 4.0)
```

**코사인 유사도 계산 방법 (말로 설명)**

두 벡터 A와 B의 코사인 유사도는, A와 B의 내적(각 차원의 값을 곱해서 모두 더한 값)을 A의 크기와 B의 크기의 곱으로 나눈 값입니다. 내적은 두 사람이 같은 영상을 높게 평가할수록 커집니다. 크기로 나누는 것은 많이 시청한 사람과 적게 시청한 사람을 공평하게 비교하기 위한 정규화 과정입니다. 결과는 -1에서 1 사이이며, 1에 가까울수록 취향이 비슷합니다.

**계산 예시**
```
user_A = [4.7, 0.0, 3.5]   (영상1, 영상2, 영상3만 고려)
user_B = [5.2, 3.1, 4.0]

내적 = 4.7×5.2 + 0.0×3.1 + 3.5×4.0
     = 24.44 + 0 + 14.0 = 38.44

A의 크기 = 루트(4.7² + 0² + 3.5²) = 루트(22.09 + 0 + 12.25) = 루트(34.34) = 5.86
B의 크기 = 루트(5.2² + 3.1² + 4.0²) = 루트(27.04 + 9.61 + 16.0) = 루트(52.65) = 7.26

유사도 = 38.44 / (5.86 × 7.26) = 38.44 / 42.54 = 0.90

→ 0.90 (매우 유사한 취향!)
```

**BigQuery SQL 구현**
```sql
-- 내적 계산
SUM(a.implicit_rating * b.implicit_rating) AS dot_product

-- 벡터 크기 계산
SQRT(SUM(implicit_rating * implicit_rating)) AS norm

-- 코사인 유사도
dot_product / (norm_a * norm_b) AS cosine_similarity
```

---

### 4-4. 유사도 가중 추천 점수

**계산 방법 (말로 설명)**

특정 영상의 추천 점수는 유사한 사용자들이 준 평점을 유사도로 가중 평균하여 구합니다. 구체적으로는, 각 유사 사용자의 평점에 나와의 유사도를 곱한 값들을 모두 더하고, 이를 유사도의 합으로 나눕니다. 이렇게 하면 나와 취향이 90% 일치하는 사람의 의견이 50% 일치하는 사람의 의견보다 더 많이 반영됩니다.

```
예시: 영상X의 추천 점수 계산
  유사유저A (유사도=0.90) × 평점 6.0 = 5.4
  유사유저B (유사도=0.70) × 평점 4.0 = 2.8
  유사유저C (유사도=0.50) × 평점 5.0 = 2.5
  ─────────────────────────────────────
  합계 = 10.7 / (0.90+0.70+0.50) = 10.7/2.1 = 5.1점
```

---

### 4-5. 행렬 분해 (Matrix Factorization) — 비교 개념

BigQuery ML MATRIX_FACTORIZATION이 요구하는 방식.

**개념**
```
사용자-아이템 행렬 R (빈 칸 = 미시청):

         영상1  영상2  영상3  영상4
user_A:   4.7   ?     2.1   ?
user_B:   5.2   3.1   ?     1.8
user_C:   ?     ?     3.5   4.2

→ 두 작은 행렬로 분해:
  R ≈ U × V의 전치행렬
  U: 사용자 잠재 벡터 (사용자 취향 요약)
  V: 아이템 잠재 벡터 (영상 특성 요약)

→ 분해된 행렬의 곱으로 빈 칸 예측 가능
```

**코사인 유사도 vs 행렬 분해**

| 항목 | 코사인 유사도 | 행렬 분해 |
|------|-------------|-----------|
| 계산 방식 | 직접 유사도 계산 | 반복 학습으로 잠재 요인 추출 |
| 정확도 | 보통 | 더 높음 |
| 속도 | 빠름 (SQL 뷰) | 느림 (수십 분 학습) |
| 인프라 | 온디맨드 OK | 슬롯 예약 필요 |
| 해석가능성 | 직관적 | 잠재 요인 해석 어려움 |

---

## 5. 딥러닝 기초

### 5-1. 텐서 (Tensor)

**무엇인가?**
다차원 배열. PyTorch와 TensorFlow의 기본 데이터 단위.

```
0차원 텐서 (스칼라): 3.14
1차원 텐서 (벡터): [1, 2, 3, 4]
2차원 텐서 (행렬): [[1, 2], [3, 4]]
3차원 텐서: (배치, 시퀀스, 특성) = (32, 128, 512)
4차원 텐서: (배치, 채널, 높이, 너비) = (4, 3, 224, 224)  ← 이미지
5차원 텐서: (배치, 프레임, 채널, 높이, 너비) ← 영상
```

**PyTorch vs TensorFlow 비교**

| 항목 | PyTorch | TensorFlow |
|------|---------|------------|
| 실행 방식 | 즉시 실행 (Eager) | 정적 그래프 (TF1) / Eager (TF2) |
| 사용 편의성 | Python 친화적, 직관적 | 초기 학습 곡선 있음 |
| 연구 채택률 | 학계 압도적 우위 | 산업계 다수 사용 |
| 모바일 배포 | TorchScript/ONNX | TFLite 성숙 |
| 기본 배열 이름 | Tensor | Tensor |

---

### 5-2. Convolution (합성곱)

**직관적 설명**
이미지 위에 작은 필터(커널)를 슬라이딩하며 특징 추출.

```
원본 이미지 (5×5):      필터 (3×3):
1 2 3 4 5               1 0 -1
6 7 8 9 0     *         1 0 -1   → 엣지 감지 필터
...                     1 0 -1

결과: 각 위치에서 필터와의 점곱 → 특징 맵(Feature Map)
```

**Conv2d 파라미터 의미**
```python
nn.Conv2d(
    in_channels=3,     # 입력 채널 수 (RGB → 3)
    out_channels=64,   # 출력 채널 수 (학습할 필터 개수)
    kernel_size=3,     # 필터 크기 (3×3)
    padding=1,         # 테두리 패딩 (출력 크기 유지)
)
```

---

### 5-3. Self-Attention (자기 주의)

**핵심 아이디어**
> "문장 안의 각 단어가 다른 모든 단어와 얼마나 관련 있는지 계산"

**QKV 구조**
```
Q (Query): "나는 무엇을 찾고 있나?"
K (Key):   "나는 어떤 정보를 가지고 있나?"
V (Value): "실제 전달할 정보"

예: "나는 [사과]를 먹었다"에서 [사과]를 이해할 때
  Q = "사과가 뭔가?"
  K = [나, 는, 사과, 를, 먹었다] 각각의 관련성 점수
  V = 관련성에 따라 가중 평균한 정보
```

**Attention 계산 방법 (말로 설명)**

쿼리 행렬(Q)과 키 행렬(K)을 행렬 곱한 뒤, 키의 차원 크기의 제곱근으로 나눠 값이 너무 커지는 것을 방지합니다. 이 결과에 소프트맥스를 적용해 각 위치의 가중치 합이 1이 되는 확률 분포를 만들고, 마지막으로 밸류 행렬(V)과 곱해 가중 평균된 출력을 얻습니다.

```
Q @ K^T: 모든 쿼리-키 쌍의 유사도 계산 (행렬 곱)
√d_k로 나누기: 값이 너무 커지지 않도록 스케일링
softmax: 합이 1이 되는 확률 분포로 변환
× V: 확률에 따라 가중 평균
```

---

### 5-4. LayerNorm (레이어 정규화)

**왜 필요한가?**
딥러닝에서 각 층의 출력값 분포가 불안정해지면 학습이 어려워짐.

**정규화 방법 (말로 설명)**

현재 레이어 출력값들의 평균을 구하고 각 값에서 빼줍니다. 그 결과를 표준편차로 나눠 평균 0, 표준편차 1인 분포로 만듭니다. 0으로 나누는 오류를 막기 위해 분모에 아주 작은 값(약 0.00001)을 더합니다. 마지막으로 학습 가능한 스케일 파라미터(감마)와 이동 파라미터(베타)를 곱하고 더해서 모델이 최적의 분포를 스스로 찾도록 합니다.

```
μ (평균): 현재 레이어 출력의 평균
σ (표준편차): 현재 레이어 출력의 퍼짐 정도
정규화: (x - 평균) / 표준편차
γ, β: 학습 가능한 파라미터 (스케일, 이동)
ε: 0으로 나누는 것 방지 (보통 1e-5)
```

**정규화 방법 비교**

| 방법 | 정규화 대상 | 장점 | 단점 | 주로 사용 |
|------|-----------|------|------|---------|
| BatchNorm | 배치 전체 | 빠른 학습 | 배치 크기 의존 | CNN |
| LayerNorm | 각 샘플 개별 | 배치 크기 무관 | 약간 느림 | Transformer |
| InstanceNorm | 채널 개별 | 스타일 변환 강점 | 정보 손실 | GAN |
| GroupNorm | 채널 그룹 | 소배치에 강함 | 그룹 수 튜닝 필요 | 객체 감지 |

---

### 5-5. GELU 활성화 함수

**무엇인가?**
뉴런 출력을 비선형 변환하는 함수. Transformer에서 ReLU 대신 사용.

**GELU 계산 방법 (말로 설명)**

GELU는 입력값 x에 표준 정규분포의 누적확률(x보다 작은 값이 나올 확률)을 곱한 값입니다. 실제 구현에서는 이를 tanh 함수를 이용한 근사식으로 계산합니다. x에 0.5를 곱하고, 여기에 tanh(루트(2/파이)×(x + 0.044715×x의 세제곱))의 결과에 1을 더한 값을 곱합니다.

**활성화 함수 비교**

| 함수 | 특성 | 장점 | 단점 | 주 사용처 |
|------|------|------|------|---------|
| ReLU | x>0이면 x, 아니면 0 | 빠름, 단순 | 음수 죽는 뉴런 문제 | CNN |
| Leaky ReLU | 음수에 작은 기울기 | 죽는 뉴런 완화 | 하이퍼파라미터 추가 | GAN |
| GELU | 확률적 부드러운 경계 | Transformer에 최적 | ReLU보다 약간 느림 | BERT, GPT |
| SiLU/Swish | x × 시그모이드(x) | GELU와 유사 성능 | 계산 비용 | LLaMA |
| Sigmoid | 0~1 출력 | 확률 표현 | 기울기 소실 | 이진 분류 출력층 |

---

## 6. 신경망 구조 비교 — CNN vs ANN vs RNN vs Transformer

### 6-1. ANN (Artificial Neural Network, 인공 신경망)

**무엇인가?**
가장 기본적인 형태의 신경망. 입력층 → 은닉층 → 출력층으로 구성된 완전 연결(Fully Connected) 구조.

```
입력: [1, 2, 3, 4, 5]
  ↓ (모든 입력이 모든 뉴런에 연결)
은닉층: 각 뉴런 = 입력들의 가중합 + 활성화 함수
  ↓
출력: [0.1, 0.9] (이진 분류 예시)
```

**특징**
- 순서나 공간 정보를 고려하지 않음
- 파라미터 수가 많음 (모든 연결에 가중치)
- 구조가 단순하여 이해하기 쉬움

---

### 6-2. CNN (Convolutional Neural Network, 합성곱 신경망)

**무엇인가?**
이미지나 영상처럼 공간적 구조가 있는 데이터에 특화된 신경망. 작은 필터로 슬라이딩하며 특징을 추출.

```
이미지 입력 (224×224×3)
  ↓ Conv Layer (필터로 엣지, 텍스처 등 로컬 특징 추출)
  ↓ Pooling (크기 축소, 중요 특징만 보존)
  ↓ Conv Layer (더 추상적인 특징 추출)
  ↓ Flatten (1차원으로 펼침)
  ↓ Fully Connected (ANN과 동일)
  ↓ 출력: [강아지 0.95, 고양이 0.03, ...]
```

**CNN 특징 추출 과정**
```
1층: 엣지(가로선, 세로선, 대각선) 감지
2층: 형태(모서리, 곡선) 감지
3층: 패턴(눈, 코, 귀 형태) 감지
4층: 객체(얼굴, 몸통) 감지
→ 계층적 특징 학습
```

---

### 6-3. RNN (Recurrent Neural Network, 순환 신경망)

**무엇인가?**
순서가 있는 데이터(텍스트, 시계열)에 특화된 신경망. 이전 시점의 정보를 다음 시점에 전달.

```
"나는 사과를 먹었다" 처리:
  h0(초기) → "나는" → h1
  h1 → "사과를" → h2    ← h1 정보 포함
  h2 → "먹었다" → h3    ← h1, h2 정보 포함
  h3 → 문장 이해 결과
```

**RNN 변종**

| 종류 | 특징 | 문제 해결 | 주 사용처 |
|------|------|---------|---------|
| 기본 RNN | 간단한 순환 | - | 교육용 |
| LSTM | 장기 기억 게이트 | 기울기 소실 | 번역, 음성 |
| GRU | LSTM 간소화 | 기울기 소실 (빠름) | 번역, 시계열 |

---

### 6-4. Transformer

**무엇인가?**
Self-Attention으로 모든 위치 쌍의 관계를 동시에 계산. RNN의 순차 처리 한계를 극복.

```
"나는 사과를 먹었다" 처리:
  "나는"이 "먹었다"와 얼마나 관련? → 동시에 계산
  "사과"가 "먹었다"와 얼마나 관련? → 동시에 계산
  모든 단어 쌍 관계 → 병렬 처리 가능
```

---

### 6-5. 네 가지 구조 종합 비교

| 항목 | ANN | CNN | RNN/LSTM | Transformer |
|------|-----|-----|----------|-------------|
| 주요 데이터 | 정형 데이터 | 이미지/영상 | 시계열/텍스트 | 텍스트/멀티모달 |
| 공간 정보 활용 | 없음 | 강함 | 없음 | Attention으로 처리 |
| 순서 정보 활용 | 없음 | 없음 | 강함 | Position Encoding |
| 병렬 처리 | 가능 | 가능 | 불가 (순차) | 가능 |
| 장기 의존성 | 약함 | 약함 | LSTM으로 개선 | 강함 |
| 파라미터 수 | 많음 | 적음(공유) | 보통 | 매우 많음 |
| 대표 모델 | MLP | ResNet, VGG | LSTM, GRU | BERT, GPT |
| 주요 응용 | 분류, 회귀 | 이미지 인식 | 번역, 음성 | LLM, 번역 |

---

### 6-6. CNN 연산의 유사도 활용

CNN에서 유사도는 다음과 같이 활용됩니다.

**특징 벡터 추출 후 유사도 계산**
```python
# CNN으로 이미지 특징 추출
model = ResNet50(pretrained=True)
feature_A = model(image_A)  # 예: [1, 2048] 벡터
feature_B = model(image_B)

# 코사인 유사도로 이미지 유사성 판단
import torch.nn.functional as F
similarity = F.cosine_similarity(feature_A, feature_B)
# → 1.0에 가까울수록 유사한 이미지
```

**이미지 검색 시스템 구조**
```
질의 이미지
  ↓ CNN (특징 추출기)
  ↓ 2048차원 특징 벡터
  ↓ 코사인 유사도 계산
  ↓ DB의 수백만 이미지 벡터들과 비교
  → 가장 유사한 Top-K 이미지 반환
```

---

## 7. 유사도 연산 상세 비교

### 7-1. 유사도 측정 방법 종합 비교

**코사인 유사도 (Cosine Similarity)**

두 벡터가 이루는 각도의 코사인 값입니다. 두 벡터의 내적을 각 벡터의 크기의 곱으로 나눕니다. 벡터의 방향이 같으면 1, 수직이면 0, 반대이면 -1이 나옵니다. 크기(절댓값)를 무시하고 방향만 비교하므로, 문서 길이나 사용자 시청량 차이를 정규화할 수 있습니다.

```python
import torch
import torch.nn.functional as F

a = torch.tensor([4.7, 0.0, 3.5])
b = torch.tensor([5.2, 3.1, 4.0])
similarity = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
# → tensor([0.9003])
```

**유클리드 거리 (Euclidean Distance)**

두 벡터 사이의 직선 거리입니다. 각 차원의 차이를 제곱해서 더한 뒤 제곱근을 취합니다. 값이 0이면 동일, 클수록 멀리 떨어져 있습니다. 코사인 유사도와 달리 크기 차이도 반영합니다.

```python
distance = torch.dist(a, b, p=2)
# p=2 는 유클리드 거리
# p=1 이면 맨해튼 거리
```

**내적 (Dot Product)**

각 차원의 값을 곱해서 모두 더한 값입니다. 코사인 유사도에서 정규화를 하지 않은 형태입니다. 벡터가 정규화되어 있으면 코사인 유사도와 동일해집니다. 추천 시스템에서 빠른 유사도 계산에 사용합니다.

```python
dot_product = torch.dot(a, b)
# → tensor(38.4400)
```

**유사도 방법 비교**

| 방법 | 계산 방식 | 범위 | 크기 반영 | 주 사용처 |
|------|---------|------|---------|---------|
| 코사인 유사도 | 내적 / (크기 × 크기) | -1 ~ 1 | 아니오 | 추천, NLP |
| 유클리드 거리 | 각 차원 차이의 거리 | 0 ~ 무한 | 예 | 클러스터링 |
| 내적 (Dot Product) | 각 차원 곱의 합 | -무한 ~ 무한 | 예 | FAISS, 행렬곱 |
| 맨해튼 거리 | 각 차원 차이의 절댓값 합 | 0 ~ 무한 | 예 | 로보틱스, 격자 |
| 해밍 거리 | 다른 비트 수 | 0 ~ n | 이진 | 해시, DNA |
| Jaccard 유사도 | 교집합 / 합집합 | 0 ~ 1 | 집합 기반 | 추천, 집합 비교 |

---

### 7-2. CUDA를 이용한 유사도 연산

**배치 코사인 유사도 (대규모 벡터 검색)**

```python
import torch

# GPU에서 대규모 배치 유사도 계산
# query: 검색 벡터 (1개), candidates: DB 벡터들 (100만 개)
query = torch.randn(1, 512).cuda()           # 512차원 벡터
candidates = torch.randn(1_000_000, 512).cuda()

# L2 정규화 후 내적 = 코사인 유사도
query_norm = F.normalize(query, p=2, dim=1)
cand_norm = F.normalize(candidates, p=2, dim=1)

# 행렬 곱으로 한번에 100만 개 유사도 계산
# CUDA의 병렬 처리로 수 ms 안에 완료
similarities = torch.mm(query_norm, cand_norm.T)  # [1, 1000000]
top_k = torch.topk(similarities, k=10)
```

**CUDA 내부 연산 흐름**
```
CPU: 쿼리 벡터, DB 벡터 준비
  ↓ PCIe 전송 (CPU → GPU 메모리)
GPU:
  1. 정규화 커널: 각 벡터를 단위 벡터로 변환
  2. GEMM 커널 (TensorCore 사용):
     [1 × 512] × [512 × 1,000,000] = [1 × 1,000,000]
     → TensorCore가 행렬 곱을 수천 개 병렬 계산
  3. Top-K 커널: 가장 높은 유사도 K개 선택
  ↓ PCIe 전송 (GPU → CPU)
CPU: 결과 수신
```

**FAISS (Facebook AI Similarity Search)**

대규모 벡터 검색 라이브러리. CUDA로 GPU 가속됩니다.

```python
import faiss
import numpy as np

d = 512       # 벡터 차원
n = 1_000_000 # DB 벡터 수

# GPU 인덱스 생성 (내적 기반, 정규화 후 = 코사인 유사도)
res = faiss.StandardGpuResources()
index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
index = faiss.index_cpu_to_gpu(res, 0, index)

# 벡터 추가
vectors = np.random.randn(n, d).astype('float32')
faiss.normalize_L2(vectors)  # 코사인 유사도를 위해 정규화
index.add(vectors)

# 검색 (코사인 유사도 기반 Top-10)
query = np.random.randn(1, d).astype('float32')
faiss.normalize_L2(query)
distances, indices = index.search(query, 10)
# 100만 벡터 검색: GPU에서 ~수 ms
```

---

### 7-3. ANN (Approximate Nearest Neighbor, 근사 최근접 이웃)

> 주의: 여기서의 ANN은 '인공 신경망'이 아니라 '근사 최근접 이웃 검색'을 의미합니다.

**정확한 검색 vs 근사 검색**
```
정확한 검색 (Exact Search):
  100만 벡터 전부와 유사도 계산
  → 정확하지만 느림 (O(n) 시간 복잡도)

근사 검색 (ANN):
  인덱스 구조로 빠르게 후보 좁힘
  → 99% 정확도로 100배 빠름
  → 대규모 실서비스에 필수
```

**ANN 알고리즘 비교**

| 알고리즘 | 라이브러리 | 인덱스 구조 | 특징 |
|---------|---------|-----------|------|
| HNSW | hnswlib, FAISS | 계층적 그래프 | 속도와 정확도 균형 최고 |
| IVF | FAISS | 클러스터 기반 | GPU 가속 최적 |
| LSH | 여러 구현체 | 해시 기반 | 이론적으로 단순 |
| ScaNN | Google | 양자화 + 파티션 | 대규모 추천 시스템 |
| Annoy | Spotify | 이진 트리 | 메모리 효율 |

---

## 8. PyTorch 최적화 기술

### 8-1. torch.compile

**무엇인가?**
PyTorch 2.0에서 추가된 기능. Python 코드를 분석하여 자동으로 최적화된 GPU 커널로 변환.

**동작 원리**
```
Python 코드
    → torch._dynamo: Python 바이트코드 분석 → 연산 그래프 추출
    → torch._inductor: 그래프 최적화 → Triton 커널 코드 생성
    → Triton JIT: Triton 코드 → PTX (GPU 중간 언어)
    → CUDA 컴파일러: PTX → SASS (실제 GPU 실행 코드)
    → GPU 캐시에 저장 → 이후 재사용
```

**첫 실행이 느린 이유**
```
첫 실행: Python 분석 + 커널 컴파일 = 수 초~수십 초
이후 실행: 캐시된 커널 바로 실행 = ms 단위

∴ 추론(Inference)을 수천 번 반복하는 프로덕션에서 가치 있음
   한두 번 실행하는 실험/개발에서는 오히려 손해
```

**mode 옵션**
```python
torch.compile(model, backend="inductor", mode="default")
# default: 범용 최적화

torch.compile(model, backend="inductor", mode="reduce-overhead")
# reduce-overhead: CUDA graph 활용, 커널 실행 오버헤드 최소화
# 입력 크기가 항상 같을 때 효과적

torch.compile(model, backend="inductor", mode="max-autotune")
# max-autotune: 최대 최적화 탐색 (컴파일 수 분 소요, A100+ 권장)
```

**torch.compile 내부 컴파일 단계 비교**

| 단계 | 역할 | 출력 |
|------|------|------|
| Dynamo | Python 코드 → FX 그래프 | 연산 그래프 (Python) |
| AOTAutograd | 자동 미분 처리 | Forward/Backward 그래프 |
| Inductor | 그래프 최적화 + 코드 생성 | Triton/C++ 커널 코드 |
| Triton JIT | Triton → PTX | GPU 어셈블리 |
| NVCC | PTX → SASS | 실제 GPU 바이너리 |

---

### 8-2. 커널 퓨전 (Kernel Fusion)

**문제 상황**
```
Eager 모드에서 Linear → GELU 실행:

1. Linear 계산 → 결과를 GPU 메모리에 저장 (쓰기)
2. GELU 입력을 GPU 메모리에서 읽기
3. GELU 계산 → 결과를 GPU 메모리에 저장 (쓰기)

GPU 메모리 읽기/쓰기가 3번 발생
```

**퓨전 후**
```
torch.compile이 Linear + GELU를 하나의 커널로 합침:

1. Linear 계산 → 레지스터(초고속 임시저장)에 보관
2. GELU 계산 → 결과를 GPU 메모리에 저장 (쓰기)

GPU 메모리 읽기/쓰기가 1번으로 감소
→ 메모리 대역폭 절약 → 속도 향상
```

---

### 8-3. FP32 / FP16 / BF16

**수 표현 방식**

```
FP32 (float32, 단정밀도):
  부호(1) + 지수(8) + 가수(23) = 32비트
  범위: ±3.4 × 10^38
  정밀도: 소수점 약 7자리
  → 딥러닝 학습/추론 기본값

FP16 (float16, 반정밀도):
  부호(1) + 지수(5) + 가수(10) = 16비트
  범위: ±65,504  ← 범위가 매우 좁음
  정밀도: 소수점 약 3자리
  → gradient overflow/underflow 주의

BF16 (bfloat16, Brain Float 16):
  부호(1) + 지수(8) + 가수(7) = 16비트
  범위: FP32와 동일 ← 지수 비트 수 같음
  정밀도: FP16보다 낮음
  → A100/H100 에서 학습에 권장 (범위 넓어 안전)
```

**정밀도 형식 비교**

| 형식 | 비트 수 | 숫자 범위 | 메모리 | 속도(T4 기준) | 주 용도 |
|------|--------|---------|--------|-------------|--------|
| FP64 | 64 | 매우 넓음 | 4배 | 0.25× | 과학 계산 |
| FP32 | 32 | 넓음 | 기준 | 1× | 학습 기본값 |
| TF32 | 19 | FP32와 같음 | 기준 | 10× | A100 학습 |
| BF16 | 16 | FP32와 같음 | 0.5× | 8× | A100+ 학습 |
| FP16 | 16 | 좁음 | 0.5× | 8× | T4 추론 |
| INT8 | 8 | 정수만 | 0.25× | 16× | 추론 전용 |
| FP8 | 8 | 좁음 | 0.25× | 32× | H100 학습/추론 |

---

### 8-4. AMP (Automatic Mixed Precision)

**무엇인가?**
학습/추론 시 연산에 따라 FP32와 FP16을 자동으로 선택해주는 기능.

**연산별 자동 선택**
```
FP16으로 실행 (속도 우선):
  - 행렬 곱 (matmul) — TensorCore 활용
  - Convolution
  - Linear

FP32로 실행 (정확도 우선):
  - Softmax — 작은 값들의 합, FP16이면 소실
  - LayerNorm — 분산 계산, 정밀도 필요
  - Loss 계산
```

**사용법**
```python
# 추론
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)

# 학습 (GradScaler 함께 사용)
scaler = torch.cuda.amp.GradScaler()
with torch.autocast('cuda'):
    loss = model(input)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 9. CUDA와 Triton — 유사성·연관성·차이점 완전 정리

### 9-1. CUDA란 무엇인가?

**CUDA (Compute Unified Device Architecture)**

NVIDIA가 2006년에 발표한 GPU 병렬 컴퓨팅 플랫폼 및 프로그래밍 모델입니다. C/C++ 언어를 확장하여 GPU에서 실행되는 코드(커널)를 작성할 수 있게 합니다.

**CUDA의 핵심 개념**
```
스레드 (Thread):
  → GPU의 가장 작은 실행 단위
  → 수십만 개가 동시에 실행

워프 (Warp):
  → 32개 스레드의 묶음
  → 항상 같은 명령어를 동시에 실행 (SIMT 방식)

블록 (Block):
  → 워프들의 집합 (최대 1024 스레드)
  → 같은 SM(Streaming Multiprocessor)에서 실행
  → 공유 메모리(Shared Memory) 사용 가능

그리드 (Grid):
  → 블록들의 집합
  → 하나의 커널 실행 = 하나의 그리드
```

**CUDA 커널 예시 (벡터 덧셈)**
```c
// C++로 작성된 GPU 커널
__global__ void vector_add(float* a, float* b, float* c, int n) {
    // 이 스레드가 처리할 인덱스 계산
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // 병렬로 수행
    }
}

// CPU에서 커널 호출
int threads_per_block = 256;
int blocks = (n + threads_per_block - 1) / threads_per_block;
vector_add<<<blocks, threads_per_block>>>(a, b, c, n);
```

---

### 9-2. Triton이란 무엇인가?

**Triton**

OpenAI가 2021년에 공개한 GPU 프로그래밍 언어 및 컴파일러입니다. Python 문법으로 고성능 GPU 커널을 작성할 수 있으며, CUDA보다 훨씬 낮은 진입 장벽으로 동등하거나 더 높은 성능을 제공합니다.

**Triton 커널 예시 (벡터 덧셈)**
```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    # 이 프로그램 인스턴스가 처리할 블록 ID
    pid = tl.program_id(axis=0)
    # 이 블록이 처리할 인덱스 범위
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n  # 경계 체크
    # 메모리에서 데이터 로드
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    # 계산 후 저장
    tl.store(c_ptr + offsets, a + b, mask=mask)

# 호출
grid = (triton.cdiv(n, BLOCK_SIZE),)
vector_add_kernel[grid](a, b, c, n, BLOCK_SIZE=1024)
```

---

### 9-3. CUDA와 Triton의 유사성

두 기술은 근본적으로 같은 목적을 위해 존재하며, 내부적으로 깊이 연결되어 있습니다.

**공통 목표와 개념**

| 개념 | CUDA 용어 | Triton 용어 | 설명 |
|------|----------|-----------|------|
| 최소 실행 단위 | Thread | 스칼라 연산 | 하나의 연산 |
| 병렬 단위 | Thread Block | Program Instance | 독립적으로 실행되는 단위 |
| 데이터 묶음 | Warp(32) | Block(사용자 지정) | 동시에 처리하는 데이터 수 |
| 빠른 임시 저장 | Shared Memory | 자동 관리 | GPU 내 고속 메모리 |
| 전역 저장 | Global Memory | Pointer + Offset | GPU VRAM |
| 실행 구성 | <<<grid, block>>> | [grid](args) | 병렬 실행 설정 |

**기술적 연결 관계**
```
Triton 코드
  → Triton 컴파일러
  → LLVM IR (중간 표현)
  → PTX (NVIDIA GPU 어셈블리)
  → SASS (실제 GPU 머신코드)
       ↑
  CUDA도 동일한 PTX → SASS 경로 사용
```

즉, Triton으로 작성한 커널은 최종적으로 CUDA와 동일한 PTX/SASS 형태로 GPU에서 실행됩니다. Triton은 CUDA 위에 구축된 고수준 추상화 계층입니다.

---

### 9-4. CUDA와 Triton의 차이점

**추상화 수준 차이**
```
CUDA: 스레드 수준 추상화
  → 개발자가 스레드, 워프, 블록을 모두 직접 관리
  → 공유 메모리 사용, 동기화, 메모리 접근 패턴을 수동 최적화

Triton: 블록(타일) 수준 추상화
  → 개발자는 "BLOCK_SIZE 단위로 처리한다"만 지정
  → 워프 내 동기화, 공유 메모리 관리는 컴파일러가 자동 처리
```

**구체적 비교**

| 항목 | CUDA | Triton |
|------|------|--------|
| 프로그래밍 언어 | C/C++ 확장 | Python |
| 학습 난이도 | 높음 (수개월~수년) | 낮음 (수일~수주) |
| 스레드 관리 | 직접 (threadIdx, blockIdx) | 자동 (컴파일러 처리) |
| 공유 메모리 | 직접 선언 및 관리 | 자동 관리 |
| 메모리 접근 패턴 | 수동 최적화 필수 | 컴파일러가 최적화 |
| 디버깅 | cuda-gdb, Nsight | Python 디버거 사용 가능 |
| 성능 상한 | 이론적 최대 | CUDA 대비 ~90~100% |
| 유연성 | 매우 높음 | 높음 (일부 제한) |
| 코드 길이 | 길고 복잡 | 짧고 간결 |
| NVIDIA 의존성 | 완전 의존 | 의존 (CUDA 런타임 필요) |
| AMD GPU 지원 | 없음 (CUDA 전용) | 실험적 지원 (ROCm) |

---

### 9-5. torch.compile에서 CUDA와 Triton의 역할

`torch.compile`의 기본 백엔드인 `inductor`가 자동으로 Triton 커널을 생성하는 과정:

```
PyTorch 모델 (Python)
      ↓
  [Dynamo]
  Python 코드 분석, FX 그래프 생성
      ↓
  [Inductor 최적화]
  - 연산 퓨전 (kernel fusion)
  - 메모리 레이아웃 최적화
  - 루프 언롤링, 타일링 결정
      ↓
  [Triton 코드 생성]
  → 자동으로 최적화된 Triton 커널 코드 생성
  → 개발자가 직접 작성한 것과 유사한 품질
      ↓
  [Triton JIT 컴파일]
  → PTX → SASS → GPU 실행
```

**torch.compile이 생성한 Triton 코드 확인**
```python
import torch

model = torch.nn.Linear(512, 512).cuda()
compiled = torch.compile(model)

# 생성된 Triton 코드 출력 (디버깅용)
import torch._inductor.config as config
config.trace.enabled = True

with torch.no_grad():
    x = torch.randn(64, 512).cuda()
    y = compiled(x)
# → logs/ 폴더에 생성된 Triton 커널 코드 저장됨
```

---

### 9-6. Triton이 CUDA보다 효과적인 경우

#### 케이스 1: 새로운 연산자/레이어 개발

새로운 활성화 함수, 어텐션 변형, 정규화 방법을 연구할 때, CUDA로 커스텀 커널을 짜는 데 수 주가 걸리는 작업을 Triton으로 하루 만에 구현할 수 있습니다.

```python
# 커스텀 Flash Attention 구현 예시
@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # 블록 단위로 Q, K, V 처리
    # → VRAM과 SRAM(공유 메모리) 사이 데이터 이동 최소화
    # → 메모리 대역폭 병목 해결
    ...
```

Flash Attention은 Triton으로 구현되어 표준 Attention 대비 메모리 사용량 10배 절감, 속도 3~5배 향상을 달성했습니다.

#### 케이스 2: 메모리 병목(Memory-Bound) 연산

GPU 연산이 계산 속도가 아닌 메모리 읽기/쓰기 속도에 의해 제한될 때, Triton의 자동 타일링이 효과적입니다.

```
메모리 병목 연산 예시:
  - Element-wise 연산 (ReLU, GELU, Sigmoid)
  - 정규화 (LayerNorm, BatchNorm)
  - Dropout
  - Embedding 조회

→ 이런 연산들은 계산보다 데이터 이동이 병목
→ Triton으로 여러 연산을 하나로 퓨전하면 메모리 접근 횟수 감소
→ 수 배 성능 향상 가능
```

**Triton 퓨전 효과 측정 예시**
```python
# 기본 PyTorch (별도 커널 3회 실행)
def naive_gelu_norm(x, weight, bias):
    x = F.gelu(x)           # 커널 1
    x = F.layer_norm(x, ...)  # 커널 2
    x = x * weight + bias    # 커널 3
    return x

# Triton 퓨전 커널 (커널 1회)
@triton.jit
def fused_gelu_norm_kernel(...):
    # GELU + LayerNorm + Scale/Shift를 하나의 커널에서 처리
    # → 중간 결과를 VRAM에 저장/읽기 없이 레지스터에서 처리
    ...

# 성능 차이: 메모리 대역폭 3배 절약 → 속도 2~3배 향상
```

#### 케이스 3: 희소 연산 (Sparse Operations)

행렬이 대부분 0으로 채워진 희소 행렬을 다룰 때, CUDA의 cuSPARSE보다 Triton으로 커스텀 구현이 더 유리한 경우가 있습니다.

```python
# 희소 어텐션 마스크 적용 예시
# 특정 패턴(대각선, 블록 등)만 계산하고 나머지 무시
@triton.jit
def sparse_attention_kernel(
    q_ptr, k_ptr, mask_ptr, out_ptr,
    ...
):
    # 마스크가 1인 위치만 계산 → 불필요한 연산 건너뜀
    # CUDA cuSPARSE: 범용 희소 연산, 패턴 최적화 어려움
    # Triton: 특정 패턴에 맞춘 최적화 가능
    ...
```

#### 케이스 4: 연구 프로토타이핑

```
연구자 workflow:
  새 아이디어 → 빠른 구현 → 성능 검증 → 논문 발표

CUDA 방식:
  아이디어 (1일) → CUDA 커널 구현 (2~4주) → 검증 → 발표
  → 구현 비용이 너무 커서 아이디어 탐색 제한

Triton 방식:
  아이디어 (1일) → Triton 커널 구현 (1~3일) → 검증 → 발표
  → 더 많은 아이디어를 빠르게 탐색 가능

실제 사례:
  - FlashAttention (Triton 구현)
  - Mamba (SSM 커널, Triton으로 프로토타입)
  - SparseGPT (가지치기 커널)
```

#### 케이스 5: torch.compile 자동 최적화

기존 PyTorch 코드에 `torch.compile` 한 줄 추가만으로 Triton 커널의 이점을 누릴 수 있습니다.

```python
# 변경 없이 Triton 최적화 적용
model = MyTransformerModel().cuda()
model = torch.compile(model)  # 이것만 추가
# → Inductor가 자동으로 최적 Triton 커널 생성
# → 일반적으로 1.5~3× 속도 향상
```

---

### 9-7. Triton이 CUDA보다 불리한 경우

균형 잡힌 이해를 위해 Triton의 한계도 알아야 합니다.

| 상황 | 이유 | 권장 |
|------|------|------|
| 최극단 최적화 필요 | CUDA는 워프 수준까지 제어 가능 | CUDA 직접 작성 |
| 비정형 메모리 접근 | Triton은 규칙적인 타일 패턴에 최적화 | CUDA |
| cuDNN, cuBLAS 사용 | 이미 극도로 최적화된 라이브러리 존재 | 기존 라이브러리 |
| NVIDIA 전용 기능 (NVLink 등) | CUDA만 완전 지원 | CUDA |
| 디버깅 복잡한 버그 | CUDA는 성숙한 디버깅 툴 (Nsight) | CUDA |

---

### 9-8. CUDA vs Triton 선택 가이드

```
새 GPU 연산 구현이 필요한가?
  ├─ Yes → 연산이 메모리 병목인가?
  │          ├─ Yes → Triton 선택 (퓨전 효과 극대화)
  │          └─ No → 계산 병목
  │                    ├─ 행렬 곱 등 표준 연산 → cuBLAS/cuDNN 사용
  │                    └─ 커스텀 계산 → CUDA 또는 Triton
  └─ No → 기존 PyTorch 연산만 사용
            → torch.compile 추가 (자동 Triton 최적화)

연구/프로토타입인가?
  → Triton (빠른 개발)

프로덕션 최적화인가?
  → CUDA (세밀한 제어) 또는 Triton (충분한 경우 많음)

GPU가 NVIDIA가 아닌가?
  → OpenCL, Metal (Apple), ROCm (AMD) 고려
  → Triton의 AMD 지원은 실험적 수준
```

---

### 9-9. 실제 성능 비교 데이터 (A100 GPU 기준)

| 연산 | PyTorch Eager | torch.compile (Triton) | 수작업 CUDA | Triton 수작업 |
|------|-------------|----------------------|------------|-------------|
| GELU + LayerNorm 퓨전 | 기준 (1×) | 2.1× | 2.5× | 2.3× |
| Flash Attention | 기준 (1×) | 1.8× | - | 3.2× (FlashAttn2) |
| Softmax | 기준 (1×) | 1.5× | 1.6× | 1.5× |
| Linear (행렬곱) | 기준 (1×) | 1.0× | cuBLAS 동등 | cuBLAS 수준 |
| RMS Norm | 기준 (1×) | 1.9× | 2.1× | 2.0× |

> 행렬 곱(GEMM) 같은 계산 집약적 연산은 cuBLAS가 이미 극도로 최적화되어 있어 Triton이 추가 이득을 주기 어렵습니다. 반면 메모리 병목 연산에서 Triton의 퓨전이 큰 효과를 냅니다.

---

## 10. GPU 하드웨어 개념

### 10-1. SM (Streaming Multiprocessor)

**무엇인가?**
GPU의 기본 연산 단위. CPU의 "코어"에 해당.

```
GPU 내부 구조:
  GPU
  ├─ SM 0
  │    ├─ CUDA Core × 128개
  │    ├─ TensorCore × 4개
  │    └─ 공유 메모리 (96KB)
  ├─ SM 1
  ├─ SM 2
  ...
  └─ SM N
```

| GPU | SM 수 | CUDA Core 총수 | 용도 |
|-----|------|-----------|----|
| T4 | 40 | 2,560 | Colab 무료 |
| V100 | 80 | 5,120 | 이전 세대 프로덕션 |
| A100 | 108 | 6,912 | 현재 프로덕션 표준 |
| H100 | 132 | 16,896 | 최신 LLM 학습 |

**max_autotune이 T4에서 안 되는 이유**
```
max_autotune = 다양한 타일 크기(예: 16×16, 32×32, 64×64, 128×128)로
               실제 실행해보고 가장 빠른 것 선택

필요 SM 수 임계값: 약 80개 이상
T4: 40개 → 임계값 미달 → 탐색 포기 → 기본값 사용
```

---

### 10-2. TensorCore

**무엇인가?**
행렬 곱셈(GEMM)을 전용으로 처리하는 하드웨어 가속 유닛. AI 연산을 위해 설계.

```
일반 CUDA Core:
  덧셈 또는 곱셈 1개 처리 / cycle

TensorCore (T4 기준):
  4×4 FP16 행렬 곱 = 64개 연산 / cycle
  → 같은 시간에 64배 더 많은 연산

연산 설명: D = A × B + C
  A, B, C: 4×4 행렬
  cycle당 처리: 4×4×4 = 64 곱합(FMA) 연산
```

**세대별 발전**

| 세대 | GPU | 지원 정밀도 | 특징 |
|------|-----|------------|------|
| 1세대 | V100 | FP16 | TensorCore 최초 도입 |
| 2세대 | T4 | FP16, INT8 | 추론 최적화 |
| 3세대 | A100 | BF16, TF32, INT8 | 학습+추론 균형 |
| 4세대 | H100 | FP8 추가 | LLM 특화 |

---

### 10-3. VRAM vs RAM

```
RAM (시스템 메모리):
  CPU가 사용
  Python 변수, 데이터 로딩에 사용
  Colab: 12~52GB

VRAM (GPU 메모리):
  GPU가 사용
  모델 파라미터, 활성화값, gradient 저장
  T4: 16GB

데이터 이동:
  RAM → VRAM: PCIe 버스 (약 16GB/s, 병목)
  VRAM 내부: 약 320GB/s
```

**OOM (Out of Memory) 해결법**
```python
# 1. 그래디언트 불필요 시 (추론)
with torch.no_grad():
    output = model(input)

# 2. 배치 크기 줄이기
BATCH = 4  # 8 → 4

# 3. FP16으로 메모리 절반
model = model.half()

# 4. 사용 완료 후 즉시 삭제
del model, optimizer
torch.cuda.empty_cache()
import gc; gc.collect()

# 5. Gradient Checkpointing (학습 시)
model.gradient_checkpointing_enable()
# 활성화값 버리고 역전파 시 재계산 → 메모리 절반, 속도 20% 감소
```

---

## 11. 데이터 포맷 및 파일 형식

### 11-1. nbformat (Jupyter Notebook 형식)

**구조**
```json
{
  "nbformat": 4,
  "nbformat_minor": 5,
  "cells": [
    {
      "id": "a1b2c3d4",
      "cell_type": "code",
      "execution_count": null,
      "source": "print('hello')",
      "outputs": []
    }
  ]
}
```

**버전별 변경사항**

| nbformat_minor | 주요 변경 |
|----------------|----------|
| 4 | 기본 구조 |
| 5 | 셀 `id` 필드 필수 추가 |

**GitHub 렌더링 실패 원인**
```
nbformat v5.10.4가 노트북을 렌더링할 때 검증 실행
→ id 또는 execution_count 없으면 렌더링 거부
→ "This notebook could not be rendered" 오류
```

---

### 11-2. YAML (config.yaml)

**무엇인가?**
"YAML Ain't Markup Language" — 사람이 읽기 쉬운 설정 파일 형식.

```yaml
# setup/config.yaml
gcp:
  project_id: gen-lang-client-0318067486
  region: asia-northeast3
  bucket_name: media-ai-lab-bucket

bigquery:
  dataset_id: media_ai_lab
  location: asia-northeast3
```

**설정 파일 형식 비교**

| 형식 | 가독성 | 주석 | 타입 지원 | 주 사용처 |
|------|--------|------|---------|---------|
| JSON | 보통 | 불가 | 제한적 | API 응답, 설정 |
| YAML | 높음 | 가능 (#) | 풍부 | 설정 파일, CI/CD |
| TOML | 높음 | 가능 (#) | 명시적 | Rust 프로젝트, pyproject |
| INI | 높음 | 가능 (;) | 문자열 중심 | 구형 설정 |
| XML | 낮음 | 가능 | 명시적 | 엔터프라이즈, SOAP |

---

## 12. Python 생태계 도구

### 12-1. yt-dlp

**무엇인가?**
YouTube, Vimeo, Bilibili 등 1000개 이상 사이트에서 영상을 다운로드하는 Python 라이브러리.

```python
import yt_dlp

ydl_opts = {
    "format": "bestvideo[height<=720]+bestaudio/best",
    "merge_output_format": "mp4",
    "outtmpl": "/tmp/%(id)s_%(title)s.%(ext)s",
    "quiet": True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=True)
```

---

### 12-2. python-dotenv

**무엇인가?**
`.env` 파일에서 환경변수를 읽어 Python에 로드하는 라이브러리.

```bash
# .env 파일 (절대 git에 커밋하지 않음!)
GCP_PROJECT_ID=gen-lang-client-0318067486
BUCKET_NAME=my-media-bucket
```

```python
from dotenv import load_dotenv
import os

load_dotenv()
project_id = os.environ.get("GCP_PROJECT_ID")
```

---

### 12-3. tqdm

**무엇인가?**
"taqaddum" (아랍어: 진행) — Python 루프에 프로그레스 바를 추가하는 라이브러리.

```python
from tqdm import tqdm

for item in tqdm(large_list, desc="처리 중"):
    process(item)
# 처리 중: 73%|████████████████████░░░░░░| 730/1000 [01:12<00:27, 9.99it/s]
```

---

### 12-4. 주요 GCP Python 클라이언트 라이브러리

```python
from google.cloud import videointelligence
from google.cloud import speech
from google.cloud import translate_v3
from google.cloud import storage
from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel
```

**공통 패턴**
```python
# 1. 클라이언트 생성 (인증 포함)
client = storage.Client(project=project_id)

# 2. 요청 생성
bucket = client.bucket("my-bucket")

# 3. 실행
blob = bucket.blob("file.mp4")
blob.upload_from_filename("/local/file.mp4")
```

---

## 빠른 참조 카드

### 핵심 연산 설명 모음

| 이름 | 계산 방법 (말로) | 용도 |
|------|----------------|------|
| 암시적 평점 | 시청률×3 + 좋아요×2 + 공유×1 | 시청 행동 → 숫자 변환 |
| 코사인 유사도 | 내적 나누기 (A크기 × B크기) | 사용자 취향 유사도 |
| 가중 추천 점수 | (평점×유사도의 합) 나누기 (유사도의 합) | 유사 사용자 의견 통합 |
| Attention | 소프트맥스(QK전치 나누기 루트dk) 곱하기 V | 시퀀스 내 관계 포착 |
| LayerNorm | (x 빼기 평균) 나누기 표준편차 × 감마 더하기 베타 | 출력 분포 안정화 |

### GPU 선택 가이드

```
학습 필요 없는 추론만:       T4 + FP16  (비용 효율)
중소규모 모델 학습:          A10G + BF16 (균형)
LLM 학습 (7B 이상):         A100/H100 + BF16 + torch.compile
torch.compile 효과 측정:    A100 이상 필수 (SM 80개+)
커스텀 GPU 커널 개발:        Triton 우선 시도 → 성능 부족 시 CUDA
```

### 단위 변환

```
1 GHz = 10억 Hz = 초당 10억 번 연산
1 TFLOPS = 초당 1조 번 부동소수점 연산
1 GB = 약 10억 bytes
FP32 파라미터 1개 = 4 bytes
FP16 파라미터 1개 = 2 bytes
1억 파라미터 FP32 모델 = 400 MB VRAM
1억 파라미터 FP16 모델 = 200 MB VRAM
```

### 신경망 구조 선택 가이드

```
이미지/영상 분류·감지:       CNN (ResNet, EfficientNet)
텍스트 이해/생성:            Transformer (BERT, GPT)
시계열 예측:                 LSTM/GRU 또는 Transformer
음성 인식:                   Transformer (Whisper)
추천 시스템:                 ANN(MLP) + 임베딩 또는 협업 필터링
```

### 유사도 방법 선택 가이드

```
텍스트/문서 유사도:                코사인 유사도
위치/좌표 거리:                    유클리드 거리
대규모 벡터 검색 (100만 개 이상):  FAISS + ANN (HNSW/IVF)
집합 비교 (태그, 키워드):           Jaccard 유사도
이진 데이터 (해시):                해밍 거리
```

---

*문서 생성일: 2026-03-11*
