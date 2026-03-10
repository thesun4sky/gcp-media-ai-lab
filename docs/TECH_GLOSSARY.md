# GCP Media AI Lab — 기술 용어 및 개념 완전 정리

> 이 문서는 실습에 등장한 모든 기술, 용어, 모델, 연산식을 초보자도 이해할 수 있도록 상세하게 정리합니다.

---

## 목차

1. [GCP 서비스](#1-gcp-서비스)
2. [미디어 처리 기술](#2-미디어-처리-기술)
3. [인증 및 보안](#3-인증-및-보안)
4. [추천 시스템 — 수식 포함](#4-추천-시스템--수식-포함)
5. [딥러닝 기초](#5-딥러닝-기초)
6. [PyTorch 최적화 기술](#6-pytorch-최적화-기술)
7. [GPU 하드웨어 개념](#7-gpu-하드웨어-개념)
8. [데이터 포맷 및 파일 형식](#8-데이터-포맷-및-파일-형식)
9. [Python 생태계 도구](#9-python-생태계-도구)

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

**왜 "Fast Forward"인가?**
MPEG 표준의 빠른 순방향(fast forward) 처리를 목표로 개발했기 때문.

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
음질 손상 없이 파일 크기를 줄이는 무손실 오디오 압축 포맷. "무손실"이 핵심.

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

**STT에 FLAC를 쓰는 이유**
```
MP3/AAC: AI가 분석할 때 이미 손실된 데이터 → 인식 오류 가능
FLAC:    원본 그대로 → 더 높은 인식 정확도
```

**비교군**

| 포맷 | 압축 방식 | 파일크기 | 음질 | 주 용도 |
|------|----------|---------|------|---------|
| WAV | 무압축 | 100% | 완벽 | 스튜디오 원본 |
| FLAC | 무손실 | ~60% | 완벽 | STT, 아카이브 |
| AAC | 손실 | ~10% | 우수 | 스트리밍 |
| MP3 | 손실 | ~10% | 양호 | 범용 |
| OGG | 손실 | ~8% | 양호 | 게임, 웹 |

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

SRT와 거의 동일하지만 웹 표준(HTML5 `<track>` 태그) 지원.

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

## 4. 추천 시스템 — 수식 포함

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

---

### 4-2. 암시적 평점 (Implicit Rating)

**명시적 vs 암시적**
```
명시적 평점: 사용자가 직접 별점 부여 (1~5점)
  → 데이터 적음, 바이어스 있음 (극단적 평가 경향)

암시적 평점: 행동 데이터에서 간접적으로 추론
  → 데이터 많음, 자연스러운 선호도 반영
```

**본 프로젝트의 암시적 평점 공식**

$$\text{implicit\_rating} = \text{completion\_rate} \times 3.0 + \text{liked} \times 2.0 + \text{shared} \times 1.0$$

```
completion_rate: 영상 시청 완료율 (0.0 ~ 1.0)
liked: 좋아요 여부 (0 또는 1)
shared: 공유 여부 (0 또는 1)

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

**코사인 유사도 공식**

$$\text{cosine\_similarity}(A, B) = \frac{A \cdot B}{|A| \times |B|}$$

각 항목 설명:
```
A · B (내적, Dot Product):
  = A₁×B₁ + A₂×B₂ + ... + Aₙ×Bₙ
  = 두 벡터에서 공통으로 높은 값일수록 크게 기여

|A| (벡터의 크기, Norm):
  = √(A₁² + A₂² + ... + Aₙ²)
  = 벡터의 "길이"

÷ (|A| × |B|): 정규화
  → 많이 시청한 사람과 적게 시청한 사람을 공평하게 비교
```

**계산 예시**
```
user_A = [4.7, 0.0, 3.5]   (영상1, 영상2, 영상3만 고려)
user_B = [5.2, 3.1, 4.0]

내적 = 4.7×5.2 + 0.0×3.1 + 3.5×4.0
     = 24.44 + 0 + 14.0 = 38.44

|A| = √(4.7² + 0² + 3.5²) = √(22.09 + 0 + 12.25) = √34.34 = 5.86
|B| = √(5.2² + 3.1² + 4.0²) = √(27.04 + 9.61 + 16.0) = √52.65 = 7.26

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

**공식**

$$\text{confidence\_score}(v) = \frac{\sum_{u \in \text{similar\_users}} \text{rating}(u, v) \times \text{similarity}(me, u)}{\sum_{u \in \text{similar\_users}} \text{similarity}(me, u)}$$

```
쉬운 설명:
  "나와 90% 유사한 사람이 준 점수" + "나와 50% 유사한 사람이 준 점수"
  → 유사한 사람의 의견을 더 많이 반영

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
  R ≈ U × V^T
  U: 사용자 잠재 벡터 (사용자 취향 요약)
  V: 아이템 잠재 벡터 (영상 특성 요약)

→ U × V^T로 빈 칸 예측 가능
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

**수식**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

```
Q @ K^T: 모든 쿼리-키 쌍의 유사도 계산
√d_k로 나누기: 값이 너무 커지지 않도록 스케일링
softmax: 합이 1이 되는 확률 분포로 변환
× V: 확률에 따라 가중 평균
```

---

### 5-4. LayerNorm (레이어 정규화)

**왜 필요한가?**
딥러닝에서 각 층의 출력값 분포가 불안정해지면 학습이 어려워짐.

**공식**

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

```
μ (평균): 현재 레이어 출력의 평균
σ (표준편차): 현재 레이어 출력의 퍼짐 정도
(x - μ) / σ: 평균 0, 표준편차 1로 정규화
γ, β: 학습 가능한 파라미터 (스케일, 이동)
ε: 0으로 나누는 것 방지 (보통 1e-5)
```

---

### 5-5. GELU 활성화 함수

**무엇인가?**
뉴런 출력을 0~1 사이로 변환하는 비선형 함수. Transformer에서 ReLU 대신 사용.

**공식**

$$\text{GELU}(x) = x \cdot \Phi(x)$$

여기서 $\Phi(x)$는 표준 정규분포의 CDF (누적분포함수).

**근사식** (실제 구현)

$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)$$

**ReLU와 비교**
```
ReLU(x) = max(0, x)  → x<0이면 무조건 0 (딱딱한 경계)
GELU(x)              → x<0에서도 작은 값 통과 (부드러운 경계)
                     → Transformer 계열에서 더 좋은 성능
```

---

## 6. PyTorch 최적화 기술

### 6-1. torch.compile

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

---

### 6-2. Triton

**무엇인가?**
OpenAI가 개발한 GPU 프로그래밍 언어/컴파일러. CUDA보다 훨씬 쉽게 고성능 GPU 커널 작성 가능.

**CUDA vs Triton 비교**
```python
# CUDA C++ (저수준, 복잡)
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
// 스레드 관리, 메모리 레이아웃 직접 제어해야 함

# Triton Python (고수준, 간결)
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    tl.store(c_ptr + offsets, a + b)
# 타일 단위 추상화, Python 문법 사용
```

**Triton이 torch.compile에 포함된 이유**
```
연구자: "새 모델 구조 만들 때마다 CUDA 커널 직접 짜기 힘들어"
OpenAI: "Triton으로 자동 생성하면 되잖아"
PyTorch: "torch.compile 백엔드(inductor)로 Triton 채택"
→ 개발자가 CUDA 몰라도 자동으로 최적 GPU 커널 생성
```

---

### 6-3. 커널 퓨전 (Kernel Fusion)

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

### 6-4. FP32 / FP16 / BF16

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

**T4에서 FP16이 4배 빠른 이유**
```
TensorCore (T4 기준):
  FP32 행렬곱: 초당 65 TFLOPS
  FP16 행렬곱: 초당 130 TFLOPS (2배)

추가로:
  FP16은 메모리 절반 사용 → 메모리 대역폭 병목도 절반
  합산 효과 → 실측 ~4배 향상
```

---

### 6-5. AMP (Automatic Mixed Precision)

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

**순수 FP16 vs AMP**

| | 순수 FP16 | AMP |
|--|---------|-----|
| 속도 | 더 빠름 | 약간 느림 (형변환 오버헤드) |
| 안정성 | gradient 소실 위험 | 자동 보호 |
| 용도 | 추론 전용 | 학습 + 추론 |

---

## 7. GPU 하드웨어 개념

### 7-1. SM (Streaming Multiprocessor)

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

### 7-2. TensorCore

**무엇인가?**
행렬 곱셈(GEMM)을 전용으로 처리하는 하드웨어 가속 유닛. AI 연산을 위해 설계.

```
일반 CUDA Core:
  덧셈 또는 곱셈 1개 처리 / cycle

TensorCore (T4 기준):
  4×4 FP16 행렬 곱 = 64개 연산 / cycle
  → 같은 시간에 64배 더 많은 연산

수식: D = A × B + C
  A, B, C: 4×4 행렬
  cycle당 처리: 4×4×4 = 64 FMA(곱합) 연산
```

**세대별 발전**

| 세대 | GPU | 지원 정밀도 | 특징 |
|------|-----|------------|------|
| 1세대 | V100 | FP16 | TensorCore 최초 도입 |
| 2세대 | T4 | FP16, INT8 | 추론 최적화 |
| 3세대 | A100 | BF16, TF32, INT8 | 학습+추론 균형 |
| 4세대 | H100 | FP8 추가 | LLM 특화 |

---

### 7-3. VRAM vs RAM

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
gc.collect()

# 5. Gradient Checkpointing (학습 시)
model.gradient_checkpointing_enable()
# 활성화값 버리고 역전파 시 재계산 → 메모리 절반, 속도 20% 감소
```

---

## 8. 데이터 포맷 및 파일 형식

### 8-1. nbformat (Jupyter Notebook 형식)

**구조**
```json
{
  "nbformat": 4,
  "nbformat_minor": 5,
  "cells": [
    {
      "id": "a1b2c3d4",           ← nbformat 4.5부터 필수
      "cell_type": "code",
      "execution_count": null,   ← code 셀 필수
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

### 8-2. YAML (config.yaml)

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

labs:
  lab01:
    features:
      - SHOT_CHANGE_DETECTION
      - LABEL_DETECTION
      - OBJECT_TRACKING
```

**JSON vs YAML**
```
JSON: 기계가 파싱하기 쉬움, 주석 불가
YAML: 사람이 읽기 쉬움, 주석 가능 (#), 파일 더 짧음
TOML: YAML과 유사하지만 더 엄격한 문법 (Rust 생태계 선호)
```

---

## 9. Python 생태계 도구

### 9-1. yt-dlp

**무엇인가?**
YouTube, Vimeo, Bilibili 등 1000개 이상 사이트에서 영상을 다운로드하는 Python 라이브러리.

**원래 youtube-dl에서 파생된 프로젝트** — 더 빠르고 많은 기능.

```python
import yt_dlp

ydl_opts = {
    "format": "bestvideo[height<=720]+bestaudio/best",
    # 720p 이하 최고화질 영상 + 최고화질 음성 합성
    "merge_output_format": "mp4",  # 결과를 MP4로 저장
    "outtmpl": "/tmp/%(id)s_%(title)s.%(ext)s",  # 파일명 형식
    "quiet": True,  # 진행 출력 숨김
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=True)
    filename = ydl.prepare_filename(info)
```

**format 문자열 의미**
```
bestvideo[height<=720]: 720p 이하 최고화질 영상 스트림
+bestaudio:             + 최고화질 음성 스트림 (별도 파일)
/best:                  위가 안 되면 합쳐진 최고화질
```

---

### 9-2. python-dotenv

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

load_dotenv()  # .env 파일 로드

project_id = os.environ.get("GCP_PROJECT_ID")
```

**왜 환경변수에 저장하는가?**
```
하드코딩 (나쁜 방법):
  project_id = "gen-lang-client-0318067486"
  → Git에 올리면 누구나 볼 수 있음
  → 실제 비밀 키(API key)라면 보안 사고

환경변수 (.env):
  → .gitignore에 추가 → Git에 올라가지 않음
  → 팀원마다 다른 값 사용 가능
```

---

### 9-3. tqdm

**무엇인가?**
"taqaddum" (아랍어: 진행) — Python 루프에 프로그레스 바를 추가하는 라이브러리.

```python
from tqdm import tqdm

# 없을 때
for item in large_list:
    process(item)  # 얼마나 걸릴지 모름

# tqdm 사용
for item in tqdm(large_list, desc="처리 중"):
    process(item)
# 처리 중: 73%|████████████████████░░░░░░░| 730/1000 [01:12<00:27, 9.99it/s]
```

---

### 9-4. 주요 GCP Python 클라이언트 라이브러리

```python
# Video Intelligence
from google.cloud import videointelligence

# Speech-to-Text
from google.cloud import speech

# Translation
from google.cloud import translate_v3

# Cloud Storage
from google.cloud import storage

# BigQuery
from google.cloud import bigquery

# Vertex AI (Gemini)
import vertexai
from vertexai.generative_models import GenerativeModel
```

**모두 공통 패턴**
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

### 수식 모음

| 이름 | 수식 | 용도 |
|------|------|------|
| 암시적 평점 | `rate×3 + liked×2 + shared×1` | 시청 행동 → 숫자 변환 |
| 코사인 유사도 | `A·B / (|A|×|B|)` | 사용자 취향 유사도 |
| 가중 추천 점수 | `Σ(rating×sim) / Σ(sim)` | 유사 사용자 의견 통합 |
| Attention | `softmax(QK^T/√d) × V` | 시퀀스 내 관계 포착 |
| LayerNorm | `γ(x-μ)/σ + β` | 출력 분포 안정화 |

### 단위 변환

```
1 GHz = 10억 Hz = 초당 10억 번 연산
1 TFLOPS = 초당 1조 번 부동소수점 연산
1 GB = 1,073,741,824 bytes ≈ 10억 bytes
FP32 파라미터 1개 = 4 bytes
FP16 파라미터 1개 = 2 bytes
1억 파라미터 FP32 모델 = 400 MB VRAM
1억 파라미터 FP16 모델 = 200 MB VRAM
```

### GPU 선택 가이드

```
학습 필요 없는 추론만:      T4 + FP16  (비용 효율)
중소규모 모델 학습:         A10G + BF16 (균형)
LLM 학습 (7B 이상):        A100/H100 + BF16 + torch.compile
torch.compile 효과 측정:   A100 이상 필수 (SM 80개+)
```

---

*문서 생성일: 2026-03-11*
