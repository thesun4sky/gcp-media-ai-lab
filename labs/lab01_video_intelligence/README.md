# Lab 1: 영상 AI 분석 (Google Cloud Video Intelligence API)

## 학습 목표

Google Cloud Video Intelligence API를 사용하여 영상에서 다음을 자동으로 감지하는 방법을 배웁니다:

- **장면 전환 감지(Shot Change Detection)**: 영상의 장면이 바뀌는 시점 자동 감지
- **레이블 감지(Label Detection)**: 영상에 등장하는 물체, 장소, 활동 자동 태깅
- **객체 추적(Object Tracking)**: 영상 내 특정 객체를 프레임 간 추적
- **텍스트 감지(Text Detection)**: 영상 화면에 나타나는 텍스트 인식

이 기술은 네이버 동영상 플랫폼의 자동 태깅, 썸네일 추천, 콘텐츠 검색 등에 활용됩니다.

---

## 사전 요구사항

### API 활성화
```bash
gcloud services enable videointelligence.googleapis.com
```

### 권한 설정
서비스 계정에 다음 권한이 필요합니다:
- `roles/videointelligence.user`
- `roles/storage.objectViewer` (GCS URI 사용 시)

---

## 파일 구조

```
lab01_video_intelligence/
├── README.md           # 이 파일
├── analyze_video.py    # 메인 분석 스크립트
└── sample_output.json  # 예시 출력 결과
```

---

## 실습 단계

### 1단계: 환경 설정

```bash
# 프로젝트 루트로 이동
cd /path/to/mediaAI

# Python 패키지 설치 (아직 설치 안 했다면)
pip install -r setup/requirements.txt

# GCP 인증
gcloud auth application-default login
```

### 2단계: 샘플 영상 준비

실습을 위한 영상 파일이 필요합니다. 다음 중 하나를 준비하세요:
- **로컬 파일**: MP4, AVI, MOV 등 영상 파일
- **GCS URI**: `gs://your-bucket/your-video.mp4` 형식

> 팁: YouTube에서 짧은 영상을 다운로드하거나, GCS에 공개된 샘플 영상을 사용하세요.
> 공개 샘플: `gs://cloud-samples-data/video/animals.mp4`

### 3단계: 영상 분석 실행

#### 기본 분석 (모든 기능)
```bash
python labs/lab01_video_intelligence/analyze_video.py \
    --video gs://cloud-samples-data/video/animals.mp4 \
    --project YOUR_PROJECT_ID
```

#### 특정 기능만 선택
```bash
# 장면 전환 + 레이블 감지만
python labs/lab01_video_intelligence/analyze_video.py \
    --video gs://cloud-samples-data/video/animals.mp4 \
    --project YOUR_PROJECT_ID \
    --features SHOT_CHANGE_DETECTION LABEL_DETECTION
```

#### 로컬 파일 분석
```bash
python labs/lab01_video_intelligence/analyze_video.py \
    --video /path/to/your/video.mp4 \
    --project YOUR_PROJECT_ID \
    --bucket YOUR_BUCKET_NAME
```

#### 결과를 JSON 파일로 저장
```bash
python labs/lab01_video_intelligence/analyze_video.py \
    --video gs://cloud-samples-data/video/animals.mp4 \
    --project YOUR_PROJECT_ID \
    --output data/output/video_analysis.json
```

### 4단계: 결과 확인

분석이 완료되면 다음 정보가 출력됩니다:

- **장면 전환**: 각 장면의 시작/종료 시간
- **레이블**: 감지된 객체/활동과 신뢰도 점수
- **객체 추적**: 추적된 객체의 위치 정보
- **텍스트**: 화면에 나타난 텍스트

---

## 코드 설명

### 주요 API 기능

#### Shot Change Detection
```python
features = [videointelligence.Feature.SHOT_CHANGE_DETECTION]
```
장면이 전환되는 시점(컷)을 자동으로 감지합니다.
하이라이트 추출이나 자동 챕터 생성에 활용됩니다.

#### Label Detection
```python
features = [videointelligence.Feature.LABEL_DETECTION]
```
영상에서 물체, 장면, 활동 등을 감지하고 레이블을 붙입니다.
자동 태깅이나 검색 인덱싱에 활용됩니다.

#### Object Tracking
```python
features = [videointelligence.Feature.OBJECT_TRACKING]
```
감지된 객체를 프레임 간 추적하여 이동 경로를 파악합니다.
스포츠 영상 분석이나 특정 물체 감지에 활용됩니다.

#### Text Detection
```python
features = [videointelligence.Feature.TEXT_DETECTION]
```
화면에 나타나는 텍스트(자막, 로고, 간판 등)를 인식합니다.
검색 가능한 콘텐츠 인덱싱에 활용됩니다.

---

## 출력 결과 형식

```json
{
  "video_uri": "gs://...",
  "analysis_time": "2025-01-01T00:00:00",
  "shot_changes": [
    {
      "shot_index": 0,
      "start_time": 0.0,
      "end_time": 3.5,
      "duration": 3.5
    }
  ],
  "labels": [
    {
      "description": "dog",
      "confidence": 0.98,
      "type": "shot_level"
    }
  ],
  "objects": [
    {
      "description": "cat",
      "confidence": 0.95,
      "frames": [...]
    }
  ],
  "texts": [
    {
      "text": "Hello World",
      "confidence": 0.99,
      "segments": [...]
    }
  ]
}
```

---

## GCP 비용 안내

| 기능 | 무료 한도 | 초과 비용 |
|------|-----------|-----------|
| Shot Change Detection | 최초 1,000분/월 | $0.10/분 |
| Label Detection | 최초 1,000분/월 | $0.10/분 |
| Object Tracking | 최초 1,000분/월 | $0.10/분 |
| Text Detection | 최초 1,000분/월 | $0.10/분 |

> 5분짜리 영상을 4개 기능으로 분석하면 약 20분 분량의 API 사용 → 무료 한도 내

---

## 문제 해결

### API가 활성화되지 않은 경우
```
PERMISSION_DENIED: Video Intelligence API has not been used in project
```
```bash
gcloud services enable videointelligence.googleapis.com
```

### 로컬 파일을 GCS에 업로드할 수 없는 경우
```
AccessDenied: Access denied
```
```bash
gsutil iam ch user:$(gcloud config get-value account):objectCreator gs://YOUR_BUCKET
```

### 영상 처리 시간 초과
긴 영상은 처리에 시간이 걸립니다. 먼저 짧은 클립(5분 이하)으로 테스트하세요.

---

## 다음 단계

Lab 1을 완료했다면 [Lab 2: 자막 자동 생성](../lab02_speech_subtitle/)으로 이동하세요.
