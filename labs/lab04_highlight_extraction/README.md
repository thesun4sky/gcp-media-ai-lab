# Lab 4: 하이라이트 자동 추출 (Video Intelligence API + Vertex AI)

## 학습 목표

Video Intelligence API와 Vertex AI를 결합하여 영상에서 핵심 장면을 자동으로 감지하고
하이라이트 클립을 생성하는 파이프라인을 구축하는 방법을 배웁니다.

- **장면 점수 계산**: 각 장면의 중요도 점수 자동 계산
- **하이라이트 선택**: 점수 기반으로 최적의 장면 선택
- **클립 추출**: ffmpeg를 사용하여 선택된 장면 클립 추출
- **하이라이트 영상 합성**: 클립들을 하나의 영상으로 합치기

이 기술은 네이버 동영상 플랫폼의 자동 숏클립 생성 및 미리보기 기능에 활용됩니다.

---

## 사전 요구사항

### API 활성화
```bash
gcloud services enable videointelligence.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
```

### 시스템 요구사항
- ffmpeg 설치 필요 (클립 추출 및 합성용)
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`

---

## 파일 구조

```
lab04_highlight_extraction/
├── README.md               # 이 파일
└── extract_highlights.py   # 메인 하이라이트 추출 스크립트
```

---

## 하이라이트 점수 계산 방법

각 장면(Shot)에 대해 다음 요소를 기반으로 점수를 계산합니다:

### 1. 레이블 점수 (Label Score)
```
감지된 레이블의 평균 신뢰도 * 가중치
```
레이블이 많이 감지될수록, 신뢰도가 높을수록 높은 점수

### 2. 객체 추적 점수 (Object Score)
```
추적된 객체 수 * 평균 신뢰도 * 가중치
```
활동적인 장면(객체가 많은)에 높은 점수

### 3. 텍스트 감지 점수 (Text Score)
```
화면 텍스트 포함 여부 * 가중치
```
중요 정보가 표시되는 장면에 높은 점수

### 4. 장면 다양성 점수 (Diversity Score)
```
직전 선택된 장면과의 시간적 거리 * 가중치
```
비슷한 장면이 연속으로 선택되는 것 방지

**최종 점수 = Label Score + Object Score + Text Score + Diversity Score**

---

## 실습 단계

### 1단계: 기본 하이라이트 추출

```bash
python labs/lab04_highlight_extraction/extract_highlights.py \
    --video gs://your-bucket/video.mp4 \
    --project YOUR_PROJECT_ID \
    --bucket YOUR_BUCKET_NAME \
    --max-highlights 5 \
    --output-dir data/output/highlights/
```

### 2단계: 로컬 파일로 하이라이트 추출

```bash
python labs/lab04_highlight_extraction/extract_highlights.py \
    --video /path/to/your/video.mp4 \
    --project YOUR_PROJECT_ID \
    --bucket YOUR_BUCKET_NAME \
    --max-highlights 5 \
    --clip-duration 15 \
    --output-dir data/output/highlights/
```

### 3단계: 하이라이트 영상 합성

```bash
python labs/lab04_highlight_extraction/extract_highlights.py \
    --video /path/to/your/video.mp4 \
    --project YOUR_PROJECT_ID \
    --bucket YOUR_BUCKET_NAME \
    --max-highlights 5 \
    --merge-clips \
    --output-dir data/output/highlights/
```

### 4단계: 점수 임계값으로 필터링

```bash
python labs/lab04_highlight_extraction/extract_highlights.py \
    --video /path/to/video.mp4 \
    --project YOUR_PROJECT_ID \
    --bucket YOUR_BUCKET_NAME \
    --score-threshold 0.7 \
    --output-dir data/output/highlights/
```

---

## 출력 결과

실행 후 `data/output/highlights/` 디렉토리에 다음 파일이 생성됩니다:

```
highlights/
├── highlight_001_23s-38s.mp4     # 하이라이트 클립 1
├── highlight_002_95s-110s.mp4    # 하이라이트 클립 2
├── highlight_003_147s-162s.mp4   # 하이라이트 클립 3
├── merged_highlight.mp4          # 합쳐진 하이라이트 영상 (--merge-clips 사용 시)
└── highlight_analysis.json       # 분석 결과 JSON
```

---

## 하이라이트 분석 결과 JSON

```json
{
  "video_uri": "gs://...",
  "total_shots": 25,
  "selected_highlights": 5,
  "highlights": [
    {
      "rank": 1,
      "shot_index": 5,
      "start_time": 23.5,
      "end_time": 38.2,
      "score": 0.892,
      "score_breakdown": {
        "label_score": 0.45,
        "object_score": 0.28,
        "text_score": 0.10,
        "diversity_score": 0.062
      },
      "top_labels": ["action", "sport", "excitement"],
      "clip_path": "highlights/highlight_001_23s-38s.mp4"
    }
  ]
}
```

---

## GCP 비용 안내

| 서비스 | 사용량 | 예상 비용 |
|--------|--------|-----------|
| Video Intelligence API | 10분 영상 분석 | ~$1.00 (무료 한도 이후) |
| Cloud Storage | 클립 파일 저장 | ~$0.02/GB |

---

## 활용 시나리오

### 스포츠 하이라이트
- 골 장면, 파울 장면, 역전 장면 자동 감지
- 실시간 하이라이트 생성

### 뉴스 하이라이트
- 주요 발언 장면, 자료 화면 자동 추출
- 요약 클립 자동 생성

### 예능/드라마 하이라이트
- 웃음 포인트, 반전 장면 감지
- 예고편 자동 생성

---

## 다음 단계

Lab 4를 완료했다면 [Lab 5: 콘텐츠 추천 시스템](../lab05_content_recommendation/)으로 이동하세요.
