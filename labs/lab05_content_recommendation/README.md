# Lab 5: 콘텐츠 추천 시스템 (Vertex AI + BigQuery)

## 학습 목표

BigQuery에 저장된 시청 이력 데이터를 기반으로 Vertex AI를 활용하여
개인화 콘텐츠 추천 시스템을 구축하는 방법을 배웁니다.

- **데이터 준비**: 시청 이력, 영상 메타데이터, 사용자 프로필 데이터 구조 설계
- **협업 필터링**: 유사한 사용자의 시청 패턴 분석
- **콘텐츠 기반 필터링**: 영상 태그와 레이블을 활용한 유사 콘텐츠 추천
- **BigQuery ML**: SQL로 추천 모델 학습 및 예측
- **Vertex AI**: 확장 가능한 추천 모델 배포

이 기술은 네이버 동영상 플랫폼의 "비슷한 동영상", "추천 동영상" 기능에 활용됩니다.

---

## 사전 요구사항

### API 활성화
```bash
gcloud services enable bigquery.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

### BigQuery 데이터셋 생성
```bash
bq mk --dataset --location=asia-northeast3 YOUR_PROJECT_ID:media_ai_lab
```

---

## 파일 구조

```
lab05_content_recommendation/
├── README.md                   # 이 파일
└── recommendation_system.py    # 메인 추천 시스템 스크립트
```

---

## 데이터 구조

### 1. 시청 이력 테이블 (watch_history)

```sql
CREATE TABLE media_ai_lab.watch_history (
    user_id STRING NOT NULL,
    video_id STRING NOT NULL,
    watch_timestamp TIMESTAMP NOT NULL,
    watch_duration_seconds FLOAT64,
    completion_rate FLOAT64,    -- 0.0 ~ 1.0
    liked BOOL,
    disliked BOOL,
    shared BOOL
);
```

### 2. 영상 메타데이터 테이블 (video_metadata)

```sql
CREATE TABLE media_ai_lab.video_metadata (
    video_id STRING NOT NULL,
    title STRING,
    description STRING,
    tags ARRAY<STRING>,
    category STRING,
    duration_seconds FLOAT64,
    upload_timestamp TIMESTAMP,
    view_count INT64,
    like_count INT64,
    ai_labels ARRAY<STRING>,    -- Video Intelligence API에서 추출
    ai_summary STRING           -- Gemini로 생성한 요약
);
```

### 3. 사용자 프로필 테이블 (user_profiles)

```sql
CREATE TABLE media_ai_lab.user_profiles (
    user_id STRING NOT NULL,
    preferred_categories ARRAY<STRING>,
    avg_watch_duration FLOAT64,
    total_watch_count INT64,
    last_active TIMESTAMP
);
```

---

## 추천 알고리즘

### 1. 협업 필터링 (Collaborative Filtering)

사용자 A와 비슷한 시청 패턴을 가진 사용자 B가 본 영상 추천:

```sql
-- BigQuery ML로 행렬 분해 모델 학습
CREATE MODEL media_ai_lab.collab_filter_model
OPTIONS(
    model_type = 'matrix_factorization',
    user_col = 'user_id',
    item_col = 'video_id',
    rating_col = 'rating'
)
AS
SELECT
    user_id,
    video_id,
    (completion_rate * 5 + CAST(liked AS INT64) * 2) AS rating
FROM media_ai_lab.watch_history;
```

### 2. 콘텐츠 기반 필터링 (Content-Based Filtering)

시청한 영상과 비슷한 태그/카테고리를 가진 영상 추천:

```sql
-- 사용자가 좋아하는 카테고리의 인기 영상 추천
SELECT v.video_id, v.title, v.view_count
FROM media_ai_lab.video_metadata v
WHERE v.category IN (
    SELECT DISTINCT category
    FROM media_ai_lab.video_metadata
    WHERE video_id IN (
        SELECT video_id FROM media_ai_lab.watch_history
        WHERE user_id = @user_id AND completion_rate > 0.8
    )
)
AND v.video_id NOT IN (
    SELECT video_id FROM media_ai_lab.watch_history
    WHERE user_id = @user_id
)
ORDER BY v.view_count DESC
LIMIT 10;
```

---

## 실습 단계

### 1단계: 샘플 데이터 생성 및 BigQuery 로드

```bash
python labs/lab05_content_recommendation/recommendation_system.py \
    --project YOUR_PROJECT_ID \
    --action setup_data \
    --num-users 100 \
    --num-videos 500
```

### 2단계: 추천 모델 학습

```bash
python labs/lab05_content_recommendation/recommendation_system.py \
    --project YOUR_PROJECT_ID \
    --action train_model
```

### 3단계: 개인화 추천 실행

```bash
python labs/lab05_content_recommendation/recommendation_system.py \
    --project YOUR_PROJECT_ID \
    --action recommend \
    --user-id user_001 \
    --top-k 10
```

### 4단계: 배치 추천 (전체 사용자)

```bash
python labs/lab05_content_recommendation/recommendation_system.py \
    --project YOUR_PROJECT_ID \
    --action batch_recommend \
    --output-table media_ai_lab.recommendations
```

---

## GCP 비용 안내

| 서비스 | 사용량 | 예상 비용 |
|--------|--------|-----------|
| BigQuery 저장 | 1GB 이하 | 무료 |
| BigQuery 쿼리 | 1TB 이하/월 | 무료 |
| BigQuery ML 학습 | 처음 10GB | 무료 |
| Vertex AI 추론 | 최초 배포 이후 | 노드 시간 기준 |

---

## 다음 단계

Lab 5를 완료했다면 [Lab 6: 생성형 AI 콘텐츠 창작](../lab06_generative_ai/)으로 이동하세요.
