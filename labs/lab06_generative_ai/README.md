# Lab 6: 생성형 AI 콘텐츠 창작 (Vertex AI Gemini)

## 학습 목표

Vertex AI의 Gemini 1.5 Pro를 활용하여 영상 관련 콘텐츠를 자동 생성하는 방법을 배웁니다.

- **영상 요약**: 영상 내용을 간결하게 요약하는 텍스트 자동 생성
- **자동 태그 생성**: 영상 내용 기반으로 검색 태그 자동 생성
- **썸네일 설명 생성**: 효과적인 썸네일을 위한 AI 설명 생성
- **제목 최적화**: 클릭률을 높이는 제목 변형 제안
- **자막 기반 콘텐츠**: 자막 데이터를 활용한 다양한 콘텐츠 생성

이 기술은 네이버 동영상 플랫폼의 크리에이터 지원 도구와
콘텐츠 자동화 기능에 활용됩니다.

---

## 사전 요구사항

### API 활성화
```bash
gcloud services enable aiplatform.googleapis.com
```

### Gemini API 접근 권한
```bash
# Vertex AI 사용자 권한 부여
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="user:YOUR_EMAIL" \
    --role="roles/aiplatform.user"
```

---

## 파일 구조

```
lab06_generative_ai/
├── README.md               # 이 파일
└── gemini_media_ai.py      # 메인 생성형 AI 스크립트
```

---

## 지원 기능

### 1. 영상 요약 (Video Summarization)
자막 또는 영상 설명을 입력으로 받아 핵심 내용을 요약합니다.

```bash
python labs/lab06_generative_ai/gemini_media_ai.py \
    --project YOUR_PROJECT_ID \
    --action summarize \
    --subtitle labs/lab02_speech_subtitle/sample.srt \
    --language ko
```

### 2. 태그 자동 생성 (Auto Tagging)
영상 내용을 분석하여 검색 최적화 태그를 생성합니다.

```bash
python labs/lab06_generative_ai/gemini_media_ai.py \
    --project YOUR_PROJECT_ID \
    --action generate_tags \
    --subtitle labs/lab02_speech_subtitle/sample.srt \
    --num-tags 20
```

### 3. 썸네일 설명 생성 (Thumbnail Description)
효과적인 클릭을 유도하는 썸네일 이미지 설명을 생성합니다.

```bash
python labs/lab06_generative_ai/gemini_media_ai.py \
    --project YOUR_PROJECT_ID \
    --action thumbnail_desc \
    --title "Media AI 완벽 가이드" \
    --summary "GCP를 활용한 AI 영상 분석 실습"
```

### 4. 제목 최적화 (Title Optimization)
클릭률을 높이는 제목 변형 옵션을 제안합니다.

```bash
python labs/lab06_generative_ai/gemini_media_ai.py \
    --project YOUR_PROJECT_ID \
    --action optimize_title \
    --title "Media AI 소개" \
    --subtitle labs/lab02_speech_subtitle/sample.srt
```

### 5. 콘텐츠 설명 생성 (Content Description)
영상 설명란에 사용할 SEO 최적화 설명을 생성합니다.

```bash
python labs/lab06_generative_ai/gemini_media_ai.py \
    --project YOUR_PROJECT_ID \
    --action generate_description \
    --subtitle labs/lab02_speech_subtitle/sample.srt \
    --title "Media AI 실습 가이드"
```

### 6. 댓글 분석 및 응답 초안 생성 (Comment Analysis)
시청자 댓글을 분석하고 크리에이터 응답 초안을 생성합니다.

```bash
python labs/lab06_generative_ai/gemini_media_ai.py \
    --project YOUR_PROJECT_ID \
    --action analyze_comments \
    --comments "정말 유익한 영상이에요! 다음에는 Vertex AI도 다뤄주세요."
```

### 7. 전체 파이프라인 (All-in-One)
모든 기능을 한 번에 실행합니다.

```bash
python labs/lab06_generative_ai/gemini_media_ai.py \
    --project YOUR_PROJECT_ID \
    --action all \
    --subtitle labs/lab02_speech_subtitle/sample.srt \
    --title "Media AI 실습 가이드" \
    --output data/output/gemini_results.json
```

---

## Gemini 1.5 Pro 활용 전략

### 프롬프트 엔지니어링 원칙

```python
# 1. 역할 부여 (Role Prompting)
system_prompt = """당신은 동영상 플랫폼의 AI 콘텐츠 분석가입니다.
영상 자막을 분석하여 시청자에게 유용한 정보를 제공합니다."""

# 2. 구조화된 출력 요청 (Structured Output)
prompt = """
다음 자막을 분석하여 아래 JSON 형식으로 응답하세요:
{
    "summary": "영상 요약 (3-5문장)",
    "tags": ["태그1", "태그2", ...],
    "key_topics": ["주제1", "주제2"]
}
"""

# 3. Few-shot 예시 제공 (Few-shot Prompting)
prompt = """
영상 태그 생성 예시:
입력: "요리 초보자를 위한 파스타 레시피"
출력: ["요리", "파스타", "초보요리", "레시피", "이탈리안"]

이제 다음 영상의 태그를 생성해주세요:
입력: {subtitle_text}
출력:
"""
```

### 다국어 생성 지원

```python
# 한국어 콘텐츠를 영어로도 생성
prompt = "영상 요약을 한국어와 영어로 각각 생성하세요."
```

---

## 출력 예시

### 영상 요약
```
이 영상은 Google Cloud Platform의 Media AI 기술을 소개합니다.
Video Intelligence API를 사용하여 영상을 자동으로 분석하고,
Speech-to-Text API로 자막을 자동 생성하는 방법을 실습합니다.
Translation API를 통한 다국어 지원과 Gemini를 활용한
생성형 AI 기능도 다룹니다.
```

### 자동 태그
```
["GCP", "Media AI", "Video Intelligence", "Speech-to-Text",
 "Translation API", "Gemini", "Vertex AI", "자막 자동생성",
 "AI영상분석", "클라우드", "머신러닝", "딥러닝", "자동화",
 "콘텐츠제작", "네이버동영상"]
```

---

## GCP 비용 안내

| 모델 | 입력 | 출력 | 무료 한도 |
|------|------|------|-----------|
| Gemini 1.5 Pro | $3.50/1M 토큰 | $10.50/1M 토큰 | - |
| Gemini 1.5 Flash | $0.075/1M 토큰 | $0.30/1M 토큰 | - |
| Gemini 1.0 Pro | $0.50/1K 자 | $1.50/1K 자 | - |

> 실습에는 Gemini 1.5 Flash 사용을 권장합니다 (비용 절약)

---

## 창작 AI 활용 시나리오

### 크리에이터 지원 도구
- 영상 업로드 후 자동 메타데이터 생성
- SEO 최적화 태그 및 제목 추천
- 시청자 댓글 자동 분류 및 응답 초안

### 플랫폼 자동화
- 영상 콘텐츠 자동 분류
- 부적절 콘텐츠 자동 필터링
- 개인화 추천을 위한 콘텐츠 임베딩

---

## 다음 단계

모든 랩을 완료했습니다! 이제 다음을 시도해보세요:

1. **통합 파이프라인**: Lab 1 ~ Lab 6을 연결하여 완전 자동화 파이프라인 구축
2. **실제 영상 적용**: 실제 영상 파일로 전체 파이프라인 실행
3. **API 서버 구축**: Flask/FastAPI로 Media AI API 서버 개발
4. **대시보드 구축**: Looker Studio로 분석 결과 시각화
