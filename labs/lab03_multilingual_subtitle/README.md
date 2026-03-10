# Lab 3: 다국어 자막 & 번역 (Translation API)

## 학습 목표

Google Cloud Translation API를 사용하여 한국어 자막을 여러 언어로 자동 번역하고
다국어 자막 파일을 생성하는 방법을 배웁니다.

- **자동 번역**: 한국어 → 영어/일본어/중국어(간체/번체) 자동 번역
- **자막 파일 번역**: SRT/VTT 파일을 통째로 번역하면서 타임스탬프 보존
- **일괄 번역**: 여러 자막 항목을 한 번에 번역하여 API 비용 최소화
- **용어 일관성**: 도메인 특화 용어의 일관된 번역

이 기술은 네이버 동영상의 글로벌 서비스 확장을 위한 다국어 지원 기능에 활용됩니다.

---

## 사전 요구사항

### API 활성화
```bash
gcloud services enable translate.googleapis.com
```

### Lab 2 완료
이 랩은 Lab 2에서 생성한 SRT 자막 파일을 입력으로 사용합니다.
Lab 2를 완료하지 않았다면 `labs/lab02_speech_subtitle/sample.srt`를 사용할 수 있습니다.

---

## 파일 구조

```
lab03_multilingual_subtitle/
├── README.md               # 이 파일
└── translate_subtitle.py   # 메인 번역 스크립트
```

---

## 실습 단계

### 1단계: 자막 파일 번역

#### 기본 번역 (영어, 일본어, 중국어)
```bash
python labs/lab03_multilingual_subtitle/translate_subtitle.py \
    --input labs/lab02_speech_subtitle/sample.srt \
    --project YOUR_PROJECT_ID \
    --languages en ja zh-CN
```

#### 특정 언어만 번역
```bash
python labs/lab03_multilingual_subtitle/translate_subtitle.py \
    --input labs/lab02_speech_subtitle/sample.srt \
    --project YOUR_PROJECT_ID \
    --languages en \
    --output data/output/subtitle_en.srt
```

#### 여러 자막 파일 일괄 번역
```bash
python labs/lab03_multilingual_subtitle/translate_subtitle.py \
    --input data/output/*.srt \
    --project YOUR_PROJECT_ID \
    --languages en ja zh-CN \
    --output-dir data/output/translated/
```

#### VTT 형식으로 번역
```bash
python labs/lab03_multilingual_subtitle/translate_subtitle.py \
    --input labs/lab02_speech_subtitle/sample.srt \
    --project YOUR_PROJECT_ID \
    --languages en \
    --format vtt
```

---

## 지원 언어

| 언어 | 코드 | 네이버 글로벌 서비스 |
|------|------|---------------------|
| 영어 | en | LINE 글로벌 |
| 일본어 | ja | LINE 일본 |
| 중국어 간체 | zh-CN | 중국 시장 |
| 중국어 번체 | zh-TW | 대만/홍콩 |
| 스페인어 | es | 라틴아메리카 |
| 프랑스어 | fr | 유럽 |
| 독일어 | de | 유럽 |
| 태국어 | th | 동남아시아 |
| 베트남어 | vi | 동남아시아 |
| 인도네시아어 | id | 동남아시아 |

전체 지원 언어 목록: https://cloud.google.com/translate/docs/languages

---

## Translation API 특징

### v3 (Advanced) API
- **일괄 번역**: 여러 텍스트를 한 번 요청으로 처리
- **용어집(Glossary)**: 도메인 특화 용어 일관성 보장
- **번역 메모리**: 반복되는 문구 캐싱으로 비용 절감
- **형식 유지**: HTML 태그, 구두점 자동 처리

### 자막 번역 최적화 전략
```python
# 좋은 방법: 여러 자막을 묶어서 한 번에 번역
texts = [entry.text for entry in entries]
translations = translate_batch(texts, target_language="en")

# 나쁜 방법: 각 자막을 개별적으로 번역 (API 비용 증가)
for entry in entries:
    translation = translate(entry.text, target_language="en")
```

---

## 번역 품질 향상 팁

1. **자막 텍스트 정리**: 번역 전에 불필요한 공백이나 특수문자 제거
2. **문맥 유지**: 너무 짧은 자막 항목은 앞뒤 맥락이 없어 번역 품질 저하
3. **용어집 사용**: 고유명사나 브랜드명은 용어집으로 일관성 확보
4. **후처리 검토**: 자동 번역 후 전문 번역가 검토 권장

### 용어집(Glossary) 설정 예시
```python
# BigQuery에 용어집 데이터 업로드 후 Translation API와 연결
glossary_config = translate.TranslateTextGlossaryConfig(
    glossary=f"projects/{project_id}/locations/global/glossaries/media_terms"
)
```

---

## 출력 결과 예시

### 원본 (한국어)
```srt
1
00:00:01,200 --> 00:00:03,800
안녕하세요, Media AI 실습에 오신 것을 환영합니다.
```

### 영어 번역
```srt
1
00:00:01,200 --> 00:00:03,800
Hello, welcome to the Media AI practice.
```

### 일본어 번역
```srt
1
00:00:01,200 --> 00:00:03,800
こんにちは、Media AIの実習へようこそ。
```

### 중국어 간체 번역
```srt
1
00:00:01,200 --> 00:00:03,800
您好，欢迎来到Media AI实习。
```

---

## GCP 비용 안내

| 기능 | 무료 한도 | 초과 비용 |
|------|-----------|-----------|
| Translation v3 | 최초 500K 문자/월 | $20/1M 문자 |

> 1,000개 자막 × 평균 30자 × 3개 언어 = 90,000 문자 → 무료 한도 내

---

## 문제 해결

### 번역 결과가 이상한 경우
- 원본 자막이 너무 짧거나 문맥이 없는 경우 발생 가능
- 여러 자막 항목을 병합하여 더 긴 텍스트로 번역하면 품질 향상

### 특수 문자가 깨지는 경우
```bash
# 파일 인코딩 확인
file -i subtitle.srt

# UTF-8로 변환
iconv -f euc-kr -t utf-8 subtitle.srt > subtitle_utf8.srt
```

### API 할당량 초과 오류
```
RESOURCE_EXHAUSTED: 429 Quota exceeded
```
- 잠시 후 다시 시도하거나 일괄 처리 크기를 줄이세요
- GCP 콘솔에서 할당량 증가 요청 가능

---

## 다음 단계

Lab 3을 완료했다면 [Lab 4: 하이라이트 자동 추출](../lab04_highlight_extraction/)으로 이동하세요.
