# Lab 2: 자막 자동 생성 (Speech-to-Text API)

## 학습 목표

Google Cloud Speech-to-Text API를 사용하여 영상의 음성을 자동으로 텍스트로 변환하고
SRT/VTT 자막 파일을 생성하는 방법을 배웁니다.

- **음성 인식**: 영상/오디오에서 음성을 텍스트로 변환
- **타임스탬프**: 단어별 시작/종료 시간 추출
- **자막 파일 생성**: SRT(SubRip) 및 VTT(WebVTT) 형식으로 저장
- **긴 영상 처리**: 비동기 API로 긴 영상 처리

이 기술은 네이버 동영상 플랫폼에서 자동 자막 생성 기능에 활용됩니다.

---

## 사전 요구사항

### API 활성화
```bash
gcloud services enable speech.googleapis.com
gcloud services enable storage.googleapis.com
```

### 시스템 요구사항
- ffmpeg 설치 필요 (오디오 추출용)
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`
  - Windows: https://ffmpeg.org/download.html

---

## 파일 구조

```
lab02_speech_subtitle/
├── README.md               # 이 파일
├── generate_subtitle.py    # 메인 자막 생성 스크립트
└── sample.srt              # 예시 SRT 자막 파일
```

---

## 실습 단계

### 1단계: 환경 설정

```bash
# ffmpeg 설치 확인
ffmpeg -version

# 패키지 설치
pip install -r setup/requirements.txt
```

### 2단계: 자막 생성 실행

#### 기본 실행 (한국어, GCS URI)
```bash
python labs/lab02_speech_subtitle/generate_subtitle.py \
    --audio gs://your-bucket/your-audio.mp4 \
    --project YOUR_PROJECT_ID \
    --bucket YOUR_BUCKET_NAME \
    --language ko-KR
```

#### 로컬 영상/오디오 파일로 실행
```bash
python labs/lab02_speech_subtitle/generate_subtitle.py \
    --audio /path/to/video.mp4 \
    --project YOUR_PROJECT_ID \
    --bucket YOUR_BUCKET_NAME \
    --language ko-KR \
    --output data/output/subtitle.srt
```

#### 영어 음성 자막 생성
```bash
python labs/lab02_speech_subtitle/generate_subtitle.py \
    --audio gs://cloud-samples-data/speech/brooklyn_bridge.raw \
    --project YOUR_PROJECT_ID \
    --language en-US \
    --output data/output/english_subtitle.srt
```

#### SRT와 VTT 모두 생성
```bash
python labs/lab02_speech_subtitle/generate_subtitle.py \
    --audio /path/to/video.mp4 \
    --project YOUR_PROJECT_ID \
    --bucket YOUR_BUCKET_NAME \
    --output data/output/subtitle \
    --format both
```

---

## 지원 언어 코드

| 언어 | 코드 |
|------|------|
| 한국어 | ko-KR |
| 영어 (미국) | en-US |
| 영어 (영국) | en-GB |
| 일본어 | ja-JP |
| 중국어 간체 | zh-CN |
| 중국어 번체 | zh-TW |
| 스페인어 | es-ES |
| 프랑스어 | fr-FR |

---

## SRT 자막 형식 설명

```
1                               ← 자막 번호
00:00:01,500 --> 00:00:03,200   ← 시작시간 --> 종료시간 (HH:MM:SS,mmm)
안녕하세요, 반갑습니다.           ← 자막 텍스트

2
00:00:04,000 --> 00:00:06,500
오늘 Media AI에 대해 알아보겠습니다.

```

## VTT 자막 형식 설명

```
WEBVTT

1
00:00:01.500 --> 00:00:03.200
안녕하세요, 반갑습니다.

2
00:00:04.000 --> 00:00:06.500
오늘 Media AI에 대해 알아보겠습니다.
```

---

## API 요청 구성

### 짧은 오디오 (1분 이하)
```python
# synchronous recognize
response = client.recognize(config=config, audio=audio)
```

### 긴 오디오 (1분 초과)
```python
# Long-running (비동기) API 사용
# GCS에 파일 업로드 후 URI로 요청
operation = client.long_running_recognize(config=config, audio=audio)
response = operation.result(timeout=600)
```

---

## 자막 품질 향상 팁

1. **오디오 품질**: 배경 소음이 적을수록 인식률이 높습니다
2. **말하기 속도**: 천천히 명확하게 말할수록 정확합니다
3. **언어 힌트**: `speech_contexts`에 도메인 특화 단어를 추가하면 정확도가 높아집니다
4. **화자 분리**: `diarization_config`로 여러 화자를 구분할 수 있습니다
5. **전처리**: 오디오를 16kHz 모노로 변환하면 인식률이 향상됩니다

---

## GCP 비용 안내

| 기능 | 무료 한도 | 초과 비용 |
|------|-----------|-----------|
| Speech-to-Text (표준) | 최초 60분/월 | $0.006/15초 |
| Speech-to-Text (향상) | 없음 | $0.009/15초 |

> 5분 영상 = 20 * $0.006 = $0.12 (무료 한도 초과 시)

---

## 문제 해결

### 빈 결과가 반환되는 경우
- 오디오 파일이 손상되지 않았는지 확인
- 언어 코드가 정확한지 확인 (예: `ko-KR`, `en-US`)
- 오디오에 실제 음성이 포함되어 있는지 확인

### "Audio too long" 오류
- 1분 이상인 파일은 long_running_recognize API를 사용해야 합니다
- 또는 GCS URI를 사용하세요

### 인식률이 낮은 경우
```python
# 강화된 모델 사용
config = speech.RecognitionConfig(
    model="latest_long",  # 또는 "video" 모델
    use_enhanced=True,
)
```

---

## 다음 단계

Lab 2를 완료했다면 [Lab 3: 다국어 자막 & 번역](../lab03_multilingual_subtitle/)으로 이동하세요.
