#!/bin/bash
# GCP Media AI 실습 환경 설정 스크립트
# 사용법: bash setup/setup_gcp.sh

set -e

# 색상 출력 유틸리티
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo ""
echo "====================================="
echo "  GCP Media AI 실습 환경 설정"
echo "====================================="
echo ""

# 1. gcloud CLI 설치 확인
log_info "gcloud CLI 설치 확인 중..."
if ! command -v gcloud &> /dev/null; then
    log_error "gcloud CLI가 설치되어 있지 않습니다."
    echo "설치 방법: https://cloud.google.com/sdk/docs/install"
    exit 1
fi
log_success "gcloud CLI 확인 완료: $(gcloud --version | head -1)"

# 2. 현재 프로젝트 확인
log_info "현재 GCP 프로젝트 확인 중..."
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    log_error "GCP 프로젝트가 설정되지 않았습니다."
    echo "다음 명령어로 프로젝트를 설정하세요:"
    echo "  gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi
log_success "현재 프로젝트: $PROJECT_ID"

# 3. 인증 확인
log_info "GCP 인증 상태 확인 중..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q "@"; then
    log_warning "활성화된 GCP 인증이 없습니다. 로그인을 진행합니다..."
    gcloud auth login
fi
ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1)
log_success "인증된 계정: $ACTIVE_ACCOUNT"

# 4. 필요한 API 활성화
log_info "필요한 GCP API를 활성화하는 중..."

APIS=(
    "videointelligence.googleapis.com"
    "speech.googleapis.com"
    "translate.googleapis.com"
    "aiplatform.googleapis.com"
    "bigquery.googleapis.com"
    "storage.googleapis.com"
    "storage-component.googleapis.com"
)

for API in "${APIS[@]}"; do
    log_info "API 활성화 중: $API"
    gcloud services enable "$API" --project="$PROJECT_ID" 2>/dev/null && \
        log_success "  $API 활성화 완료" || \
        log_warning "  $API 활성화 실패 또는 이미 활성화됨"
done

# 5. Cloud Storage 버킷 생성 (config.yaml이 있는 경우 버킷 이름 읽기)
BUCKET_NAME="${PROJECT_ID}-mediaai-lab"
log_info "Cloud Storage 버킷 확인 중: gs://$BUCKET_NAME"

if gsutil ls "gs://$BUCKET_NAME" &>/dev/null; then
    log_success "버킷이 이미 존재합니다: gs://$BUCKET_NAME"
else
    log_info "버킷 생성 중: gs://$BUCKET_NAME"
    gsutil mb -p "$PROJECT_ID" -l asia-northeast3 "gs://$BUCKET_NAME" && \
        log_success "버킷 생성 완료: gs://$BUCKET_NAME" || \
        log_warning "버킷 생성 실패. 수동으로 생성하거나 기존 버킷 이름을 config.yaml에 설정하세요."
fi

# 6. 샘플 영상 디렉토리 생성
log_info "샘플 데이터 디렉토리 구조 생성 중..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

mkdir -p "$PROJECT_ROOT/data/sample_videos"
mkdir -p "$PROJECT_ROOT/data/sample_audio"
mkdir -p "$PROJECT_ROOT/data/output"

log_success "디렉토리 구조 생성 완료"

# 7. Application Default Credentials 설정
log_info "애플리케이션 기본 인증(ADC) 설정 중..."
if gcloud auth application-default print-access-token &>/dev/null; then
    log_success "ADC가 이미 설정되어 있습니다."
else
    log_warning "ADC가 설정되지 않았습니다. 설정을 시작합니다..."
    gcloud auth application-default login
    gcloud auth application-default set-quota-project "$PROJECT_ID"
fi

# 8. config.yaml 생성 안내
if [ ! -f "$PROJECT_ROOT/setup/config.yaml" ]; then
    log_info "config.yaml 파일 생성 중..."
    cat > "$PROJECT_ROOT/setup/config.yaml" << EOF
# GCP Media AI 실습 설정 파일
# 아래 값들을 실제 프로젝트 정보로 수정하세요.

gcp:
  project_id: "$PROJECT_ID"
  region: "asia-northeast3"
  bucket_name: "$BUCKET_NAME"

# Video Intelligence API 설정
video_intelligence:
  features:
    - SHOT_CHANGE_DETECTION
    - LABEL_DETECTION
    - OBJECT_TRACKING
    - TEXT_DETECTION
  location: "asia-east1"

# Speech-to-Text API 설정
speech:
  language_code: "ko-KR"
  encoding: "LINEAR16"
  sample_rate_hertz: 16000
  enable_word_time_offsets: true
  enable_automatic_punctuation: true

# Translation API 설정
translation:
  source_language: "ko"
  target_languages:
    - "en"
    - "ja"
    - "zh-CN"

# Vertex AI 설정
vertex_ai:
  location: "asia-northeast3"
  gemini_model: "gemini-1.5-pro"

# BigQuery 설정
bigquery:
  dataset_id: "media_ai_lab"
  location: "asia-northeast3"

# 로컬 설정
local:
  data_dir: "data"
  output_dir: "data/output"
  sample_videos_dir: "data/sample_videos"
EOF
    log_success "config.yaml 파일이 생성되었습니다: $PROJECT_ROOT/setup/config.yaml"
else
    log_success "config.yaml 파일이 이미 존재합니다."
fi

echo ""
echo "====================================="
echo "  설정 완료!"
echo "====================================="
echo ""
echo "다음 단계:"
echo "  1. setup/config.yaml 파일을 열어 설정 값 확인"
echo "  2. Python 패키지 설치: pip install -r setup/requirements.txt"
echo "  3. Lab 1부터 시작: labs/lab01_video_intelligence/"
echo ""
log_success "GCP Media AI 실습 환경 설정이 완료되었습니다!"
