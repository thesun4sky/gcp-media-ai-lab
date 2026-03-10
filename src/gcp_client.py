"""
GCP 클라이언트 공통 모듈

모든 GCP 서비스 클라이언트를 초기화하고 관리하는 공통 모듈입니다.
각 랩에서 이 모듈을 가져와 GCP 서비스에 접근합니다.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> dict:
    """
    설정 파일을 로드합니다.

    Args:
        config_path: config.yaml 파일 경로. None이면 기본 경로에서 검색합니다.

    Returns:
        설정 딕셔너리

    Raises:
        FileNotFoundError: 설정 파일을 찾을 수 없는 경우
    """
    if config_path is None:
        # 프로젝트 루트에서 config.yaml 검색
        project_root = Path(__file__).parent.parent
        possible_paths = [
            project_root / "setup" / "config.yaml",
            project_root / "config.yaml",
        ]
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path is None or not Path(config_path).exists():
        logger.warning(
            "config.yaml 파일을 찾을 수 없습니다. "
            "setup/config.yaml.example을 참고하여 setup/config.yaml을 생성하세요."
        )
        # 환경 변수에서 프로젝트 ID 읽기
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
        return {
            "gcp": {
                "project_id": project_id,
                "region": "asia-northeast3",
                "bucket_name": f"{project_id}-mediaai-lab",
            },
            "vertex_ai": {
                "location": "asia-northeast3",
                "gemini_model": "gemini-1.5-pro",
            },
            "bigquery": {
                "dataset_id": "media_ai_lab",
                "location": "asia-northeast3",
            },
        }

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"설정 파일 로드 완료: {config_path}")
    return config


def get_project_id(config: Optional[dict] = None) -> str:
    """
    GCP 프로젝트 ID를 반환합니다.

    우선순위: 환경 변수 > 설정 파일 > 기본값

    Args:
        config: 설정 딕셔너리. None이면 설정 파일에서 로드합니다.

    Returns:
        GCP 프로젝트 ID
    """
    # 환경 변수 확인
    env_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
    if env_project:
        return env_project

    # 설정 파일 확인
    if config is None:
        config = load_config()

    return config.get("gcp", {}).get("project_id", "YOUR_PROJECT_ID")


def get_storage_client(project_id: Optional[str] = None):
    """
    Google Cloud Storage 클라이언트를 반환합니다.

    Args:
        project_id: GCP 프로젝트 ID

    Returns:
        storage.Client 인스턴스
    """
    from google.cloud import storage

    if project_id is None:
        project_id = get_project_id()

    client = storage.Client(project=project_id)
    logger.debug(f"Cloud Storage 클라이언트 초기화 완료 (프로젝트: {project_id})")
    return client


def get_video_intelligence_client():
    """
    Google Cloud Video Intelligence 클라이언트를 반환합니다.

    Returns:
        videointelligence.VideoIntelligenceServiceClient 인스턴스
    """
    from google.cloud import videointelligence

    client = videointelligence.VideoIntelligenceServiceClient()
    logger.debug("Video Intelligence 클라이언트 초기화 완료")
    return client


def get_speech_client():
    """
    Google Cloud Speech-to-Text 클라이언트를 반환합니다.

    Returns:
        speech.SpeechClient 인스턴스
    """
    from google.cloud import speech

    client = speech.SpeechClient()
    logger.debug("Speech-to-Text 클라이언트 초기화 완료")
    return client


def get_translation_client():
    """
    Google Cloud Translation 클라이언트를 반환합니다.

    Returns:
        translate.TranslationServiceClient 인스턴스
    """
    from google.cloud import translate_v3 as translate

    client = translate.TranslationServiceClient()
    logger.debug("Translation 클라이언트 초기화 완료")
    return client


def get_bigquery_client(project_id: Optional[str] = None):
    """
    BigQuery 클라이언트를 반환합니다.

    Args:
        project_id: GCP 프로젝트 ID

    Returns:
        bigquery.Client 인스턴스
    """
    from google.cloud import bigquery

    if project_id is None:
        project_id = get_project_id()

    client = bigquery.Client(project=project_id)
    logger.debug(f"BigQuery 클라이언트 초기화 완료 (프로젝트: {project_id})")
    return client


def get_vertex_ai_client(project_id: Optional[str] = None, location: str = "asia-northeast3"):
    """
    Vertex AI를 초기화합니다.

    Args:
        project_id: GCP 프로젝트 ID
        location: Vertex AI 처리 위치

    Returns:
        초기화된 vertexai 모듈
    """
    import vertexai

    if project_id is None:
        project_id = get_project_id()

    vertexai.init(project=project_id, location=location)
    logger.debug(f"Vertex AI 초기화 완료 (프로젝트: {project_id}, 위치: {location})")
    return vertexai


def upload_file_to_gcs(
    local_file_path: str,
    bucket_name: str,
    destination_blob_name: Optional[str] = None,
    project_id: Optional[str] = None,
) -> str:
    """
    로컬 파일을 Google Cloud Storage에 업로드합니다.

    Args:
        local_file_path: 업로드할 로컬 파일 경로
        bucket_name: GCS 버킷 이름
        destination_blob_name: GCS 내 파일 경로. None이면 파일 이름 그대로 사용
        project_id: GCP 프로젝트 ID

    Returns:
        업로드된 파일의 GCS URI (gs://bucket/path 형식)

    Raises:
        FileNotFoundError: 로컬 파일이 존재하지 않는 경우
    """
    local_path = Path(local_file_path)
    if not local_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {local_file_path}")

    if destination_blob_name is None:
        destination_blob_name = local_path.name

    storage_client = get_storage_client(project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(str(local_path))

    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    logger.info(f"파일 업로드 완료: {local_file_path} -> {gcs_uri}")
    return gcs_uri


def download_file_from_gcs(
    gcs_uri: str,
    local_file_path: str,
    project_id: Optional[str] = None,
) -> str:
    """
    Google Cloud Storage에서 파일을 다운로드합니다.

    Args:
        gcs_uri: GCS URI (gs://bucket/path 형식)
        local_file_path: 저장할 로컬 파일 경로
        project_id: GCP 프로젝트 ID

    Returns:
        다운로드된 로컬 파일 경로

    Raises:
        ValueError: GCS URI 형식이 올바르지 않은 경우
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"올바른 GCS URI 형식이 아닙니다: {gcs_uri}")

    # gs://bucket/path 형식 파싱
    path_parts = gcs_uri[5:].split("/", 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1] if len(path_parts) > 1 else ""

    storage_client = get_storage_client(project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    local_path = Path(local_file_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    blob.download_to_filename(str(local_path))
    logger.info(f"파일 다운로드 완료: {gcs_uri} -> {local_file_path}")
    return str(local_path)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    로깅을 설정합니다.

    Args:
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
        log_file: 로그 파일 경로. None이면 콘솔만 출력
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logger.info(f"로깅 설정 완료 (레벨: {level})")
