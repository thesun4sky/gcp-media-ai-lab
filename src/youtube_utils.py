"""
YouTube 영상 다운로드 및 GCS 업로드 유틸리티

YouTube URL을 받아 yt-dlp로 다운로드한 뒤 GCS에 업로드하여
GCS URI를 반환합니다. 이후 기존 GCP Media AI 파이프라인에서
그대로 사용할 수 있습니다.

사용 예시:
    from src.youtube_utils import youtube_to_gcs

    gcs_uri = youtube_to_gcs(
        youtube_url="https://youtu.be/xxxx",
        bucket_name="my-bucket",
        project_id="my-project",
    )
    # → "gs://my-bucket/youtube/xxxx_title.mp4"
"""

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def extract_video_id(youtube_url: str) -> str:
    """YouTube URL에서 비디오 ID를 추출합니다."""
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
        r"(?:shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, youtube_url)
        if m:
            return m.group(1)
    raise ValueError(f"YouTube URL에서 비디오 ID를 찾을 수 없습니다: {youtube_url}")


def download_youtube_video(
    youtube_url: str,
    output_dir: str,
    max_height: int = 720,
) -> str:
    """
    yt-dlp로 YouTube 영상을 다운로드합니다.

    Args:
        youtube_url: YouTube 영상 URL
        output_dir:  저장할 로컬 디렉토리
        max_height:  최대 해상도 (기본 720p — API 분석에 충분)

    Returns:
        다운로드된 파일의 로컬 경로
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp가 설치되지 않았습니다. "
            "pip install yt-dlp 로 설치하세요."
        )

    video_id = extract_video_id(youtube_url)
    output_template = str(Path(output_dir) / f"{video_id}_%(title)s.%(ext)s")

    ydl_opts = {
        "format": f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
    }

    logger.info(f"YouTube 영상 다운로드 중: {youtube_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = ydl.prepare_filename(info)
        # merge 후 확장자가 .mp4로 고정될 수 있음
        if not Path(filename).exists():
            filename = str(Path(filename).with_suffix(".mp4"))

    logger.info(f"다운로드 완료: {filename}")
    return filename


def youtube_to_gcs(
    youtube_url: str,
    bucket_name: str,
    project_id: Optional[str] = None,
    gcs_prefix: str = "youtube",
    max_height: int = 720,
) -> str:
    """
    YouTube 영상을 다운로드하여 GCS에 업로드하고 URI를 반환합니다.

    Args:
        youtube_url: YouTube 영상 URL (youtu.be 단축 URL 포함)
        bucket_name: 업로드할 GCS 버킷 이름
        project_id:  GCP 프로젝트 ID (None이면 환경 변수에서 읽음)
        gcs_prefix:  GCS 내 저장 경로 prefix (기본: "youtube")
        max_height:  다운로드 최대 해상도 (기본: 720p)

    Returns:
        GCS URI (gs://bucket/prefix/filename.mp4)
    """
    from src.gcp_client import upload_file_to_gcs

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1) 다운로드
        local_path = download_youtube_video(youtube_url, tmpdir, max_height)
        filename = Path(local_path).name

        # 2) GCS 업로드
        gcs_destination = f"{gcs_prefix}/{filename}"
        logger.info(f"GCS 업로드 중: gs://{bucket_name}/{gcs_destination}")
        gcs_uri = upload_file_to_gcs(
            local_file_path=local_path,
            bucket_name=bucket_name,
            destination_blob_name=gcs_destination,
            project_id=project_id,
        )

    logger.info(f"완료: {gcs_uri}")
    return gcs_uri
