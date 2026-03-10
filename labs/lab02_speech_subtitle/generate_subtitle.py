#!/usr/bin/env python3
"""
Lab 2: 자막 자동 생성 - Speech-to-Text API

이 스크립트는 영상 또는 오디오 파일에서 음성을 텍스트로 변환하고
SRT/VTT 형식의 자막 파일을 자동 생성합니다.

사용법:
    python generate_subtitle.py --audio /path/to/video.mp4 --project PROJECT_ID

DAN25 연관:
    네이버 동영상 플랫폼의 자동 자막 생성 기능의 핵심 기술입니다.
    영상 업로드 후 AI가 자동으로 자막을 생성하여 접근성과 UX를 개선합니다.
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

# 상위 디렉토리의 src 모듈을 임포트하기 위한 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gcp_client import (
    get_project_id,
    get_speech_client,
    load_config,
    setup_logging,
    upload_file_to_gcs,
)
from src.subtitle_utils import (
    SubtitleEntry,
    merge_short_entries,
    save_srt,
    save_vtt,
)
from src.video_processor import check_ffmpeg, extract_audio

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """CLI 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="Speech-to-Text API를 사용한 자막 자동 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # GCS URI로 한국어 자막 생성
  python generate_subtitle.py \\
      --audio gs://your-bucket/video.mp4 \\
      --project my-project-id \\
      --language ko-KR \\
      --output data/output/subtitle.srt

  # 로컬 파일로 자막 생성 (GCS에 자동 업로드)
  python generate_subtitle.py \\
      --audio /path/to/video.mp4 \\
      --project my-project-id \\
      --bucket my-bucket \\
      --language ko-KR

  # SRT와 VTT 모두 생성
  python generate_subtitle.py \\
      --audio /path/to/video.mp4 \\
      --project my-project-id \\
      --bucket my-bucket \\
      --output data/output/subtitle \\
      --format both
        """,
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="음성 파일 경로 또는 GCS URI (MP4, WAV, MP3 등)",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="GCP 프로젝트 ID",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="로컬 파일 업로드용 GCS 버킷 이름",
    )
    parser.add_argument(
        "--language",
        default="ko-KR",
        help="음성 언어 코드 (기본: ko-KR). 예: ko-KR, en-US, ja-JP",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="출력 자막 파일 경로 (확장자 없이 지정 시 형식에 따라 자동 추가)",
    )
    parser.add_argument(
        "--format",
        choices=["srt", "vtt", "both"],
        default="srt",
        help="자막 파일 형식 (기본: srt)",
    )
    parser.add_argument(
        "--model",
        default="default",
        choices=["default", "latest_long", "video", "phone_call"],
        help="음성 인식 모델 (기본: default)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=2.0,
        help="자막 항목 최대 길이 초 (기본: 2.0초)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="자막 항목 최소 길이 초 (기본: 0.5초). 짧은 항목 병합 기준",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config.yaml 파일 경로",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="로그 레벨 (기본: INFO)",
    )

    return parser.parse_args()


def prepare_audio_for_api(
    audio_path: str,
    project_id: str,
    bucket_name: Optional[str],
    config: dict,
) -> tuple:
    """
    API 요청을 위한 오디오를 준비합니다.

    로컬 파일인 경우 WAV 변환 후 GCS에 업로드합니다.
    영상 파일인 경우 오디오를 먼저 추출합니다.

    Args:
        audio_path: 오디오/영상 파일 경로 또는 GCS URI
        project_id: GCP 프로젝트 ID
        bucket_name: GCS 버킷 이름
        config: 설정 딕셔너리

    Returns:
        (gcs_uri, audio_encoding, sample_rate) 튜플
    """
    from google.cloud import speech

    if audio_path.startswith("gs://"):
        # GCS URI: MP4/FLAC 등 다양한 포맷 → encoding=None으로 API 자동 감지
        logger.info(f"GCS URI 사용: {audio_path}")
        return audio_path, None, 16000

    # 로컬 파일 처리
    local_path = Path(audio_path)
    if not local_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_path}")

    temp_files = []

    try:
        # 영상 파일인 경우 오디오 추출
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
        if local_path.suffix.lower() in video_extensions:
            if not check_ffmpeg():
                raise RuntimeError(
                    "ffmpeg가 설치되지 않았습니다. "
                    "https://ffmpeg.org/download.html 에서 설치하세요."
                )

            logger.info(f"영상에서 오디오 추출 중: {audio_path}")
            temp_wav = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ).name
            temp_files.append(temp_wav)
            extract_audio(
                video_path=str(local_path),
                output_path=temp_wav,
                sample_rate=16000,
                channels=1,
            )
            local_path = Path(temp_wav)

        # GCS에 업로드
        if bucket_name is None:
            bucket_name = config.get("gcp", {}).get("bucket_name")
        if not bucket_name:
            raise ValueError(
                "로컬 파일을 처리하려면 --bucket 옵션 또는 config.yaml의 bucket_name이 필요합니다."
            )

        logger.info(f"오디오 파일 GCS 업로드 중...")
        gcs_uri = upload_file_to_gcs(
            local_file_path=str(local_path),
            bucket_name=bucket_name,
            destination_blob_name=f"lab02/{local_path.name}",
            project_id=project_id,
        )

        return gcs_uri, speech.RecognitionConfig.AudioEncoding.LINEAR16, 16000

    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                logger.debug(f"임시 파일 삭제: {temp_file}")


def transcribe_audio(
    gcs_uri: str,
    language_code: str,
    model: str = "default",
    encoding=None,
    sample_rate_hertz: int = 16000,
) -> List[dict]:
    """
    Speech-to-Text API로 음성을 텍스트로 변환합니다.

    Args:
        gcs_uri: 오디오 파일의 GCS URI
        language_code: 언어 코드 (예: ko-KR)
        model: 음성 인식 모델
        encoding: 오디오 인코딩. None이면 API가 자동 감지 (GCS MP4 등에 필요)
        sample_rate_hertz: 샘플 레이트. encoding=None 시 무시됨

    Returns:
        단어별 타임스탬프 포함 전사 결과 목록
    """
    from google.cloud import speech

    client = get_speech_client()

    # encoding이 None이면 API 자동 감지 (MP4 등 GCS 직접 사용 시)
    # encoding이 지정되면 WAV 업로드 후 명시적으로 설정
    if encoding is None:
        config = speech.RecognitionConfig(
            language_code=language_code,
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
            model=model,
        )
    else:
        config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=sample_rate_hertz,
            language_code=language_code,
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
            model=model,
        )

    audio = speech.RecognitionAudio(uri=gcs_uri)

    logger.info(f"음성 인식 시작: {gcs_uri} (언어: {language_code})")
    logger.info("비동기 API 요청 중... (긴 파일은 수 분 소요될 수 있습니다)")

    # long_running_recognize: 긴 오디오 파일을 비동기로 처리
    operation = client.long_running_recognize(config=config, audio=audio)

    # 완료 대기 (최대 10분)
    response = operation.result(timeout=600)
    logger.info("음성 인식 완료")

    # 단어별 타임스탬프 추출
    words_with_timestamps = []
    for result in response.results:
        alternative = result.alternatives[0]
        logger.debug(f"전사 결과: {alternative.transcript[:50]}...")

        for word_info in alternative.words:
            words_with_timestamps.append({
                "word": word_info.word,
                "start_time": word_info.start_time.total_seconds(),
                "end_time": word_info.end_time.total_seconds(),
                "confidence": alternative.confidence,
            })

    logger.info(f"총 {len(words_with_timestamps)}개 단어 인식")
    return words_with_timestamps


def words_to_subtitle_entries(
    words: List[dict],
    max_words_per_entry: int = 10,
    max_duration: float = 5.0,
) -> List[SubtitleEntry]:
    """
    단어 목록을 자막 항목으로 변환합니다.

    Args:
        words: 단어별 타임스탬프 목록
        max_words_per_entry: 자막 항목당 최대 단어 수
        max_duration: 자막 항목 최대 지속 시간 (초)

    Returns:
        자막 항목 목록
    """
    if not words:
        return []

    entries = []
    current_words = []
    entry_index = 1

    for word_info in words:
        current_words.append(word_info)

        # 자막 항목 분할 조건 확인
        should_split = False

        if len(current_words) >= max_words_per_entry:
            should_split = True
        elif len(current_words) > 1:
            duration = (
                current_words[-1]["end_time"] - current_words[0]["start_time"]
            )
            if duration >= max_duration:
                should_split = True

        if should_split:
            text = " ".join(w["word"] for w in current_words)
            entries.append(
                SubtitleEntry(
                    index=entry_index,
                    start_time=current_words[0]["start_time"],
                    end_time=current_words[-1]["end_time"],
                    text=text,
                )
            )
            entry_index += 1
            current_words = []

    # 마지막 남은 단어들 처리
    if current_words:
        text = " ".join(w["word"] for w in current_words)
        entries.append(
            SubtitleEntry(
                index=entry_index,
                start_time=current_words[0]["start_time"],
                end_time=current_words[-1]["end_time"],
                text=text,
            )
        )

    logger.info(f"자막 항목 생성: {len(entries)}개")
    return entries


def save_subtitles(
    entries: List[SubtitleEntry],
    output_path: str,
    output_format: str,
    language_code: str,
) -> List[str]:
    """
    자막 항목을 지정된 형식으로 저장합니다.

    Args:
        entries: 자막 항목 목록
        output_path: 출력 파일 경로 (확장자 제외)
        output_format: 출력 형식 ('srt', 'vtt', 'both')
        language_code: 언어 코드

    Returns:
        저장된 파일 경로 목록
    """
    saved_files = []
    base_path = Path(output_path)

    # 확장자가 이미 있는 경우 제거
    if base_path.suffix in {".srt", ".vtt"}:
        base_path = base_path.with_suffix("")

    if output_format in ("srt", "both"):
        srt_path = str(base_path) + ".srt"
        save_srt(entries, srt_path)
        saved_files.append(srt_path)
        logger.info(f"SRT 파일 저장: {srt_path}")

    if output_format in ("vtt", "both"):
        vtt_path = str(base_path) + ".vtt"
        lang = language_code.split("-")[0]  # ko-KR -> ko
        save_vtt(entries, vtt_path, language=lang)
        saved_files.append(vtt_path)
        logger.info(f"VTT 파일 저장: {vtt_path}")

    return saved_files


def print_subtitle_preview(entries: List[SubtitleEntry], max_entries: int = 10) -> None:
    """자막 미리보기를 출력합니다."""
    print("\n" + "=" * 60)
    print("생성된 자막 미리보기")
    print("=" * 60)

    for entry in entries[:max_entries]:
        print(f"\n[{entry.index}]")
        print(
            f"  시간: {entry.start_time:.1f}s ~ {entry.end_time:.1f}s "
            f"({entry.duration:.1f}초)"
        )
        print(f"  내용: {entry.text}")

    if len(entries) > max_entries:
        print(f"\n  ... 외 {len(entries) - max_entries}개 항목")

    print("\n" + "=" * 60)
    print(f"총 자막 항목 수: {len(entries)}개")
    if entries:
        total_duration = entries[-1].end_time - entries[0].start_time
        print(f"자막 전체 시간: {total_duration:.1f}초")
    print("=" * 60)


def main() -> int:
    """메인 실행 함수."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        config = load_config(args.config)
        project_id = args.project or get_project_id(config)

        logger.info(f"프로젝트 ID: {project_id}")
        logger.info(f"음성 파일: {args.audio}")
        logger.info(f"언어: {args.language}")

        # 오디오 준비 (필요 시 추출 및 GCS 업로드)
        gcs_uri, encoding, sample_rate = prepare_audio_for_api(
            audio_path=args.audio,
            project_id=project_id,
            bucket_name=args.bucket,
            config=config,
        )

        # 음성 인식 (encoding=None이면 API가 자동 감지 — GCS MP4에 필요)
        words = transcribe_audio(
            gcs_uri=gcs_uri,
            language_code=args.language,
            model=args.model,
            encoding=encoding,
            sample_rate_hertz=sample_rate,
        )

        if not words:
            logger.warning("음성이 인식되지 않았습니다. 오디오 파일과 언어 설정을 확인하세요.")
            return 1

        # 자막 항목 생성
        entries = words_to_subtitle_entries(
            words=words,
            max_duration=args.max_duration,
        )

        # 짧은 항목 병합
        entries = merge_short_entries(
            entries=entries,
            min_duration=args.min_duration,
        )

        # 자막 미리보기 출력
        print_subtitle_preview(entries)

        # 자막 파일 저장
        if args.output:
            saved_files = save_subtitles(
                entries=entries,
                output_path=args.output,
                output_format=args.format,
                language_code=args.language,
            )
            print(f"\n저장된 파일:")
            for f in saved_files:
                print(f"  {f}")
        else:
            # 현재 디렉토리에 기본 이름으로 저장
            default_output = f"outputs/subtitle_{args.language}"
            saved_files = save_subtitles(
                entries=entries,
                output_path=default_output,
                output_format=args.format,
                language_code=args.language,
            )
            print(f"\n저장된 파일:")
            for f in saved_files:
                print(f"  {f}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"파일 오류: {e}")
        return 1
    except ValueError as e:
        logger.error(f"설정 오류: {e}")
        return 1
    except Exception as e:
        logger.error(f"자막 생성 중 오류 발생: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
