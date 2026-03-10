"""
영상 처리 유틸리티 모듈

영상 파일의 로드, 변환, 클립 추출 등 공통 영상 처리 기능을 제공합니다.
ffmpeg 및 moviepy를 활용합니다.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
    """
    ffmpeg가 시스템에 설치되어 있는지 확인합니다.

    Returns:
        ffmpeg가 설치되어 있으면 True, 아니면 False
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_video_info(video_path: str) -> dict:
    """
    영상 파일의 메타데이터를 반환합니다.

    Args:
        video_path: 영상 파일 경로

    Returns:
        영상 메타데이터 딕셔너리 (duration, width, height, fps, codec 등)

    Raises:
        FileNotFoundError: 영상 파일이 존재하지 않는 경우
        RuntimeError: ffprobe 실행 실패 시
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe 실행 실패: {result.stderr}")

        import json
        data = json.loads(result.stdout)

        info = {
            "path": str(path),
            "filename": path.name,
            "size_bytes": path.stat().st_size,
        }

        # 포맷 정보
        fmt = data.get("format", {})
        info["duration"] = float(fmt.get("duration", 0))
        info["bit_rate"] = int(fmt.get("bit_rate", 0))
        info["format_name"] = fmt.get("format_name", "")

        # 스트림 정보
        for stream in data.get("streams", []):
            codec_type = stream.get("codec_type", "")
            if codec_type == "video":
                info["video_codec"] = stream.get("codec_name", "")
                info["width"] = stream.get("width", 0)
                info["height"] = stream.get("height", 0)
                # FPS 계산
                fps_str = stream.get("r_frame_rate", "0/1")
                num, den = map(int, fps_str.split("/"))
                info["fps"] = round(num / den, 2) if den > 0 else 0
            elif codec_type == "audio":
                info["audio_codec"] = stream.get("codec_name", "")
                info["audio_channels"] = stream.get("channels", 0)
                info["audio_sample_rate"] = int(stream.get("sample_rate", 0))

        logger.info(
            f"영상 정보: {path.name} "
            f"(길이: {info.get('duration', 0):.1f}초, "
            f"해상도: {info.get('width', 0)}x{info.get('height', 0)})"
        )
        return info

    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe 출력 파싱 실패: {e}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffprobe 실행 시간 초과")


def extract_audio(
    video_path: str,
    output_path: Optional[str] = None,
    sample_rate: int = 16000,
    channels: int = 1,
) -> str:
    """
    영상에서 오디오를 추출합니다.

    Args:
        video_path: 입력 영상 파일 경로
        output_path: 출력 오디오 파일 경로. None이면 임시 파일 생성
        sample_rate: 출력 샘플링 레이트 (Hz)
        channels: 출력 채널 수 (1: 모노, 2: 스테레오)

    Returns:
        추출된 오디오 파일 경로

    Raises:
        FileNotFoundError: 영상 파일이 존재하지 않는 경우
        RuntimeError: ffmpeg 실행 실패 시
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    if output_path is None:
        suffix = ".wav"
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        output_path = temp_file.name
        temp_file.close()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(path),
        "-vn",                          # 비디오 스트림 제외
        "-acodec", "pcm_s16le",         # PCM 16bit 인코딩
        "-ar", str(sample_rate),        # 샘플링 레이트
        "-ac", str(channels),           # 채널 수
        "-y",                           # 덮어쓰기 허용
        str(output),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"오디오 추출 실패: {result.stderr}")

        logger.info(f"오디오 추출 완료: {video_path} -> {output_path}")
        return str(output)

    except subprocess.TimeoutExpired:
        raise RuntimeError("오디오 추출 시간 초과 (5분 이상)")


def extract_clip(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: Optional[str] = None,
) -> str:
    """
    영상에서 특정 구간의 클립을 추출합니다.

    Args:
        video_path: 입력 영상 파일 경로
        start_time: 시작 시간 (초)
        end_time: 종료 시간 (초)
        output_path: 출력 클립 파일 경로. None이면 자동 생성

    Returns:
        추출된 클립 파일 경로

    Raises:
        FileNotFoundError: 영상 파일이 존재하지 않는 경우
        ValueError: 시작/종료 시간이 유효하지 않은 경우
        RuntimeError: ffmpeg 실행 실패 시
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    if start_time < 0 or end_time <= start_time:
        raise ValueError(
            f"유효하지 않은 시간 범위: start={start_time}, end={end_time}"
        )

    duration = end_time - start_time

    if output_path is None:
        stem = path.stem
        output_path = str(
            path.parent / f"{stem}_clip_{int(start_time)}s_{int(end_time)}s{path.suffix}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-ss", str(start_time),     # 시작 시간 (입력 앞에 배치하면 빠름)
        "-i", str(path),
        "-t", str(duration),        # 지속 시간
        "-c", "copy",               # 재인코딩 없이 복사 (빠름)
        "-y",
        str(output),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"클립 추출 실패: {result.stderr}")

        logger.info(
            f"클립 추출 완료: {start_time:.1f}s ~ {end_time:.1f}s -> {output_path}"
        )
        return str(output)

    except subprocess.TimeoutExpired:
        raise RuntimeError("클립 추출 시간 초과")


def extract_frame(
    video_path: str,
    timestamp: float,
    output_path: Optional[str] = None,
) -> str:
    """
    영상에서 특정 시간의 프레임을 이미지로 추출합니다.

    Args:
        video_path: 입력 영상 파일 경로
        timestamp: 프레임 추출 시간 (초)
        output_path: 출력 이미지 파일 경로. None이면 자동 생성

    Returns:
        추출된 이미지 파일 경로

    Raises:
        FileNotFoundError: 영상 파일이 존재하지 않는 경우
        RuntimeError: ffmpeg 실행 실패 시
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    if output_path is None:
        stem = path.stem
        output_path = str(path.parent / f"{stem}_frame_{int(timestamp)}s.jpg")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", str(path),
        "-frames:v", "1",           # 1프레임만 추출
        "-q:v", "2",                # 이미지 품질 (낮을수록 고품질)
        "-y",
        str(output),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"프레임 추출 실패: {result.stderr}")

        logger.info(f"프레임 추출 완료: {timestamp:.1f}s -> {output_path}")
        return str(output)

    except subprocess.TimeoutExpired:
        raise RuntimeError("프레임 추출 시간 초과")


def merge_clips(
    clip_paths: List[str],
    output_path: str,
    add_transitions: bool = False,
) -> str:
    """
    여러 영상 클립을 하나로 합칩니다.

    Args:
        clip_paths: 합칠 클립 파일 경로 목록
        output_path: 출력 영상 파일 경로
        add_transitions: 클립 사이에 전환 효과 추가 여부

    Returns:
        합쳐진 영상 파일 경로

    Raises:
        ValueError: 클립 목록이 비어 있는 경우
        FileNotFoundError: 클립 파일이 존재하지 않는 경우
        RuntimeError: ffmpeg 실행 실패 시
    """
    if not clip_paths:
        raise ValueError("합칠 클립 목록이 비어 있습니다.")

    for clip_path in clip_paths:
        if not Path(clip_path).exists():
            raise FileNotFoundError(f"클립 파일을 찾을 수 없습니다: {clip_path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg concat demuxer를 위한 파일 목록 생성
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for clip_path in clip_paths:
            f.write(f"file '{os.path.abspath(clip_path)}'\n")
        list_file = f.name

    try:
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            "-y",
            str(output),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"클립 합치기 실패: {result.stderr}")

        logger.info(
            f"클립 합치기 완료: {len(clip_paths)}개 클립 -> {output_path}"
        )
        return str(output)

    except subprocess.TimeoutExpired:
        raise RuntimeError("클립 합치기 시간 초과")
    finally:
        os.unlink(list_file)


def convert_video(
    input_path: str,
    output_path: str,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    resolution: Optional[Tuple[int, int]] = None,
    crf: int = 23,
) -> str:
    """
    영상 파일을 변환합니다.

    Args:
        input_path: 입력 영상 파일 경로
        output_path: 출력 영상 파일 경로
        video_codec: 비디오 코덱 (기본: libx264)
        audio_codec: 오디오 코덱 (기본: aac)
        resolution: 출력 해상도 (width, height). None이면 원본 유지
        crf: 비디오 품질 (0-51, 낮을수록 고품질, 기본: 23)

    Returns:
        변환된 영상 파일 경로

    Raises:
        FileNotFoundError: 입력 파일이 존재하지 않는 경우
        RuntimeError: ffmpeg 실행 실패 시
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(path),
        "-c:v", video_codec,
        "-crf", str(crf),
        "-c:a", audio_codec,
    ]

    if resolution is not None:
        width, height = resolution
        cmd.extend(["-vf", f"scale={width}:{height}"])

    cmd.extend(["-y", str(output)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"영상 변환 실패: {result.stderr}")

        logger.info(f"영상 변환 완료: {input_path} -> {output_path}")
        return str(output)

    except subprocess.TimeoutExpired:
        raise RuntimeError("영상 변환 시간 초과")


def seconds_to_timestamp(seconds: float) -> str:
    """
    초 단위 시간을 HH:MM:SS.mmm 형식으로 변환합니다.

    Args:
        seconds: 초 단위 시간

    Returns:
        HH:MM:SS.mmm 형식 타임스탬프 문자열
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def timestamp_to_seconds(timestamp: str) -> float:
    """
    HH:MM:SS.mmm 또는 MM:SS.mmm 형식의 타임스탬프를 초로 변환합니다.

    Args:
        timestamp: 타임스탬프 문자열

    Returns:
        초 단위 시간

    Raises:
        ValueError: 올바르지 않은 타임스탬프 형식
    """
    parts = timestamp.replace(",", ".").split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        else:
            return float(parts[0])
    except ValueError as e:
        raise ValueError(f"올바르지 않은 타임스탬프 형식: {timestamp}") from e
