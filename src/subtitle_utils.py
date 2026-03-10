"""
자막 유틸리티 모듈

SRT, VTT 등 자막 파일 형식의 파싱, 생성, 변환 기능을 제공합니다.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SubtitleEntry:
    """
    단일 자막 항목을 나타내는 데이터 클래스입니다.

    Attributes:
        index: 자막 번호 (1부터 시작)
        start_time: 시작 시간 (초)
        end_time: 종료 시간 (초)
        text: 자막 텍스트
        translations: 언어 코드별 번역 딕셔너리
    """

    index: int
    start_time: float
    end_time: float
    text: str
    translations: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """자막 표시 시간 (초)"""
        return self.end_time - self.start_time


def seconds_to_srt_time(seconds: float) -> str:
    """
    초 단위 시간을 SRT 형식(HH:MM:SS,mmm)으로 변환합니다.

    Args:
        seconds: 초 단위 시간

    Returns:
        SRT 형식 타임스탬프 문자열 (예: 00:01:23,456)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def seconds_to_vtt_time(seconds: float) -> str:
    """
    초 단위 시간을 VTT 형식(HH:MM:SS.mmm)으로 변환합니다.

    Args:
        seconds: 초 단위 시간

    Returns:
        VTT 형식 타임스탬프 문자열 (예: 00:01:23.456)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def srt_time_to_seconds(srt_time: str) -> float:
    """
    SRT 형식(HH:MM:SS,mmm)의 타임스탬프를 초로 변환합니다.

    Args:
        srt_time: SRT 형식 타임스탬프 (예: 00:01:23,456)

    Returns:
        초 단위 시간

    Raises:
        ValueError: 올바르지 않은 SRT 타임스탬프 형식
    """
    pattern = r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    match = re.match(pattern, srt_time.strip())
    if not match:
        raise ValueError(f"올바르지 않은 SRT 타임스탬프: {srt_time}")

    hours, minutes, secs, millis = match.groups()
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(secs)
        + int(millis) / 1000
    )


def entries_to_srt(entries: List[SubtitleEntry]) -> str:
    """
    자막 항목 목록을 SRT 형식 문자열로 변환합니다.

    Args:
        entries: 자막 항목 목록

    Returns:
        SRT 형식 자막 문자열
    """
    lines = []
    for entry in entries:
        lines.append(str(entry.index))
        lines.append(
            f"{seconds_to_srt_time(entry.start_time)} --> "
            f"{seconds_to_srt_time(entry.end_time)}"
        )
        lines.append(entry.text)
        lines.append("")  # 빈 줄로 항목 구분

    return "\n".join(lines)


def entries_to_vtt(entries: List[SubtitleEntry], language: Optional[str] = None) -> str:
    """
    자막 항목 목록을 VTT 형식 문자열로 변환합니다.

    Args:
        entries: 자막 항목 목록
        language: 자막 언어 코드 (예: ko, en)

    Returns:
        VTT 형식 자막 문자열
    """
    lines = ["WEBVTT"]
    if language:
        lines[0] += f"\nLanguage: {language}"
    lines.append("")

    for entry in entries:
        lines.append(f"{entry.index}")
        lines.append(
            f"{seconds_to_vtt_time(entry.start_time)} --> "
            f"{seconds_to_vtt_time(entry.end_time)}"
        )
        lines.append(entry.text)
        lines.append("")

    return "\n".join(lines)


def parse_srt(srt_content: str) -> List[SubtitleEntry]:
    """
    SRT 형식 자막 문자열을 파싱하여 자막 항목 목록을 반환합니다.

    Args:
        srt_content: SRT 형식 자막 문자열

    Returns:
        자막 항목 목록

    Raises:
        ValueError: SRT 파싱 실패 시
    """
    entries = []
    blocks = re.split(r"\n\s*\n", srt_content.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0].strip())
        except ValueError:
            logger.warning(f"자막 인덱스 파싱 실패: {lines[0]}")
            continue

        # 타임코드 파싱
        time_pattern = r"(.+?)\s+-->\s+(.+)"
        time_match = re.match(time_pattern, lines[1].strip())
        if not time_match:
            logger.warning(f"타임코드 파싱 실패: {lines[1]}")
            continue

        try:
            start_time = srt_time_to_seconds(time_match.group(1))
            end_time = srt_time_to_seconds(time_match.group(2))
        except ValueError as e:
            logger.warning(f"시간 변환 실패: {e}")
            continue

        # 나머지 줄이 자막 텍스트
        text = "\n".join(lines[2:]).strip()

        entries.append(
            SubtitleEntry(
                index=index,
                start_time=start_time,
                end_time=end_time,
                text=text,
            )
        )

    logger.info(f"SRT 파싱 완료: {len(entries)}개 자막 항목")
    return entries


def parse_vtt(vtt_content: str) -> List[SubtitleEntry]:
    """
    VTT 형식 자막 문자열을 파싱하여 자막 항목 목록을 반환합니다.

    Args:
        vtt_content: VTT 형식 자막 문자열

    Returns:
        자막 항목 목록
    """
    # WEBVTT 헤더 제거
    content = re.sub(r"^WEBVTT.*?\n\n", "", vtt_content, flags=re.DOTALL).strip()

    entries = []
    blocks = re.split(r"\n\s*\n", content)
    index_counter = 1

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        # 인덱스가 있는 경우와 없는 경우 처리
        time_line_idx = 0
        current_index = index_counter

        if lines[0].strip().isdigit():
            current_index = int(lines[0].strip())
            time_line_idx = 1

        if time_line_idx >= len(lines):
            continue

        # 타임코드 파싱
        time_pattern = r"(.+?)\s+-->\s+(.+)"
        time_match = re.match(time_pattern, lines[time_line_idx].strip())
        if not time_match:
            continue

        try:
            # VTT도 srt_time_to_seconds로 파싱 가능 (점/쉼표 모두 처리)
            start_time = srt_time_to_seconds(time_match.group(1))
            end_time = srt_time_to_seconds(time_match.group(2))
        except ValueError:
            continue

        text = "\n".join(lines[time_line_idx + 1:]).strip()
        if not text:
            continue

        entries.append(
            SubtitleEntry(
                index=current_index,
                start_time=start_time,
                end_time=end_time,
                text=text,
            )
        )
        index_counter += 1

    logger.info(f"VTT 파싱 완료: {len(entries)}개 자막 항목")
    return entries


def save_srt(entries: List[SubtitleEntry], output_path: str) -> str:
    """
    자막 항목을 SRT 파일로 저장합니다.

    Args:
        entries: 자막 항목 목록
        output_path: 출력 파일 경로

    Returns:
        저장된 파일 경로
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    srt_content = entries_to_srt(entries)
    path.write_text(srt_content, encoding="utf-8")

    logger.info(f"SRT 파일 저장 완료: {output_path} ({len(entries)}개 항목)")
    return str(path)


def save_vtt(
    entries: List[SubtitleEntry],
    output_path: str,
    language: Optional[str] = None,
) -> str:
    """
    자막 항목을 VTT 파일로 저장합니다.

    Args:
        entries: 자막 항목 목록
        output_path: 출력 파일 경로
        language: 자막 언어 코드

    Returns:
        저장된 파일 경로
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vtt_content = entries_to_vtt(entries, language=language)
    path.write_text(vtt_content, encoding="utf-8")

    logger.info(f"VTT 파일 저장 완료: {output_path} ({len(entries)}개 항목)")
    return str(path)


def merge_short_entries(
    entries: List[SubtitleEntry],
    min_duration: float = 1.0,
    max_gap: float = 0.5,
) -> List[SubtitleEntry]:
    """
    너무 짧은 자막 항목들을 인접 항목과 합칩니다.

    Args:
        entries: 자막 항목 목록
        min_duration: 최소 자막 표시 시간 (초)
        max_gap: 합칠 항목 간 최대 간격 (초)

    Returns:
        처리된 자막 항목 목록
    """
    if not entries:
        return []

    merged = []
    current = entries[0]

    for next_entry in entries[1:]:
        gap = next_entry.start_time - current.end_time
        current_duration = current.end_time - current.start_time

        if current_duration < min_duration and gap <= max_gap:
            # 항목 합치기
            current = SubtitleEntry(
                index=current.index,
                start_time=current.start_time,
                end_time=next_entry.end_time,
                text=current.text + " " + next_entry.text,
            )
        else:
            merged.append(current)
            current = next_entry

    merged.append(current)

    # 인덱스 재번호 매기기
    for i, entry in enumerate(merged, 1):
        entry.index = i

    logger.info(
        f"자막 항목 병합 완료: {len(entries)}개 -> {len(merged)}개"
    )
    return merged


def split_long_entries(
    entries: List[SubtitleEntry],
    max_chars_per_line: int = 40,
    max_lines: int = 2,
) -> List[SubtitleEntry]:
    """
    너무 긴 자막 텍스트를 여러 줄로 분할합니다.

    Args:
        entries: 자막 항목 목록
        max_chars_per_line: 줄당 최대 문자 수
        max_lines: 최대 줄 수

    Returns:
        처리된 자막 항목 목록
    """
    result = []

    for entry in entries:
        text = entry.text
        if len(text) <= max_chars_per_line:
            result.append(entry)
            continue

        # 긴 텍스트를 여러 줄로 분할
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars_per_line:
                current_line = (current_line + " " + word).strip()
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        # 최대 줄 수 제한
        lines = lines[:max_lines]
        new_text = "\n".join(lines)

        result.append(
            SubtitleEntry(
                index=entry.index,
                start_time=entry.start_time,
                end_time=entry.end_time,
                text=new_text,
                translations=entry.translations,
            )
        )

    return result


def calculate_subtitle_stats(entries: List[SubtitleEntry]) -> dict:
    """
    자막 통계 정보를 계산합니다.

    Args:
        entries: 자막 항목 목록

    Returns:
        통계 딕셔너리
    """
    if not entries:
        return {}

    total_duration = sum(e.duration for e in entries)
    all_text = " ".join(e.text for e in entries)
    char_count = len(all_text.replace(" ", ""))
    word_count = len(all_text.split())

    return {
        "total_entries": len(entries),
        "total_subtitle_duration": round(total_duration, 2),
        "average_entry_duration": round(total_duration / len(entries), 2),
        "total_characters": char_count,
        "total_words": word_count,
        "average_chars_per_entry": round(char_count / len(entries), 1),
        "start_time": entries[0].start_time,
        "end_time": entries[-1].end_time,
        "available_languages": list(
            set(lang for e in entries for lang in e.translations.keys())
        ),
    }
