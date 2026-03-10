#!/usr/bin/env python3
"""
Lab 3: 다국어 자막 & 번역 - Translation API

이 스크립트는 Cloud Translation API v3를 사용하여
SRT/VTT 자막 파일을 여러 언어로 자동 번역합니다.

사용법:
    python translate_subtitle.py --input subtitle.srt --project PROJECT_ID

DAN25 연관:
    네이버 동영상의 글로벌 서비스를 위한 다국어 자막 자동 생성 기술입니다.
    한국어 영상을 전 세계 사용자가 자국어로 즐길 수 있도록 지원합니다.
"""

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 상위 디렉토리의 src 모듈을 임포트하기 위한 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gcp_client import (
    get_project_id,
    get_translation_client,
    load_config,
    setup_logging,
)
from src.subtitle_utils import (
    SubtitleEntry,
    parse_srt,
    parse_vtt,
    save_srt,
    save_vtt,
)

logger = logging.getLogger(__name__)

# 지원 언어 및 이름
SUPPORTED_LANGUAGES = {
    "en": "영어",
    "ja": "일본어",
    "zh-CN": "중국어 간체",
    "zh-TW": "중국어 번체",
    "es": "스페인어",
    "fr": "프랑스어",
    "de": "독일어",
    "th": "태국어",
    "vi": "베트남어",
    "id": "인도네시아어",
    "pt": "포르투갈어",
    "ar": "아랍어",
    "ru": "러시아어",
}


def parse_args() -> argparse.Namespace:
    """CLI 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="Translation API를 사용한 다국어 자막 자동 번역",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 영어, 일본어, 중국어로 번역
  python translate_subtitle.py \\
      --input labs/lab02_speech_subtitle/sample.srt \\
      --project my-project-id \\
      --languages en ja zh-CN

  # 특정 언어로만 번역
  python translate_subtitle.py \\
      --input subtitle.srt \\
      --project my-project-id \\
      --languages en \\
      --output data/output/subtitle_en.srt

  # 출력 디렉토리 지정 (여러 언어)
  python translate_subtitle.py \\
      --input subtitle.srt \\
      --project my-project-id \\
      --languages en ja zh-CN \\
      --output-dir data/output/translated/
        """,
    )

    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="번역할 자막 파일 경로 (SRT 또는 VTT). 와일드카드 사용 가능",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="GCP 프로젝트 ID",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "ja", "zh-CN"],
        help=f"번역할 언어 코드 목록 (기본: en ja zh-CN). 선택 가능: {', '.join(SUPPORTED_LANGUAGES.keys())}",
    )
    parser.add_argument(
        "--source-language",
        default="ko",
        help="원본 언어 코드 (기본: ko 한국어)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="단일 출력 파일 경로 (한 언어만 번역할 때 사용)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="번역 파일 저장 디렉토리 (여러 언어/파일 번역 시 사용)",
    )
    parser.add_argument(
        "--format",
        choices=["srt", "vtt", "both"],
        default="srt",
        help="출력 자막 파일 형식 (기본: srt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="한 번에 번역할 자막 항목 수 (기본: 100). API 비용 최적화용",
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


def load_subtitle_file(file_path: str) -> List[SubtitleEntry]:
    """
    자막 파일을 로드합니다.

    Args:
        file_path: SRT 또는 VTT 파일 경로

    Returns:
        자막 항목 목록

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
        ValueError: 지원하지 않는 파일 형식인 경우
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"자막 파일을 찾을 수 없습니다: {file_path}")

    content = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".srt":
        entries = parse_srt(content)
    elif suffix == ".vtt":
        entries = parse_vtt(content)
    else:
        raise ValueError(
            f"지원하지 않는 파일 형식: {suffix}. SRT 또는 VTT 파일만 지원합니다."
        )

    logger.info(f"자막 파일 로드: {file_path} ({len(entries)}개 항목)")
    return entries


def translate_texts_batch(
    texts: List[str],
    target_language: str,
    source_language: str,
    project_id: str,
    batch_size: int = 100,
) -> List[str]:
    """
    텍스트 목록을 일괄 번역합니다.

    Args:
        texts: 번역할 텍스트 목록
        target_language: 목표 언어 코드 (예: en, ja, zh-CN)
        source_language: 원본 언어 코드 (예: ko)
        project_id: GCP 프로젝트 ID
        batch_size: 한 번에 처리할 항목 수

    Returns:
        번역된 텍스트 목록 (입력과 동일한 순서)
    """
    from google.cloud import translate_v3 as translate

    client = get_translation_client()
    parent = f"projects/{project_id}/locations/global"

    translated = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    logger.info(
        f"번역 시작: {source_language} -> {target_language}, "
        f"총 {len(texts)}개 항목 ({total_batches}개 배치)"
    )

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1

        logger.debug(f"배치 {batch_num}/{total_batches} 번역 중...")

        response = client.translate_text(
            request={
                "parent": parent,
                "contents": batch,
                "source_language_code": source_language,
                "target_language_code": target_language,
                "mime_type": "text/plain",
            }
        )

        for translation in response.translations:
            translated.append(translation.translated_text)

        logger.debug(f"배치 {batch_num}/{total_batches} 완료")

    logger.info(f"번역 완료: {len(translated)}개 항목")
    return translated


def translate_subtitle_entries(
    entries: List[SubtitleEntry],
    target_language: str,
    source_language: str,
    project_id: str,
    batch_size: int = 100,
) -> List[SubtitleEntry]:
    """
    자막 항목 목록을 번역합니다. 타임스탬프는 유지합니다.

    Args:
        entries: 번역할 자막 항목 목록
        target_language: 목표 언어 코드
        source_language: 원본 언어 코드
        project_id: GCP 프로젝트 ID
        batch_size: 배치 크기

    Returns:
        번역된 자막 항목 목록
    """
    if not entries:
        return []

    # 모든 텍스트를 한 번에 번역 (타임스탬프 제외)
    original_texts = [entry.text for entry in entries]
    translated_texts = translate_texts_batch(
        texts=original_texts,
        target_language=target_language,
        source_language=source_language,
        project_id=project_id,
        batch_size=batch_size,
    )

    # 번역된 텍스트로 새 자막 항목 생성 (타임스탬프 유지)
    translated_entries = []
    for entry, translated_text in zip(entries, translated_texts):
        new_entry = SubtitleEntry(
            index=entry.index,
            start_time=entry.start_time,
            end_time=entry.end_time,
            text=translated_text,
            translations={**entry.translations, target_language: translated_text},
        )
        translated_entries.append(new_entry)

    return translated_entries


def save_translated_subtitle(
    entries: List[SubtitleEntry],
    output_path: str,
    output_format: str,
    language_code: str,
) -> List[str]:
    """
    번역된 자막을 파일로 저장합니다.

    Args:
        entries: 번역된 자막 항목 목록
        output_path: 출력 파일 경로 (확장자 포함 또는 제외)
        output_format: 출력 형식 ('srt', 'vtt', 'both')
        language_code: 언어 코드

    Returns:
        저장된 파일 경로 목록
    """
    saved_files = []
    base_path = Path(output_path)

    if base_path.suffix in {".srt", ".vtt"}:
        base_path = base_path.with_suffix("")

    if output_format in ("srt", "both"):
        srt_path = str(base_path) + ".srt"
        save_srt(entries, srt_path)
        saved_files.append(srt_path)

    if output_format in ("vtt", "both"):
        vtt_path = str(base_path) + ".vtt"
        lang_short = language_code.split("-")[0]
        save_vtt(entries, vtt_path, language=lang_short)
        saved_files.append(vtt_path)

    return saved_files


def process_subtitle_file(
    input_path: str,
    target_languages: List[str],
    source_language: str,
    project_id: str,
    output_path: Optional[str],
    output_dir: Optional[str],
    output_format: str,
    batch_size: int,
) -> Dict[str, List[str]]:
    """
    단일 자막 파일을 여러 언어로 번역합니다.

    Args:
        input_path: 입력 자막 파일 경로
        target_languages: 번역할 언어 코드 목록
        source_language: 원본 언어 코드
        project_id: GCP 프로젝트 ID
        output_path: 단일 출력 파일 경로
        output_dir: 출력 디렉토리
        output_format: 출력 파일 형식
        batch_size: 배치 크기

    Returns:
        언어 코드 -> 저장된 파일 경로 목록 딕셔너리
    """
    # 자막 파일 로드
    entries = load_subtitle_file(input_path)

    input_path_obj = Path(input_path)
    results = {}

    for lang_code in target_languages:
        lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code)
        logger.info(f"{lang_name}({lang_code}) 번역 중...")

        # 번역 실행
        translated_entries = translate_subtitle_entries(
            entries=entries,
            target_language=lang_code,
            source_language=source_language,
            project_id=project_id,
            batch_size=batch_size,
        )

        # 출력 경로 결정
        if output_path and len(target_languages) == 1:
            # 단일 언어 단일 파일
            file_output_path = output_path
        elif output_dir:
            # 출력 디렉토리 지정
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            lang_safe = lang_code.replace("-", "_")
            file_output_path = str(
                out_dir / f"{input_path_obj.stem}_{lang_safe}"
            )
        else:
            # 입력 파일과 같은 디렉토리
            lang_safe = lang_code.replace("-", "_")
            file_output_path = str(
                input_path_obj.parent
                / f"{input_path_obj.stem}_{lang_safe}"
            )

        # 저장
        saved_files = save_translated_subtitle(
            entries=translated_entries,
            output_path=file_output_path,
            output_format=output_format,
            language_code=lang_code,
        )

        results[lang_code] = saved_files
        logger.info(
            f"{lang_name} 번역 완료: {', '.join(saved_files)}"
        )

    return results


def print_translation_summary(
    input_files: List[str],
    results: Dict[str, Dict[str, List[str]]],
) -> None:
    """번역 결과 요약을 출력합니다."""
    print("\n" + "=" * 60)
    print("다국어 자막 번역 결과")
    print("=" * 60)

    total_files = 0
    for input_file, lang_results in results.items():
        print(f"\n원본 파일: {input_file}")
        for lang_code, saved_files in lang_results.items():
            lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code)
            print(f"  [{lang_name} ({lang_code})]")
            for f in saved_files:
                print(f"    저장됨: {f}")
                total_files += 1

    print(f"\n총 {total_files}개 번역 파일 생성 완료")
    print("=" * 60)


def main() -> int:
    """메인 실행 함수."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        config = load_config(args.config)
        project_id = args.project or get_project_id(config)

        logger.info(f"프로젝트 ID: {project_id}")
        logger.info(f"번역 언어: {', '.join(args.languages)}")

        # 입력 파일 목록 확장 (와일드카드 처리)
        input_files = []
        for pattern in args.input:
            matched = glob.glob(pattern)
            if matched:
                input_files.extend(matched)
            else:
                input_files.append(pattern)

        if not input_files:
            logger.error("처리할 자막 파일을 찾을 수 없습니다.")
            return 1

        logger.info(f"처리할 파일: {len(input_files)}개")

        # 언어 코드 유효성 검사
        unknown_langs = [
            lang for lang in args.languages if lang not in SUPPORTED_LANGUAGES
        ]
        if unknown_langs:
            logger.warning(
                f"알 수 없는 언어 코드: {', '.join(unknown_langs)}. "
                f"번역이 실패할 수 있습니다."
            )

        # 각 파일 번역
        all_results = {}
        for input_file in input_files:
            logger.info(f"파일 처리 중: {input_file}")
            try:
                file_results = process_subtitle_file(
                    input_path=input_file,
                    target_languages=args.languages,
                    source_language=args.source_language,
                    project_id=project_id,
                    output_path=args.output,
                    output_dir=args.output_dir,
                    output_format=args.format,
                    batch_size=args.batch_size,
                )
                all_results[input_file] = file_results
            except Exception as e:
                logger.error(f"파일 처리 실패 ({input_file}): {e}")
                all_results[input_file] = {}

        # 결과 출력
        print_translation_summary(input_files, all_results)
        return 0

    except FileNotFoundError as e:
        logger.error(f"파일 오류: {e}")
        return 1
    except ValueError as e:
        logger.error(f"설정 오류: {e}")
        return 1
    except Exception as e:
        logger.error(f"번역 중 오류 발생: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
