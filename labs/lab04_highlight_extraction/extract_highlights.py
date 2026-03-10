#!/usr/bin/env python3
"""
Lab 4: 하이라이트 자동 추출 - Video Intelligence API + Vertex AI

이 스크립트는 Video Intelligence API로 영상을 분석하고
장면별 중요도 점수를 계산하여 하이라이트 클립을 자동으로 추출합니다.

사용법:
    python extract_highlights.py --video /path/to/video.mp4 --project PROJECT_ID

DAN25 연관:
    네이버 동영상의 숏클립 자동 생성, 영상 미리보기, 하이라이트 요약 기능의
    핵심 기술입니다. AI가 영상에서 가장 흥미로운 장면을 자동으로 찾아냅니다.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# 상위 디렉토리의 src 모듈을 임포트하기 위한 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gcp_client import (
    get_project_id,
    get_video_intelligence_client,
    load_config,
    setup_logging,
    upload_file_to_gcs,
)
from src.video_processor import (
    check_ffmpeg,
    extract_clip,
    get_video_info,
    merge_clips,
)

logger = logging.getLogger(__name__)


@dataclass
class ShotInfo:
    """영상 장면(Shot) 정보를 나타내는 데이터 클래스."""

    shot_index: int
    start_time: float
    end_time: float
    labels: List[dict] = field(default_factory=list)
    objects: List[dict] = field(default_factory=list)
    has_text: bool = False
    score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class HighlightResult:
    """하이라이트 추출 결과를 나타내는 데이터 클래스."""

    rank: int
    shot: ShotInfo
    clip_path: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.shot.end_time - self.shot.start_time

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "shot_index": self.shot.shot_index,
            "start_time": round(self.shot.start_time, 3),
            "end_time": round(self.shot.end_time, 3),
            "duration": round(self.duration, 3),
            "score": round(self.shot.score, 4),
            "score_breakdown": {
                k: round(v, 4) for k, v in self.shot.score_breakdown.items()
            },
            "top_labels": [
                label["description"]
                for label in sorted(
                    self.shot.labels,
                    key=lambda x: x.get("confidence", 0),
                    reverse=True,
                )[:5]
            ],
            "object_count": len(self.shot.objects),
            "has_text": self.shot.has_text,
            "clip_path": self.clip_path,
        }


def parse_args() -> argparse.Namespace:
    """CLI 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="Video Intelligence API를 사용한 하이라이트 자동 추출",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 하이라이트 추출 (상위 5개)
  python extract_highlights.py \\
      --video gs://your-bucket/video.mp4 \\
      --project my-project-id \\
      --output-dir data/output/highlights/

  # 로컬 파일에서 15초 클립 추출
  python extract_highlights.py \\
      --video /path/to/video.mp4 \\
      --project my-project-id \\
      --bucket my-bucket \\
      --max-highlights 5 \\
      --clip-duration 15 \\
      --output-dir data/output/highlights/

  # 클립을 하나의 영상으로 합치기
  python extract_highlights.py \\
      --video /path/to/video.mp4 \\
      --project my-project-id \\
      --bucket my-bucket \\
      --merge-clips \\
      --output-dir data/output/highlights/
        """,
    )

    parser.add_argument(
        "--video",
        required=True,
        help="분석할 영상 파일 경로 또는 GCS URI",
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
        "--output-dir",
        default="data/output/highlights",
        help="하이라이트 클립 저장 디렉토리 (기본: data/output/highlights)",
    )
    parser.add_argument(
        "--max-highlights",
        type=int,
        default=5,
        help="추출할 최대 하이라이트 수 (기본: 5)",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=None,
        help="각 클립의 고정 길이 (초). 지정 안 하면 원본 장면 길이 사용",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="하이라이트 선택 최소 점수 (0.0~1.0, 기본: 0.0 = 상위 N개 선택)",
    )
    parser.add_argument(
        "--merge-clips",
        action="store_true",
        help="추출된 클립들을 하나의 영상으로 합치기",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="분석만 수행, 클립 추출 안 함 (결과 JSON만 생성)",
    )
    parser.add_argument(
        "--label-weight",
        type=float,
        default=0.4,
        help="레이블 점수 가중치 (기본: 0.4)",
    )
    parser.add_argument(
        "--object-weight",
        type=float,
        default=0.3,
        help="객체 점수 가중치 (기본: 0.3)",
    )
    parser.add_argument(
        "--text-weight",
        type=float,
        default=0.1,
        help="텍스트 점수 가중치 (기본: 0.1)",
    )
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.2,
        help="다양성 점수 가중치 (기본: 0.2)",
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


def prepare_video_uri(
    video_path: str,
    project_id: str,
    bucket_name: Optional[str],
    config: dict,
) -> str:
    """영상 URI를 준비합니다."""
    if video_path.startswith("gs://"):
        return video_path

    local_path = Path(video_path)
    if not local_path.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    if bucket_name is None:
        bucket_name = config.get("gcp", {}).get("bucket_name")
    if not bucket_name:
        raise ValueError("로컬 파일 사용 시 --bucket 옵션이 필요합니다.")

    gcs_uri = upload_file_to_gcs(
        local_file_path=video_path,
        bucket_name=bucket_name,
        destination_blob_name=f"lab04/{local_path.name}",
        project_id=project_id,
    )
    return gcs_uri


def analyze_video_for_highlights(video_uri: str) -> dict:
    """
    Video Intelligence API로 하이라이트 추출을 위한 영상 분석을 수행합니다.

    Args:
        video_uri: 분석할 영상의 GCS URI

    Returns:
        분석 결과 (annotation_result 객체)
    """
    from google.cloud import videointelligence

    client = get_video_intelligence_client()

    features = [
        videointelligence.Feature.SHOT_CHANGE_DETECTION,
        videointelligence.Feature.LABEL_DETECTION,
        videointelligence.Feature.OBJECT_TRACKING,
        videointelligence.Feature.TEXT_DETECTION,
    ]

    logger.info(f"하이라이트 추출을 위한 영상 분석 시작: {video_uri}")

    operation = client.annotate_video(
        request={
            "input_uri": video_uri,
            "features": features,
        }
    )

    logger.info("분석 진행 중...")
    result = operation.result(timeout=600)
    logger.info("분석 완료")

    return result.annotation_results[0]


def build_shot_info_list(annotation_result) -> List[ShotInfo]:
    """
    분석 결과에서 장면별 정보를 구성합니다.

    Args:
        annotation_result: Video Intelligence API 분석 결과

    Returns:
        장면 정보 목록
    """
    shots = []

    # 장면 목록 생성
    for i, shot in enumerate(annotation_result.shot_annotations):
        shot_start = shot.start_time_offset.total_seconds()
        shot_end = shot.end_time_offset.total_seconds()

        shots.append(
            ShotInfo(
                shot_index=i,
                start_time=shot_start,
                end_time=shot_end,
            )
        )

    # 장면별 레이블 매핑
    for label in annotation_result.shot_label_annotations:
        for segment in label.segments:
            seg_start = segment.segment.start_time_offset.total_seconds()
            seg_end = segment.segment.end_time_offset.total_seconds()

            for shot in shots:
                # 시간 범위가 겹치는 장면에 레이블 추가
                if seg_start < shot.end_time and seg_end > shot.start_time:
                    shot.labels.append({
                        "description": label.entity.description,
                        "confidence": segment.confidence,
                    })

    # 장면별 객체 매핑
    for obj in annotation_result.object_annotations:
        obj_start = obj.segment.start_time_offset.total_seconds()
        obj_end = obj.segment.end_time_offset.total_seconds()

        for shot in shots:
            if obj_start < shot.end_time and obj_end > shot.start_time:
                shot.objects.append({
                    "description": obj.entity.description,
                    "confidence": obj.confidence,
                })

    # 텍스트 감지 매핑
    for text_ann in annotation_result.text_annotations:
        for segment in text_ann.segments:
            seg_start = segment.segment.start_time_offset.total_seconds()
            seg_end = segment.segment.end_time_offset.total_seconds()

            for shot in shots:
                if seg_start < shot.end_time and seg_end > shot.start_time:
                    shot.has_text = True

    logger.info(f"장면 정보 구성 완료: {len(shots)}개 장면")
    return shots


def calculate_shot_scores(
    shots: List[ShotInfo],
    label_weight: float = 0.4,
    object_weight: float = 0.3,
    text_weight: float = 0.1,
    diversity_weight: float = 0.2,
) -> List[ShotInfo]:
    """
    각 장면의 중요도 점수를 계산합니다.

    Args:
        shots: 장면 정보 목록
        label_weight: 레이블 점수 가중치
        object_weight: 객체 점수 가중치
        text_weight: 텍스트 점수 가중치
        diversity_weight: 다양성 점수 가중치

    Returns:
        점수가 계산된 장면 정보 목록
    """
    if not shots:
        return []

    total_duration = shots[-1].end_time

    for i, shot in enumerate(shots):
        # 1. 레이블 점수: 레이블 신뢰도 평균
        if shot.labels:
            label_score = sum(l["confidence"] for l in shot.labels) / len(shot.labels)
        else:
            label_score = 0.0

        # 2. 객체 점수: 객체 수와 신뢰도
        if shot.objects:
            obj_count_normalized = min(len(shot.objects) / 5.0, 1.0)
            obj_confidence = sum(o["confidence"] for o in shot.objects) / len(shot.objects)
            object_score = (obj_count_normalized + obj_confidence) / 2.0
        else:
            object_score = 0.0

        # 3. 텍스트 점수: 텍스트 포함 여부
        text_score = 1.0 if shot.has_text else 0.0

        # 4. 다양성 점수: 영상 전체에서의 시간적 위치
        # 시작, 중간, 끝 부분보다 중간 부분을 선호
        shot_mid = (shot.start_time + shot.end_time) / 2
        if total_duration > 0:
            position = shot_mid / total_duration
            diversity_score = 1.0 - abs(position - 0.5) * 0.5
        else:
            diversity_score = 0.5

        # 최종 점수 계산
        final_score = (
            label_score * label_weight
            + object_score * object_weight
            + text_score * text_weight
            + diversity_score * diversity_weight
        )

        shot.score = final_score
        shot.score_breakdown = {
            "label_score": label_score * label_weight,
            "object_score": object_score * object_weight,
            "text_score": text_score * text_weight,
            "diversity_score": diversity_score * diversity_weight,
        }

    logger.info("장면 점수 계산 완료")
    return shots


def select_highlights(
    shots: List[ShotInfo],
    max_highlights: int = 5,
    score_threshold: float = 0.0,
    min_gap_seconds: float = 30.0,
) -> List[ShotInfo]:
    """
    점수 기반으로 하이라이트 장면을 선택합니다.
    너무 가까운 장면이 중복 선택되지 않도록 합니다.

    Args:
        shots: 점수가 계산된 장면 목록
        max_highlights: 최대 선택 수
        score_threshold: 최소 점수 임계값
        min_gap_seconds: 선택된 장면 간 최소 간격 (초)

    Returns:
        선택된 하이라이트 장면 목록 (점수 내림차순)
    """
    # 점수 기준 내림차순 정렬
    sorted_shots = sorted(shots, key=lambda x: x.score, reverse=True)

    selected = []
    selected_times = []

    for shot in sorted_shots:
        if len(selected) >= max_highlights:
            break

        if shot.score < score_threshold:
            continue

        # 이미 선택된 장면과 겹치지 않는지 확인
        is_too_close = any(
            abs(shot.start_time - prev_time) < min_gap_seconds
            for prev_time in selected_times
        )

        if not is_too_close:
            selected.append(shot)
            selected_times.append(shot.start_time)

    # 시간순으로 정렬
    selected.sort(key=lambda x: x.start_time)

    logger.info(f"하이라이트 선택 완료: {len(selected)}개 장면")
    return selected


def extract_highlight_clips(
    video_path: str,
    highlights: List[HighlightResult],
    output_dir: str,
    clip_duration: Optional[float] = None,
) -> List[HighlightResult]:
    """
    선택된 하이라이트 장면을 클립으로 추출합니다.

    Args:
        video_path: 원본 영상 파일 경로 (로컬)
        highlights: 하이라이트 결과 목록
        output_dir: 클립 저장 디렉토리
        clip_duration: 고정 클립 길이 (초). None이면 원본 장면 길이 사용

    Returns:
        클립 경로가 추가된 하이라이트 결과 목록
    """
    if not check_ffmpeg():
        logger.error("ffmpeg가 설치되지 않았습니다. 클립 추출을 건너뜁니다.")
        return highlights

    # GCS URI인 경우 로컬 영상 필요
    if video_path.startswith("gs://"):
        logger.warning(
            "클립 추출은 로컬 파일이 필요합니다. "
            "--video에 로컬 파일 경로를 지정하세요."
        )
        return highlights

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for highlight in highlights:
        shot = highlight.shot
        start = shot.start_time
        end = shot.end_time

        # 클립 길이 조정
        if clip_duration is not None:
            mid = (start + end) / 2
            start = max(0, mid - clip_duration / 2)
            end = start + clip_duration

        clip_filename = (
            f"highlight_{highlight.rank:03d}_{int(start)}s-{int(end)}s.mp4"
        )
        clip_path = str(out_dir / clip_filename)

        try:
            extract_clip(
                video_path=video_path,
                start_time=start,
                end_time=end,
                output_path=clip_path,
            )
            highlight.clip_path = clip_path
            logger.info(f"클립 추출 완료 (순위 {highlight.rank}): {clip_path}")
        except Exception as e:
            logger.error(f"클립 추출 실패 (순위 {highlight.rank}): {e}")

    return highlights


def save_analysis_results(
    video_uri: str,
    all_shots: List[ShotInfo],
    highlights: List[HighlightResult],
    merged_path: Optional[str],
    output_dir: str,
) -> str:
    """분석 결과를 JSON 파일로 저장합니다."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = str(out_dir / "highlight_analysis.json")

    result = {
        "video_uri": video_uri,
        "total_shots": len(all_shots),
        "selected_highlights": len(highlights),
        "merged_highlight_path": merged_path,
        "highlights": [h.to_dict() for h in highlights],
        "all_shots_summary": [
            {
                "shot_index": s.shot_index,
                "start_time": round(s.start_time, 3),
                "end_time": round(s.end_time, 3),
                "score": round(s.score, 4),
            }
            for s in sorted(all_shots, key=lambda x: x.shot_index)
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"분석 결과 저장: {json_path}")
    return json_path


def print_highlight_summary(highlights: List[HighlightResult], total_shots: int) -> None:
    """하이라이트 추출 결과 요약을 출력합니다."""
    print("\n" + "=" * 60)
    print("하이라이트 자동 추출 결과")
    print("=" * 60)
    print(f"전체 장면 수: {total_shots}개")
    print(f"선택된 하이라이트: {len(highlights)}개")
    print()

    for h in highlights:
        print(f"  [{h.rank}위] 장면 {h.shot.shot_index + 1}")
        print(f"    시간: {h.shot.start_time:.1f}s ~ {h.shot.end_time:.1f}s ({h.duration:.1f}초)")
        print(f"    점수: {h.shot.score:.4f}")

        breakdown = h.shot.score_breakdown
        print(f"    점수 세부:")
        print(f"      - 레이블: {breakdown.get('label_score', 0):.4f}")
        print(f"      - 객체:   {breakdown.get('object_score', 0):.4f}")
        print(f"      - 텍스트: {breakdown.get('text_score', 0):.4f}")
        print(f"      - 다양성: {breakdown.get('diversity_score', 0):.4f}")

        top_labels = [
            label["description"]
            for label in sorted(
                h.shot.labels,
                key=lambda x: x.get("confidence", 0),
                reverse=True,
            )[:3]
        ]
        if top_labels:
            print(f"    주요 레이블: {', '.join(top_labels)}")

        if h.clip_path:
            print(f"    클립 파일: {h.clip_path}")
        print()

    print("=" * 60)


def main() -> int:
    """메인 실행 함수."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        config = load_config(args.config)
        project_id = args.project or get_project_id(config)

        logger.info(f"프로젝트 ID: {project_id}")
        logger.info(f"영상: {args.video}")

        # 영상 URI 준비
        video_uri = prepare_video_uri(
            video_path=args.video,
            project_id=project_id,
            bucket_name=args.bucket,
            config=config,
        )

        # 영상 분석
        annotation_result = analyze_video_for_highlights(video_uri)

        # 장면 정보 구성
        shots = build_shot_info_list(annotation_result)

        if not shots:
            logger.error("장면 정보를 추출할 수 없습니다.")
            return 1

        # 장면 점수 계산
        shots = calculate_shot_scores(
            shots=shots,
            label_weight=args.label_weight,
            object_weight=args.object_weight,
            text_weight=args.text_weight,
            diversity_weight=args.diversity_weight,
        )

        # 하이라이트 선택
        selected_shots = select_highlights(
            shots=shots,
            max_highlights=args.max_highlights,
            score_threshold=args.score_threshold,
        )

        # HighlightResult 객체 생성
        highlights = [
            HighlightResult(rank=i + 1, shot=shot)
            for i, shot in enumerate(selected_shots)
        ]

        # 클립 추출 (요청된 경우)
        merged_path = None
        if not args.no_extract and not args.video.startswith("gs://"):
            highlights = extract_highlight_clips(
                video_path=args.video,
                highlights=highlights,
                output_dir=args.output_dir,
                clip_duration=args.clip_duration,
            )

            # 클립 합치기 (요청된 경우)
            if args.merge_clips:
                clip_paths = [h.clip_path for h in highlights if h.clip_path]
                if clip_paths:
                    merged_path = str(
                        Path(args.output_dir) / "merged_highlight.mp4"
                    )
                    try:
                        merge_clips(clip_paths, merged_path)
                        logger.info(f"하이라이트 합치기 완료: {merged_path}")
                    except Exception as e:
                        logger.error(f"하이라이트 합치기 실패: {e}")
                        merged_path = None
        elif args.no_extract:
            logger.info("--no-extract 옵션: 클립 추출 건너뜀")
        else:
            logger.warning(
                "GCS URI를 사용할 경우 클립 추출은 로컬 파일 다운로드 후 수행해야 합니다."
            )

        # 결과 출력
        print_highlight_summary(highlights, total_shots=len(shots))

        # JSON 저장
        json_path = save_analysis_results(
            video_uri=video_uri,
            all_shots=shots,
            highlights=highlights,
            merged_path=merged_path,
            output_dir=args.output_dir,
        )
        print(f"\n분석 결과 JSON: {json_path}")
        if merged_path:
            print(f"합친 하이라이트: {merged_path}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"파일 오류: {e}")
        return 1
    except ValueError as e:
        logger.error(f"설정 오류: {e}")
        return 1
    except Exception as e:
        logger.error(f"하이라이트 추출 중 오류 발생: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
