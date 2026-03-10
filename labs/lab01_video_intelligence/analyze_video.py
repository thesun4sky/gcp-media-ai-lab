#!/usr/bin/env python3
"""
Lab 1: 영상 AI 분석 - Google Cloud Video Intelligence API

이 스크립트는 Video Intelligence API를 사용하여 영상에서
장면 전환, 레이블, 객체, 텍스트를 자동으로 감지합니다.

사용법:
    python analyze_video.py --video gs://bucket/video.mp4 --project PROJECT_ID

DAN25 연관:
    네이버 동영상 플랫폼에서 자동 태깅, 썸네일 추천, 콘텐츠 검색에 활용되는
    핵심 기술을 직접 실습합니다.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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

logger = logging.getLogger(__name__)


SUPPORTED_FEATURES = {
    "SHOT_CHANGE_DETECTION": "장면 전환 감지",
    "LABEL_DETECTION": "레이블(물체/장소/활동) 감지",
    "OBJECT_TRACKING": "객체 추적",
    "TEXT_DETECTION": "텍스트 감지",
}


def parse_args() -> argparse.Namespace:
    """CLI 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="Google Cloud Video Intelligence API를 사용한 영상 분석",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # GCS URI로 분석
  python analyze_video.py \\
      --video gs://cloud-samples-data/video/animals.mp4 \\
      --project my-project-id

  # 로컬 파일 분석 (자동으로 GCS에 업로드)
  python analyze_video.py \\
      --video /path/to/video.mp4 \\
      --project my-project-id \\
      --bucket my-bucket

  # 특정 기능만 선택
  python analyze_video.py \\
      --video gs://bucket/video.mp4 \\
      --project my-project-id \\
      --features SHOT_CHANGE_DETECTION LABEL_DETECTION

  # 결과를 JSON 파일로 저장
  python analyze_video.py \\
      --video gs://bucket/video.mp4 \\
      --project my-project-id \\
      --output results.json
        """,
    )

    parser.add_argument(
        "--video",
        required=True,
        help="분석할 영상 파일 경로 또는 GCS URI (예: gs://bucket/video.mp4)",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="GCP 프로젝트 ID (지정 안 하면 환경 변수 또는 config.yaml에서 읽음)",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="로컬 파일 업로드용 GCS 버킷 이름",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=list(SUPPORTED_FEATURES.keys()),
        default=list(SUPPORTED_FEATURES.keys()),
        help=f"감지할 기능 목록 (기본: 전체). 선택 가능: {', '.join(SUPPORTED_FEATURES.keys())}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="결과를 저장할 JSON 파일 경로 (지정 안 하면 콘솔만 출력)",
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
    """
    영상 URI를 준비합니다.
    로컬 파일인 경우 GCS에 업로드합니다.

    Args:
        video_path: 영상 파일 경로 또는 GCS URI
        project_id: GCP 프로젝트 ID
        bucket_name: GCS 버킷 이름 (로컬 파일 업로드 시 필요)
        config: 설정 딕셔너리

    Returns:
        GCS URI 문자열
    """
    if video_path.startswith("gs://"):
        logger.info(f"GCS URI 사용: {video_path}")
        return video_path

    # 로컬 파일인 경우 GCS에 업로드
    local_path = Path(video_path)
    if not local_path.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    if bucket_name is None:
        bucket_name = config.get("gcp", {}).get("bucket_name")
    if not bucket_name:
        raise ValueError(
            "로컬 파일을 사용하려면 --bucket 옵션으로 GCS 버킷을 지정하거나 "
            "config.yaml에 bucket_name을 설정하세요."
        )

    logger.info(f"로컬 파일을 GCS에 업로드 중: {video_path}")
    gcs_uri = upload_file_to_gcs(
        local_file_path=video_path,
        bucket_name=bucket_name,
        destination_blob_name=f"lab01/{local_path.name}",
        project_id=project_id,
    )
    return gcs_uri


def analyze_video(
    video_uri: str,
    features: List[str],
    project_id: str,
) -> dict:
    """
    Video Intelligence API로 영상을 분석합니다.

    Args:
        video_uri: 분석할 영상의 GCS URI
        features: 감지할 기능 목록
        project_id: GCP 프로젝트 ID

    Returns:
        분석 결과 딕셔너리
    """
    from google.cloud import videointelligence

    client = get_video_intelligence_client()

    # API 기능 매핑
    feature_map = {
        "SHOT_CHANGE_DETECTION": videointelligence.Feature.SHOT_CHANGE_DETECTION,
        "LABEL_DETECTION": videointelligence.Feature.LABEL_DETECTION,
        "OBJECT_TRACKING": videointelligence.Feature.OBJECT_TRACKING,
        "TEXT_DETECTION": videointelligence.Feature.TEXT_DETECTION,
    }

    api_features = [feature_map[f] for f in features]

    logger.info(f"영상 분석 시작: {video_uri}")
    logger.info(f"감지 기능: {', '.join(SUPPORTED_FEATURES[f] for f in features)}")

    # 비동기 분석 요청
    start_time = time.time()
    operation = client.annotate_video(
        request={
            "input_uri": video_uri,
            "features": api_features,
        }
    )

    logger.info("분석 진행 중... (영상 길이에 따라 수 분 소요될 수 있습니다)")

    # 완료 대기
    result = operation.result(timeout=600)
    elapsed = time.time() - start_time
    logger.info(f"분석 완료 (소요 시간: {elapsed:.1f}초)")

    # 결과 파싱
    annotation_result = result.annotation_results[0]
    analysis_data = {
        "video_uri": video_uri,
        "analysis_time": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "features_analyzed": features,
    }

    # 장면 전환 결과 파싱
    if "SHOT_CHANGE_DETECTION" in features:
        analysis_data["shot_changes"] = parse_shot_changes(annotation_result)

    # 레이블 감지 결과 파싱
    if "LABEL_DETECTION" in features:
        analysis_data["labels"] = parse_labels(annotation_result)

    # 객체 추적 결과 파싱
    if "OBJECT_TRACKING" in features:
        analysis_data["objects"] = parse_objects(annotation_result)

    # 텍스트 감지 결과 파싱
    if "TEXT_DETECTION" in features:
        analysis_data["texts"] = parse_texts(annotation_result)

    return analysis_data


def parse_shot_changes(annotation_result) -> list:
    """장면 전환 분석 결과를 파싱합니다."""
    shots = []
    for i, shot in enumerate(annotation_result.shot_annotations):
        start = shot.start_time_offset.total_seconds()
        end = shot.end_time_offset.total_seconds()
        shots.append({
            "shot_index": i,
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "duration": round(end - start, 3),
        })

    logger.info(f"장면 전환: {len(shots)}개 장면 감지")
    return shots


def parse_labels(annotation_result) -> list:
    """레이블 감지 결과를 파싱합니다."""
    labels = []

    # 영상 전체 레이블
    for label in annotation_result.segment_label_annotations:
        for segment in label.segments:
            confidence = segment.confidence
            labels.append({
                "description": label.entity.description,
                "confidence": round(confidence, 4),
                "type": "segment_level",
                "start_time": round(
                    segment.segment.start_time_offset.total_seconds(), 3
                ),
                "end_time": round(
                    segment.segment.end_time_offset.total_seconds(), 3
                ),
            })

    # 장면별 레이블
    for label in annotation_result.shot_label_annotations:
        for segment in label.segments:
            labels.append({
                "description": label.entity.description,
                "confidence": round(segment.confidence, 4),
                "type": "shot_level",
                "start_time": round(
                    segment.segment.start_time_offset.total_seconds(), 3
                ),
                "end_time": round(
                    segment.segment.end_time_offset.total_seconds(), 3
                ),
            })

    # 신뢰도 기준 내림차순 정렬
    labels.sort(key=lambda x: x["confidence"], reverse=True)
    logger.info(f"레이블: {len(labels)}개 감지")
    return labels


def parse_objects(annotation_result) -> list:
    """객체 추적 결과를 파싱합니다."""
    objects = []

    for obj in annotation_result.object_annotations:
        frames = []
        for frame in obj.frames:
            box = frame.normalized_bounding_box
            frames.append({
                "time": round(frame.time_offset.total_seconds(), 3),
                "bounding_box": {
                    "left": round(box.left, 4),
                    "top": round(box.top, 4),
                    "right": round(box.right, 4),
                    "bottom": round(box.bottom, 4),
                },
            })

        objects.append({
            "description": obj.entity.description,
            "confidence": round(obj.confidence, 4),
            "track_id": obj.track_id,
            "start_time": round(
                obj.segment.start_time_offset.total_seconds(), 3
            ),
            "end_time": round(
                obj.segment.end_time_offset.total_seconds(), 3
            ),
            "frames": frames[:10],  # 첫 10 프레임만 포함 (결과 크기 제한)
            "total_frames": len(frames),
        })

    logger.info(f"객체 추적: {len(objects)}개 객체 감지")
    return objects


def parse_texts(annotation_result) -> list:
    """텍스트 감지 결과를 파싱합니다."""
    texts = []

    for text_ann in annotation_result.text_annotations:
        segments = []
        for segment in text_ann.segments:
            frames = []
            for frame in segment.frames:
                vertices = [
                    {"x": v.x, "y": v.y}
                    for v in frame.rotated_bounding_box.vertices
                ]
                frames.append({
                    "time": round(frame.time_offset.total_seconds(), 3),
                    "bounding_box": vertices,
                })

            segments.append({
                "confidence": round(segment.confidence, 4),
                "start_time": round(
                    segment.segment.start_time_offset.total_seconds(), 3
                ),
                "end_time": round(
                    segment.segment.end_time_offset.total_seconds(), 3
                ),
                "frames": frames[:5],
            })

        texts.append({
            "text": text_ann.text,
            "segments": segments,
        })

    logger.info(f"텍스트: {len(texts)}개 텍스트 감지")
    return texts


def print_analysis_summary(analysis_data: dict) -> None:
    """분석 결과 요약을 출력합니다."""
    print("\n" + "=" * 60)
    print("영상 분석 결과 요약")
    print("=" * 60)
    print(f"영상 URI: {analysis_data['video_uri']}")
    print(f"분석 시간: {analysis_data['analysis_time']}")
    print(f"소요 시간: {analysis_data['elapsed_seconds']}초")
    print()

    # 장면 전환
    if "shot_changes" in analysis_data:
        shots = analysis_data["shot_changes"]
        print(f"[장면 전환] 총 {len(shots)}개 장면 감지")
        for shot in shots[:5]:
            print(
                f"  장면 {shot['shot_index'] + 1}: "
                f"{shot['start_time']:.1f}s ~ {shot['end_time']:.1f}s "
                f"({shot['duration']:.1f}초)"
            )
        if len(shots) > 5:
            print(f"  ... 외 {len(shots) - 5}개 장면")
        print()

    # 레이블 (신뢰도 상위 10개)
    if "labels" in analysis_data:
        labels = analysis_data["labels"]
        unique_labels = {}
        for label in labels:
            desc = label["description"]
            if desc not in unique_labels or label["confidence"] > unique_labels[desc]:
                unique_labels[desc] = label["confidence"]

        print(f"[레이블] 총 {len(unique_labels)}개 고유 레이블 감지 (상위 10개)")
        sorted_labels = sorted(
            unique_labels.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for desc, conf in sorted_labels:
            bar = "#" * int(conf * 20)
            print(f"  {desc:30s} {conf:.2%} [{bar:20s}]")
        print()

    # 객체 추적
    if "objects" in analysis_data:
        objects = analysis_data["objects"]
        print(f"[객체 추적] 총 {len(objects)}개 객체 추적")
        for obj in objects[:5]:
            print(
                f"  {obj['description']:30s} "
                f"신뢰도: {obj['confidence']:.2%}, "
                f"{obj['start_time']:.1f}s ~ {obj['end_time']:.1f}s"
            )
        print()

    # 텍스트 감지
    if "texts" in analysis_data:
        texts = analysis_data["texts"]
        print(f"[텍스트] 총 {len(texts)}개 텍스트 감지")
        for text in texts[:5]:
            print(f"  '{text['text']}'")
        print()

    print("=" * 60)


def main() -> int:
    """메인 실행 함수."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        # 설정 로드
        config = load_config(args.config)
        project_id = args.project or get_project_id(config)

        logger.info(f"프로젝트 ID: {project_id}")
        logger.info(f"분석할 영상: {args.video}")

        # 영상 URI 준비
        video_uri = prepare_video_uri(
            video_path=args.video,
            project_id=project_id,
            bucket_name=args.bucket,
            config=config,
        )

        # 영상 분석 실행
        analysis_data = analyze_video(
            video_uri=video_uri,
            features=args.features,
            project_id=project_id,
        )

        # 결과 요약 출력
        print_analysis_summary(analysis_data)

        # JSON 파일로 저장
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            logger.info(f"분석 결과 저장 완료: {args.output}")
            print(f"\n결과 파일 저장: {args.output}")
        else:
            # 전체 결과를 JSON으로 출력
            print("\n전체 분석 결과 (JSON):")
            print(json.dumps(analysis_data, ensure_ascii=False, indent=2))

        return 0

    except FileNotFoundError as e:
        logger.error(f"파일 오류: {e}")
        return 1
    except ValueError as e:
        logger.error(f"설정 오류: {e}")
        return 1
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
