#!/usr/bin/env python3
"""
Lab 5: 콘텐츠 추천 시스템 - Vertex AI + BigQuery

이 스크립트는 BigQuery에 시청 이력 데이터를 구축하고
BigQuery ML을 사용하여 협업 필터링 추천 모델을 학습한 후
개인화된 콘텐츠 추천을 제공합니다.

사용법:
    # 샘플 데이터 설정
    python recommendation_system.py --project PROJECT_ID --action setup_data

    # 모델 학습
    python recommendation_system.py --project PROJECT_ID --action train_model

    # 추천 실행
    python recommendation_system.py --project PROJECT_ID --action recommend --user-id USER_ID

DAN25 연관:
    네이버 동영상의 "비슷한 동영상", "추천 동영상" 기능의 핵심 기술입니다.
    사용자의 시청 이력을 분석하여 개인화된 콘텐츠를 추천합니다.
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# 상위 디렉토리의 src 모듈을 임포트하기 위한 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gcp_client import (
    get_bigquery_client,
    get_project_id,
    load_config,
    setup_logging,
)

logger = logging.getLogger(__name__)

# 영상 카테고리 목록 (네이버 동영상 스타일)
VIDEO_CATEGORIES = [
    "엔터테인먼트", "음악", "스포츠", "뉴스", "교육",
    "게임", "요리", "여행", "패션/뷰티", "기술/IT",
    "영화/드라마", "동물", "건강/운동", "DIY/취미", "시사/정치"
]

# 영상 태그 예시
VIDEO_TAGS = {
    "엔터테인먼트": ["예능", "개그", "웃음", "K-POP", "아이돌", "버라이어티"],
    "음악": ["노래", "뮤직비디오", "라이브", "커버", "힙합", "발라드"],
    "스포츠": ["축구", "야구", "농구", "배구", "골프", "수영", "마라톤"],
    "뉴스": ["시사", "경제", "정치", "국제", "사회", "날씨"],
    "교육": ["강의", "학습", "영어", "수학", "과학", "역사"],
    "게임": ["롤", "스타크래프트", "마인크래프트", "FPS", "모바일게임"],
    "요리": ["레시피", "맛집", "한식", "양식", "디저트", "야식"],
    "여행": ["해외여행", "국내여행", "캠핑", "백패킹", "호텔"],
    "패션/뷰티": ["메이크업", "스킨케어", "OOTD", "헤어", "네일"],
    "기술/IT": ["AI", "코딩", "스마트폰", "컴퓨터", "유튜브팁"],
    "영화/드라마": ["리뷰", "추천", "드라마", "영화", "넷플릭스"],
    "동물": ["강아지", "고양이", "귀여운동물", "펫케어"],
    "건강/운동": ["다이어트", "운동루틴", "홈트", "요가", "필라테스"],
    "DIY/취미": ["공예", "그림", "원예", "인테리어", "독서"],
    "시사/정치": ["국제뉴스", "정치토론", "경제분석"],
}


def parse_args() -> argparse.Namespace:
    """CLI 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="BigQuery + Vertex AI를 사용한 콘텐츠 추천 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
액션:
  setup_data      샘플 데이터 생성 및 BigQuery 테이블 구성
  train_model     BigQuery ML 추천 모델 학습
  recommend       특정 사용자를 위한 추천 실행
  batch_recommend 전체 사용자에 대한 배치 추천
  stats           데이터 및 모델 통계 출력

예시:
  # 샘플 데이터 설정 (100명 사용자, 500개 영상)
  python recommendation_system.py \\
      --project my-project-id \\
      --action setup_data \\
      --num-users 100 \\
      --num-videos 500

  # 모델 학습
  python recommendation_system.py \\
      --project my-project-id \\
      --action train_model

  # user_001에 대한 추천 (상위 10개)
  python recommendation_system.py \\
      --project my-project-id \\
      --action recommend \\
      --user-id user_001 \\
      --top-k 10
        """,
    )

    parser.add_argument(
        "--project",
        default=None,
        help="GCP 프로젝트 ID",
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["setup_data", "train_model", "recommend", "batch_recommend", "stats"],
        help="실행할 액션",
    )
    parser.add_argument(
        "--dataset",
        default="media_ai_lab",
        help="BigQuery 데이터셋 ID (기본: media_ai_lab)",
    )
    parser.add_argument(
        "--location",
        default="asia-northeast3",
        help="BigQuery 위치 (기본: asia-northeast3)",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=100,
        help="생성할 샘플 사용자 수 (기본: 100)",
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=500,
        help="생성할 샘플 영상 수 (기본: 500)",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="추천 대상 사용자 ID (recommend 액션 시 필요)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="추천 영상 수 (기본: 10)",
    )
    parser.add_argument(
        "--output-table",
        default=None,
        help="배치 추천 결과를 저장할 BigQuery 테이블",
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


def setup_bigquery_dataset(
    client,
    project_id: str,
    dataset_id: str,
    location: str,
) -> None:
    """BigQuery 데이터셋과 테이블을 생성합니다."""
    from google.cloud import bigquery

    # 데이터셋 생성
    dataset_ref = f"{project_id}.{dataset_id}"
    try:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        client.create_dataset(dataset, exists_ok=True)
        logger.info(f"데이터셋 준비 완료: {dataset_ref}")
    except Exception as e:
        logger.warning(f"데이터셋 생성 중 경고: {e}")

    # 테이블 스키마 정의
    tables = {
        "watch_history": [
            bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("video_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("watch_timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("watch_duration_seconds", "FLOAT64"),
            bigquery.SchemaField("completion_rate", "FLOAT64"),
            bigquery.SchemaField("liked", "BOOL"),
            bigquery.SchemaField("disliked", "BOOL"),
            bigquery.SchemaField("shared", "BOOL"),
        ],
        "video_metadata": [
            bigquery.SchemaField("video_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("title", "STRING"),
            bigquery.SchemaField("description", "STRING"),
            bigquery.SchemaField("tags", "STRING", mode="REPEATED"),
            bigquery.SchemaField("category", "STRING"),
            bigquery.SchemaField("duration_seconds", "FLOAT64"),
            bigquery.SchemaField("upload_timestamp", "TIMESTAMP"),
            bigquery.SchemaField("view_count", "INT64"),
            bigquery.SchemaField("like_count", "INT64"),
            bigquery.SchemaField("ai_labels", "STRING", mode="REPEATED"),
        ],
        "user_profiles": [
            bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("preferred_categories", "STRING", mode="REPEATED"),
            bigquery.SchemaField("avg_watch_duration", "FLOAT64"),
            bigquery.SchemaField("total_watch_count", "INT64"),
            bigquery.SchemaField("last_active", "TIMESTAMP"),
        ],
    }

    for table_name, schema in tables.items():
        table_ref = f"{dataset_ref}.{table_name}"
        try:
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table, exists_ok=True)
            logger.info(f"테이블 준비 완료: {table_ref}")
        except Exception as e:
            logger.warning(f"테이블 생성 중 경고 ({table_name}): {e}")


def generate_sample_data(
    project_id: str,
    dataset_id: str,
    num_users: int,
    num_videos: int,
) -> tuple:
    """
    샘플 시청 이력 데이터를 생성합니다.

    Returns:
        (watch_history_rows, video_metadata_rows, user_profile_rows) 튜플
    """
    logger.info(f"샘플 데이터 생성 중: {num_users}명 사용자, {num_videos}개 영상")

    now = datetime.now()

    # 영상 메타데이터 생성
    videos = []
    for i in range(num_videos):
        category = random.choice(VIDEO_CATEGORIES)
        tags = random.sample(VIDEO_TAGS.get(category, ["일반"]), min(3, len(VIDEO_TAGS.get(category, ["일반"]))))
        duration = random.uniform(60, 1800)  # 1분 ~ 30분
        view_count = random.randint(100, 1000000)

        videos.append({
            "video_id": f"video_{i + 1:04d}",
            "title": f"{category} 영상 {i + 1:04d} - {tags[0] if tags else ''}",
            "description": f"{category} 관련 영상입니다. 태그: {', '.join(tags)}",
            "tags": tags,
            "category": category,
            "duration_seconds": round(duration, 1),
            "upload_timestamp": (
                now - timedelta(days=random.randint(1, 365))
            ).isoformat(),
            "view_count": view_count,
            "like_count": int(view_count * random.uniform(0.01, 0.1)),
            "ai_labels": random.sample(
                [tag for tag_list in VIDEO_TAGS.values() for tag in tag_list],
                min(5, 10)
            ),
        })

    # 사용자 프로필 생성
    users = []
    for i in range(num_users):
        preferred_categories = random.sample(VIDEO_CATEGORIES, random.randint(2, 5))
        users.append({
            "user_id": f"user_{i + 1:04d}",
            "preferred_categories": preferred_categories,
            "avg_watch_duration": random.uniform(180, 900),
            "total_watch_count": random.randint(10, 500),
            "last_active": (now - timedelta(days=random.randint(0, 30))).isoformat(),
        })

    # 시청 이력 생성
    watch_history = []
    video_ids = [v["video_id"] for v in videos]

    for user in users:
        # 사용자당 10~50개 시청 이력 생성
        num_watches = random.randint(10, 50)
        preferred_cats = user["preferred_categories"]

        # 선호 카테고리의 영상 가중치 높이기
        preferred_videos = [
            v for v in videos if v["category"] in preferred_cats
        ]
        other_videos = [
            v for v in videos if v["category"] not in preferred_cats
        ]

        # 선호 영상 70%, 기타 30%
        watched_videos = []
        if preferred_videos:
            watched_videos.extend(
                random.sample(
                    preferred_videos,
                    min(int(num_watches * 0.7), len(preferred_videos))
                )
            )
        if other_videos:
            watched_videos.extend(
                random.sample(
                    other_videos,
                    min(num_watches - len(watched_videos), len(other_videos))
                )
            )

        for video in watched_videos[:num_watches]:
            completion_rate = random.uniform(0.1, 1.0)
            liked = completion_rate > 0.8 and random.random() < 0.3

            watch_history.append({
                "user_id": user["user_id"],
                "video_id": video["video_id"],
                "watch_timestamp": (
                    now - timedelta(
                        days=random.randint(0, 90),
                        hours=random.randint(0, 23),
                    )
                ).isoformat(),
                "watch_duration_seconds": round(
                    video["duration_seconds"] * completion_rate, 1
                ),
                "completion_rate": round(completion_rate, 4),
                "liked": liked,
                "disliked": not liked and random.random() < 0.05,
                "shared": liked and random.random() < 0.1,
            })

    logger.info(
        f"샘플 데이터 생성 완료: "
        f"{len(videos)}개 영상, {len(users)}명 사용자, {len(watch_history)}개 시청 이력"
    )
    return watch_history, videos, users


def upload_data_to_bigquery(
    client,
    project_id: str,
    dataset_id: str,
    watch_history: list,
    videos: list,
    users: list,
) -> None:
    """생성된 샘플 데이터를 BigQuery에 업로드합니다."""
    logger.info("BigQuery에 데이터 업로드 중...")

    tables = {
        "watch_history": watch_history,
        "video_metadata": videos,
        "user_profiles": users,
    }

    for table_name, rows in tables.items():
        table_ref = f"{project_id}.{dataset_id}.{table_name}"
        table = client.get_table(table_ref)

        errors = client.insert_rows_json(table, rows)
        if errors:
            logger.error(f"데이터 업로드 오류 ({table_name}): {errors[:3]}")
        else:
            logger.info(f"업로드 완료: {table_name} ({len(rows)}행)")


def train_recommendation_model(
    client,
    project_id: str,
    dataset_id: str,
) -> None:
    """BigQuery ML을 사용하여 협업 필터링 추천 모델을 학습합니다."""
    logger.info("추천 모델 학습 시작...")

    # 시청 이력을 평점으로 변환 후 행렬 분해 모델 학습
    model_query = f"""
    CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.recommendation_model`
    OPTIONS(
        model_type = 'matrix_factorization',
        user_col = 'user_id',
        item_col = 'video_id',
        rating_col = 'implicit_rating',
        feedback_type = 'IMPLICIT',
        num_factors = 16
    )
    AS
    SELECT
        user_id,
        video_id,
        -- 시청 완료율과 좋아요를 기반으로 암시적 평점 계산
        (
            completion_rate * 3.0
            + CAST(liked AS FLOAT64) * 2.0
            + CAST(shared AS FLOAT64) * 1.0
        ) AS implicit_rating
    FROM `{project_id}.{dataset_id}.watch_history`
    WHERE completion_rate > 0.1  -- 10% 이상 시청한 영상만 포함
    """

    logger.info("BigQuery ML 모델 학습 쿼리 실행 중... (수 분 소요될 수 있습니다)")
    query_job = client.query(model_query)
    query_job.result()  # 완료 대기
    logger.info("추천 모델 학습 완료")


def get_recommendations_for_user(
    client,
    project_id: str,
    dataset_id: str,
    user_id: str,
    top_k: int = 10,
) -> List[dict]:
    """
    특정 사용자를 위한 개인화 추천을 반환합니다.

    협업 필터링 결과와 콘텐츠 기반 필터링을 앙상블합니다.
    """
    # 1. BigQuery ML 협업 필터링 추천
    collab_query = f"""
    SELECT
        predicted_video_id AS video_id,
        predicted_implicit_rating_confidence AS confidence_score,
        'collaborative' AS recommendation_type
    FROM
        ML.RECOMMEND(
            MODEL `{project_id}.{dataset_id}.recommendation_model`,
            (
                SELECT @user_id AS user_id
            )
        )
    WHERE predicted_video_id NOT IN (
        -- 이미 시청한 영상 제외
        SELECT video_id
        FROM `{project_id}.{dataset_id}.watch_history`
        WHERE user_id = @user_id
    )
    ORDER BY predicted_implicit_rating_confidence DESC
    LIMIT {top_k}
    """

    # 2. 콘텐츠 기반 필터링 (사용자가 많이 본 카테고리의 인기 영상)
    content_query = f"""
    WITH user_categories AS (
        -- 사용자가 많이 시청한 카테고리
        SELECT
            v.category,
            COUNT(*) AS watch_count,
            AVG(h.completion_rate) AS avg_completion
        FROM `{project_id}.{dataset_id}.watch_history` h
        JOIN `{project_id}.{dataset_id}.video_metadata` v
            ON h.video_id = v.video_id
        WHERE h.user_id = @user_id
        GROUP BY v.category
        ORDER BY watch_count DESC, avg_completion DESC
        LIMIT 3
    )
    SELECT
        v.video_id,
        CAST(v.view_count AS FLOAT64) / 1000000.0 AS confidence_score,
        'content_based' AS recommendation_type
    FROM `{project_id}.{dataset_id}.video_metadata` v
    WHERE v.category IN (SELECT category FROM user_categories)
    AND v.video_id NOT IN (
        SELECT video_id
        FROM `{project_id}.{dataset_id}.watch_history`
        WHERE user_id = @user_id
    )
    ORDER BY v.view_count DESC
    LIMIT {top_k}
    """

    recommendations = []
    from google.cloud import bigquery

    query_params = [
        bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
    ]
    job_config = bigquery.QueryJobConfig(query_parameters=query_params)

    # 협업 필터링 시도
    try:
        collab_job = client.query(collab_query, job_config=job_config)
        collab_results = collab_job.result()
        for row in collab_results:
            recommendations.append({
                "video_id": row.video_id,
                "confidence_score": float(row.confidence_score),
                "recommendation_type": row.recommendation_type,
            })
        logger.info(f"협업 필터링 추천: {len(recommendations)}개")
    except Exception as e:
        logger.warning(
            f"협업 필터링 실패 (모델 미학습?): {e}\n"
            "콘텐츠 기반 필터링으로 대체합니다."
        )

    # 협업 필터링 결과가 부족하면 콘텐츠 기반으로 보완
    if len(recommendations) < top_k:
        try:
            content_job = client.query(content_query, job_config=job_config)
            content_results = content_job.result()
            existing_ids = {r["video_id"] for r in recommendations}
            for row in content_results:
                if row.video_id not in existing_ids:
                    recommendations.append({
                        "video_id": row.video_id,
                        "confidence_score": float(row.confidence_score),
                        "recommendation_type": row.recommendation_type,
                    })
            logger.info(f"콘텐츠 기반 보완 후 총 추천: {len(recommendations)}개")
        except Exception as e:
            logger.error(f"콘텐츠 기반 필터링 실패: {e}")

    return recommendations[:top_k]


def enrich_recommendations_with_metadata(
    client,
    project_id: str,
    dataset_id: str,
    recommendations: List[dict],
) -> List[dict]:
    """추천 결과에 영상 메타데이터를 추가합니다."""
    if not recommendations:
        return []

    video_ids = [r["video_id"] for r in recommendations]
    ids_str = ", ".join(f"'{vid}'" for vid in video_ids)

    metadata_query = f"""
    SELECT video_id, title, category, tags, duration_seconds, view_count, like_count
    FROM `{project_id}.{dataset_id}.video_metadata`
    WHERE video_id IN ({ids_str})
    """

    try:
        result = client.query(metadata_query).result()
        metadata_map = {
            row.video_id: {
                "title": row.title,
                "category": row.category,
                "tags": list(row.tags) if row.tags else [],
                "duration_seconds": row.duration_seconds,
                "view_count": row.view_count,
                "like_count": row.like_count,
            }
            for row in result
        }

        # 추천 결과에 메타데이터 합치기
        enriched = []
        for rec in recommendations:
            meta = metadata_map.get(rec["video_id"], {})
            enriched.append({**rec, **meta})

        return enriched
    except Exception as e:
        logger.warning(f"메타데이터 조회 실패: {e}")
        return recommendations


def print_recommendations(
    user_id: str,
    recommendations: List[dict],
) -> None:
    """추천 결과를 보기 좋게 출력합니다."""
    print("\n" + "=" * 60)
    print(f"사용자 {user_id}를 위한 개인화 추천")
    print("=" * 60)

    if not recommendations:
        print("추천 결과가 없습니다.")
        return

    for i, rec in enumerate(recommendations, 1):
        print(f"\n  [{i}위] {rec.get('title', rec['video_id'])}")
        print(f"    카테고리: {rec.get('category', '알 수 없음')}")
        print(f"    추천 방식: {rec.get('recommendation_type', '알 수 없음')}")
        print(f"    신뢰도 점수: {rec.get('confidence_score', 0):.4f}")

        if rec.get("view_count"):
            print(f"    조회수: {rec['view_count']:,}회")
        if rec.get("tags"):
            print(f"    태그: {', '.join(rec['tags'][:3])}")
        if rec.get("duration_seconds"):
            mins = int(rec["duration_seconds"] // 60)
            secs = int(rec["duration_seconds"] % 60)
            print(f"    길이: {mins}분 {secs}초")

    print("\n" + "=" * 60)


def print_statistics(
    client,
    project_id: str,
    dataset_id: str,
) -> None:
    """데이터 통계를 출력합니다."""
    print("\n" + "=" * 60)
    print("데이터셋 통계")
    print("=" * 60)

    stats_queries = {
        "사용자 수": f"SELECT COUNT(DISTINCT user_id) as cnt FROM `{project_id}.{dataset_id}.user_profiles`",
        "영상 수": f"SELECT COUNT(*) as cnt FROM `{project_id}.{dataset_id}.video_metadata`",
        "시청 이력 수": f"SELECT COUNT(*) as cnt FROM `{project_id}.{dataset_id}.watch_history`",
        "평균 시청 완료율": f"SELECT ROUND(AVG(completion_rate), 4) as cnt FROM `{project_id}.{dataset_id}.watch_history`",
    }

    for label, query in stats_queries.items():
        try:
            result = list(client.query(query).result())
            value = result[0].cnt if result else "N/A"
            print(f"  {label}: {value}")
        except Exception as e:
            print(f"  {label}: 조회 실패 ({e})")

    print("=" * 60)


def main() -> int:
    """메인 실행 함수."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        config = load_config(args.config)
        project_id = args.project or get_project_id(config)
        dataset_id = args.dataset

        logger.info(f"프로젝트 ID: {project_id}")
        logger.info(f"데이터셋: {dataset_id}")
        logger.info(f"액션: {args.action}")

        client = get_bigquery_client(project_id)

        if args.action == "setup_data":
            # BigQuery 데이터셋 및 테이블 설정
            setup_bigquery_dataset(client, project_id, dataset_id, args.location)

            # 샘플 데이터 생성
            watch_history, videos, users = generate_sample_data(
                project_id=project_id,
                dataset_id=dataset_id,
                num_users=args.num_users,
                num_videos=args.num_videos,
            )

            # BigQuery에 업로드
            upload_data_to_bigquery(
                client=client,
                project_id=project_id,
                dataset_id=dataset_id,
                watch_history=watch_history,
                videos=videos,
                users=users,
            )

            print(f"\n데이터 설정 완료!")
            print(f"  사용자: {len(users)}명")
            print(f"  영상: {len(videos)}개")
            print(f"  시청 이력: {len(watch_history)}건")
            print(f"\n다음 단계: python recommendation_system.py --project {project_id} --action train_model")

        elif args.action == "train_model":
            train_recommendation_model(client, project_id, dataset_id)
            print("\n모델 학습 완료!")
            print(f"\n다음 단계: python recommendation_system.py --project {project_id} --action recommend --user-id user_0001")

        elif args.action == "recommend":
            if not args.user_id:
                logger.error("--user-id 옵션이 필요합니다.")
                return 1

            recommendations = get_recommendations_for_user(
                client=client,
                project_id=project_id,
                dataset_id=dataset_id,
                user_id=args.user_id,
                top_k=args.top_k,
            )

            # 메타데이터 추가
            recommendations = enrich_recommendations_with_metadata(
                client=client,
                project_id=project_id,
                dataset_id=dataset_id,
                recommendations=recommendations,
            )

            print_recommendations(args.user_id, recommendations)

        elif args.action == "batch_recommend":
            logger.info("전체 사용자 배치 추천 실행 중...")
            # 사용자 목록 조회
            users_query = f"""
            SELECT DISTINCT user_id
            FROM `{project_id}.{dataset_id}.user_profiles`
            LIMIT 10
            """
            users = [row.user_id for row in client.query(users_query).result()]
            logger.info(f"추천 대상 사용자: {len(users)}명")

            all_recommendations = {}
            for user_id in users:
                recs = get_recommendations_for_user(
                    client=client,
                    project_id=project_id,
                    dataset_id=dataset_id,
                    user_id=user_id,
                    top_k=args.top_k,
                )
                all_recommendations[user_id] = recs
                logger.info(f"  {user_id}: {len(recs)}개 추천")

            print(f"\n배치 추천 완료: {len(users)}명 처리")
            if args.output_table:
                logger.info(f"결과 저장: {args.output_table} (구현 예정)")

        elif args.action == "stats":
            print_statistics(client, project_id, dataset_id)

        return 0

    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
