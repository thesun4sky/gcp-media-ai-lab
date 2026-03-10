#!/usr/bin/env python3
"""
Lab 6: 생성형 AI 콘텐츠 창작 - Vertex AI Gemini 1.5 Pro

이 스크립트는 Gemini 1.5 Pro를 사용하여 영상 관련 콘텐츠를 자동으로 생성합니다.
영상 요약, 태그 생성, 썸네일 설명, 제목 최적화 등을 지원합니다.

사용법:
    python gemini_media_ai.py --project PROJECT_ID --action summarize --subtitle subtitle.srt

DAN25 연관:
    네이버 동영상의 크리에이터 지원 도구 - 영상 업로드 시 AI가 자동으로
    제목, 태그, 설명, 썸네일 방향을 제안하여 크리에이터 생산성을 향상시킵니다.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 상위 디렉토리의 src 모듈을 임포트하기 위한 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gcp_client import (
    get_project_id,
    get_vertex_ai_client,
    load_config,
    setup_logging,
)
from src.subtitle_utils import parse_srt, parse_vtt

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """CLI 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="Vertex AI Gemini를 사용한 생성형 AI 콘텐츠 창작",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
액션:
  summarize          영상 자막을 기반으로 요약 생성
  generate_tags      자동 태그 생성
  thumbnail_desc     썸네일 설명 생성
  optimize_title     제목 최적화 제안
  generate_description  SEO 최적화 영상 설명 생성
  analyze_comments   댓글 분석 및 응답 초안 생성
  all                모든 기능 실행

예시:
  # 영상 요약
  python gemini_media_ai.py \\
      --project my-project-id \\
      --action summarize \\
      --subtitle labs/lab02_speech_subtitle/sample.srt

  # 태그 생성
  python gemini_media_ai.py \\
      --project my-project-id \\
      --action generate_tags \\
      --subtitle labs/lab02_speech_subtitle/sample.srt \\
      --num-tags 20

  # 전체 파이프라인
  python gemini_media_ai.py \\
      --project my-project-id \\
      --action all \\
      --subtitle labs/lab02_speech_subtitle/sample.srt \\
      --title "Media AI 실습 가이드" \\
      --output data/output/gemini_results.json
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
        choices=[
            "summarize",
            "generate_tags",
            "thumbnail_desc",
            "optimize_title",
            "generate_description",
            "analyze_comments",
            "all",
        ],
        help="실행할 액션",
    )
    parser.add_argument(
        "--subtitle",
        default=None,
        help="자막 파일 경로 (SRT 또는 VTT)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="영상 제목",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="영상 요약 (thumbnail_desc 액션에 사용)",
    )
    parser.add_argument(
        "--comments",
        default=None,
        help="분석할 시청자 댓글 (쉼표로 구분)",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="직접 입력할 텍스트 (자막 파일 대신)",
    )
    parser.add_argument(
        "--num-tags",
        type=int,
        default=15,
        help="생성할 태그 수 (기본: 15)",
    )
    parser.add_argument(
        "--model",
        default="gemini-1.5-flash",
        choices=["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
        help="사용할 Gemini 모델 (기본: gemini-1.5-flash, 비용 절약)",
    )
    parser.add_argument(
        "--language",
        default="ko",
        choices=["ko", "en", "ja", "zh-CN"],
        help="생성 언어 (기본: ko 한국어)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="생성 창의성 조절 (0.0~1.0, 기본: 0.7)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="결과를 저장할 JSON 파일 경로",
    )
    parser.add_argument(
        "--location",
        default="asia-northeast3",
        help="Vertex AI 위치 (기본: asia-northeast3)",
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


def load_subtitle_text(subtitle_path: str) -> str:
    """
    자막 파일에서 텍스트만 추출합니다.

    Args:
        subtitle_path: SRT 또는 VTT 파일 경로

    Returns:
        자막 텍스트 (타임스탬프 제외)

    Raises:
        FileNotFoundError: 파일이 존재하지 않는 경우
    """
    path = Path(subtitle_path)
    if not path.exists():
        raise FileNotFoundError(f"자막 파일을 찾을 수 없습니다: {subtitle_path}")

    content = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".srt":
        entries = parse_srt(content)
    elif suffix == ".vtt":
        entries = parse_vtt(content)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix}")

    # 텍스트만 연결
    full_text = " ".join(entry.text for entry in entries)
    logger.info(f"자막 텍스트 로드: {len(entries)}개 항목, {len(full_text)}자")
    return full_text


def get_gemini_model(model_name: str, location: str, project_id: str):
    """Gemini 모델 인스턴스를 반환합니다."""
    from vertexai.generative_models import GenerativeModel

    get_vertex_ai_client(project_id=project_id, location=location)
    model = GenerativeModel(model_name)
    logger.info(f"Gemini 모델 로드: {model_name}")
    return model


def generate_content(
    model,
    prompt: str,
    temperature: float = 0.7,
    max_output_tokens: int = 2048,
) -> str:
    """
    Gemini 모델로 콘텐츠를 생성합니다.

    Args:
        model: Gemini 모델 인스턴스
        prompt: 입력 프롬프트
        temperature: 생성 창의성 (0.0~1.0)
        max_output_tokens: 최대 출력 토큰 수

    Returns:
        생성된 텍스트
    """
    from vertexai.generative_models import GenerationConfig

    generation_config = GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=0.9,
    )

    response = model.generate_content(
        prompt,
        generation_config=generation_config,
    )

    return response.text.strip()


def summarize_video(
    model,
    subtitle_text: str,
    language: str = "ko",
    temperature: float = 0.5,
) -> dict:
    """
    영상 자막을 기반으로 요약을 생성합니다.

    Args:
        model: Gemini 모델
        subtitle_text: 영상 자막 전체 텍스트
        language: 생성 언어 코드
        temperature: 생성 창의성

    Returns:
        요약 결과 딕셔너리
    """
    lang_instructions = {
        "ko": "한국어로 작성하세요.",
        "en": "Write in English.",
        "ja": "日本語で書いてください。",
        "zh-CN": "请用中文（简体）写。",
    }
    lang_instruction = lang_instructions.get(language, "한국어로 작성하세요.")

    prompt = f"""당신은 동영상 플랫폼의 AI 콘텐츠 분석가입니다.
다음 영상 자막을 분석하여 시청자와 크리에이터에게 유용한 요약을 제공합니다.
{lang_instruction}

영상 자막:
---
{subtitle_text[:3000]}
---

다음 형식으로 요약을 작성해주세요:

**간략 요약** (2-3문장):
[간결하고 핵심을 담은 요약]

**주요 내용** (3-5개 항목):
- [핵심 내용 1]
- [핵심 내용 2]
- [핵심 내용 3]

**대상 시청자**:
[이 영상이 도움이 될 시청자 유형]

**핵심 학습 포인트**:
[시청자가 얻을 수 있는 가장 중요한 인사이트]
"""

    logger.info("영상 요약 생성 중...")
    summary_text = generate_content(model, prompt, temperature=temperature)

    return {
        "action": "summarize",
        "language": language,
        "summary": summary_text,
        "input_length": len(subtitle_text),
    }


def generate_tags(
    model,
    subtitle_text: str,
    title: Optional[str] = None,
    num_tags: int = 15,
    language: str = "ko",
    temperature: float = 0.6,
) -> dict:
    """
    영상 내용 기반으로 검색 태그를 자동 생성합니다.

    Args:
        model: Gemini 모델
        subtitle_text: 자막 텍스트
        title: 영상 제목 (선택)
        num_tags: 생성할 태그 수
        language: 언어 코드
        temperature: 생성 창의성

    Returns:
        태그 생성 결과 딕셔너리
    """
    title_section = f"영상 제목: {title}\n" if title else ""

    prompt = f"""당신은 동영상 SEO 전문가입니다.
영상 콘텐츠를 분석하여 검색 노출을 극대화하는 태그를 생성합니다.

{title_section}영상 자막:
---
{subtitle_text[:2000]}
---

다음 기준으로 정확히 {num_tags}개의 태그를 생성해주세요:
1. 주제 관련 핵심 키워드 (5개)
2. 장르/형식 태그 (3개)
3. 연관 검색어 (4개)
4. 대상 시청자 태그 (3개)

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "tags": ["태그1", "태그2", "태그3", "..."],
    "primary_keywords": ["핵심키워드1", "핵심키워드2"],
    "category_suggestion": "카테고리 이름"
}}
"""

    logger.info(f"태그 {num_tags}개 생성 중...")
    response_text = generate_content(model, prompt, temperature=temperature)

    # JSON 파싱 시도
    try:
        # 응답에서 JSON 부분만 추출
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result_data = json.loads(json_match.group())
        else:
            # JSON 파싱 실패 시 텍스트에서 태그 추출
            tags = [t.strip().strip('"') for t in response_text.split(",")[:num_tags]]
            result_data = {"tags": tags}
    except json.JSONDecodeError:
        result_data = {"tags": [], "raw_response": response_text}

    return {
        "action": "generate_tags",
        "language": language,
        "num_requested": num_tags,
        **result_data,
    }


def generate_thumbnail_description(
    model,
    title: str,
    summary: Optional[str] = None,
    subtitle_text: Optional[str] = None,
    temperature: float = 0.8,
) -> dict:
    """
    효과적인 썸네일을 위한 AI 설명을 생성합니다.

    Args:
        model: Gemini 모델
        title: 영상 제목
        summary: 영상 요약 (선택)
        subtitle_text: 자막 텍스트 (선택)
        temperature: 생성 창의성

    Returns:
        썸네일 설명 결과 딕셔너리
    """
    context = ""
    if summary:
        context += f"영상 요약: {summary}\n"
    elif subtitle_text:
        context += f"자막 일부: {subtitle_text[:500]}\n"

    prompt = f"""당신은 동영상 플랫폼의 썸네일 디자인 전문가입니다.
클릭률(CTR)을 높이는 효과적인 썸네일을 제안합니다.

영상 제목: {title}
{context}

다음 항목을 포함하여 썸네일 방향을 제안해주세요:

**메인 비주얼**:
[썸네일 중앙에 배치할 핵심 이미지 설명]

**텍스트 오버레이**:
[썸네일에 넣을 짧고 강렬한 문구 (최대 5단어)]

**색상 팔레트**:
[배경색, 강조색, 텍스트색 제안]

**구도 및 레이아웃**:
[이미지 배치, 시선 흐름 방향]

**A/B 테스트 대안**:
[다른 스타일의 썸네일 옵션 2가지]

**클릭률 예상 포인트**:
[시청자가 클릭하게 만드는 핵심 요소]
"""

    logger.info("썸네일 설명 생성 중...")
    description = generate_content(model, prompt, temperature=temperature)

    return {
        "action": "thumbnail_desc",
        "title": title,
        "thumbnail_guide": description,
    }


def optimize_title(
    model,
    original_title: str,
    subtitle_text: Optional[str] = None,
    num_suggestions: int = 5,
    temperature: float = 0.8,
) -> dict:
    """
    클릭률을 높이는 제목 최적화 제안을 생성합니다.

    Args:
        model: Gemini 모델
        original_title: 원본 제목
        subtitle_text: 자막 텍스트 (선택)
        num_suggestions: 제안 수
        temperature: 생성 창의성

    Returns:
        제목 최적화 결과 딕셔너리
    """
    context = ""
    if subtitle_text:
        context = f"\n자막 일부:\n{subtitle_text[:1000]}"

    prompt = f"""당신은 동영상 플랫폼의 콘텐츠 최적화 전문가입니다.
클릭률을 극대화하는 제목을 제안합니다.

원본 제목: {original_title}
{context}

다음 기준으로 {num_suggestions}개의 최적화된 제목을 제안하세요:
1. 호기심 자극형 (1개)
2. 혜택/결과 강조형 (1개)
3. 숫자/리스트형 (1개)
4. 질문형 (1개)
5. 감성/공감형 (1개)

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "suggestions": [
        {{
            "title": "제목1",
            "type": "호기심 자극형",
            "ctr_strategy": "전략 설명"
        }},
        {{
            "title": "제목2",
            "type": "혜택/결과 강조형",
            "ctr_strategy": "전략 설명"
        }}
    ],
    "best_recommendation": "가장 추천하는 제목",
    "improvement_points": "원본 제목 개선 포인트"
}}
"""

    logger.info("제목 최적화 제안 생성 중...")
    response_text = generate_content(model, prompt, temperature=temperature)

    try:
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result_data = json.loads(json_match.group())
        else:
            result_data = {"raw_response": response_text}
    except json.JSONDecodeError:
        result_data = {"raw_response": response_text}

    return {
        "action": "optimize_title",
        "original_title": original_title,
        **result_data,
    }


def generate_video_description(
    model,
    subtitle_text: str,
    title: Optional[str] = None,
    language: str = "ko",
    temperature: float = 0.6,
) -> dict:
    """
    SEO 최적화된 영상 설명을 생성합니다.

    Args:
        model: Gemini 모델
        subtitle_text: 자막 텍스트
        title: 영상 제목 (선택)
        language: 언어 코드
        temperature: 생성 창의성

    Returns:
        영상 설명 생성 결과 딕셔너리
    """
    title_section = f"영상 제목: {title}\n" if title else ""

    prompt = f"""당신은 동영상 플랫폼의 SEO 전문 콘텐츠 라이터입니다.
검색 최적화와 시청자 참여를 높이는 영상 설명을 작성합니다.

{title_section}영상 자막:
---
{subtitle_text[:2500]}
---

다음 구조로 영상 설명을 작성해주세요:

**첫 문단** (2-3문장, 핵심 키워드 포함):
[시청자가 무엇을 배울 수 있는지 명확히 설명]

**영상 내용** (항목별):
⏱ 00:00 - [섹션 1]
⏱ 03:00 - [섹션 2]
⏱ 07:00 - [섹션 3]

**이 영상을 보면 좋은 분**:
✅ [대상 1]
✅ [대상 2]
✅ [대상 3]

**관련 키워드**:
#태그1 #태그2 #태그3 #태그4 #태그5

**채널 소개**:
[채널 소개 및 구독 유도 문구]
"""

    logger.info("영상 설명 생성 중...")
    description = generate_content(model, prompt, temperature=temperature)

    return {
        "action": "generate_description",
        "language": language,
        "description": description,
    }


def analyze_comments(
    model,
    comments: List[str],
    title: Optional[str] = None,
    temperature: float = 0.5,
) -> dict:
    """
    시청자 댓글을 분석하고 크리에이터 응답 초안을 생성합니다.

    Args:
        model: Gemini 모델
        comments: 댓글 목록
        title: 영상 제목 (선택)
        temperature: 생성 창의성

    Returns:
        댓글 분석 결과 딕셔너리
    """
    comments_text = "\n".join(f"- {c}" for c in comments)
    title_section = f"영상 제목: {title}\n" if title else ""

    prompt = f"""당신은 동영상 크리에이터를 돕는 AI 어시스턴트입니다.
시청자 댓글을 분석하고 효과적인 응답을 제안합니다.

{title_section}
시청자 댓글:
{comments_text}

다음 형식으로 분석 결과를 제공해주세요:

**댓글 감성 분석**:
- 긍정적 댓글: X개 (X%)
- 중립적 댓글: X개 (X%)
- 부정적 댓글: X개 (X%)

**주요 피드백 주제**:
1. [주제 1]: [내용 요약]
2. [주제 2]: [내용 요약]

**시청자 요청 사항**:
- [요청 1]
- [요청 2]

**추천 응답 초안** (각 댓글 유형별):

[긍정 댓글 응답 예시]:
[응답 내용]

[질문/요청 댓글 응답 예시]:
[응답 내용]

**다음 콘텐츠 아이디어** (댓글 기반):
1. [아이디어 1]
2. [아이디어 2]
"""

    logger.info(f"댓글 {len(comments)}개 분석 중...")
    analysis = generate_content(model, prompt, temperature=temperature)

    return {
        "action": "analyze_comments",
        "comment_count": len(comments),
        "analysis": analysis,
    }


def run_all_actions(
    model,
    subtitle_text: str,
    title: Optional[str],
    language: str,
    temperature: float,
    num_tags: int,
) -> Dict[str, Any]:
    """모든 생성 액션을 순서대로 실행합니다."""
    results = {}

    # 1. 요약
    logger.info("[1/5] 영상 요약 생성...")
    results["summarize"] = summarize_video(
        model=model,
        subtitle_text=subtitle_text,
        language=language,
        temperature=temperature,
    )

    # 2. 태그
    logger.info("[2/5] 태그 생성...")
    results["tags"] = generate_tags(
        model=model,
        subtitle_text=subtitle_text,
        title=title,
        num_tags=num_tags,
        language=language,
        temperature=temperature,
    )

    # 3. 썸네일 설명
    if title:
        logger.info("[3/5] 썸네일 설명 생성...")
        summary_text = results["summarize"].get("summary", "")
        results["thumbnail"] = generate_thumbnail_description(
            model=model,
            title=title,
            summary=summary_text[:500] if summary_text else None,
            temperature=temperature + 0.1,
        )

    # 4. 제목 최적화
    if title:
        logger.info("[4/5] 제목 최적화 제안...")
        results["title_optimization"] = optimize_title(
            model=model,
            original_title=title,
            subtitle_text=subtitle_text,
            temperature=temperature + 0.1,
        )

    # 5. 영상 설명
    logger.info("[5/5] 영상 설명 생성...")
    results["description"] = generate_video_description(
        model=model,
        subtitle_text=subtitle_text,
        title=title,
        language=language,
        temperature=temperature,
    )

    return results


def print_results(results: dict) -> None:
    """결과를 보기 좋게 출력합니다."""
    print("\n" + "=" * 60)
    print("Gemini Media AI 생성 결과")
    print("=" * 60)

    action = results.get("action", "all")

    if "summary" in results:
        print("\n[영상 요약]")
        print(results["summary"])

    elif "tags" in results:
        print("\n[생성된 태그]")
        tags = results.get("tags", [])
        if tags:
            print(", ".join(f"#{tag}" for tag in tags))
        if "primary_keywords" in results:
            print(f"\n핵심 키워드: {', '.join(results['primary_keywords'])}")
        if "category_suggestion" in results:
            print(f"추천 카테고리: {results['category_suggestion']}")

    elif "thumbnail_guide" in results:
        print("\n[썸네일 가이드]")
        print(results["thumbnail_guide"])

    elif "suggestions" in results:
        print("\n[제목 최적화 제안]")
        for sug in results.get("suggestions", []):
            print(f"\n[{sug.get('type', '')}] {sug.get('title', '')}")
            print(f"  전략: {sug.get('ctr_strategy', '')}")
        if results.get("best_recommendation"):
            print(f"\n최고 추천 제목: {results['best_recommendation']}")

    elif "description" in results:
        print("\n[영상 설명]")
        print(results["description"])

    elif "analysis" in results:
        print("\n[댓글 분석]")
        print(results["analysis"])

    elif action == "all" or isinstance(results, dict) and "summarize" in results:
        # all 액션 결과
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"\n{'='*40}")
                print(f"[{key.upper()}]")
                for sub_key, sub_val in value.items():
                    if sub_key not in ("action", "language", "input_length"):
                        if isinstance(sub_val, list):
                            print(f"{sub_key}: {', '.join(str(v) for v in sub_val[:10])}")
                        elif isinstance(sub_val, str) and len(sub_val) > 100:
                            print(f"{sub_key}:")
                            print(sub_val[:500] + ("..." if len(sub_val) > 500 else ""))
                        else:
                            print(f"{sub_key}: {sub_val}")

    print("\n" + "=" * 60)


def main() -> int:
    """메인 실행 함수."""
    args = parse_args()
    setup_logging(args.log_level)

    try:
        config = load_config(args.config)
        project_id = args.project or get_project_id(config)
        location = args.location or config.get("vertex_ai", {}).get(
            "location", "asia-northeast3"
        )

        logger.info(f"프로젝트 ID: {project_id}")
        logger.info(f"Gemini 모델: {args.model}")
        logger.info(f"액션: {args.action}")

        # 입력 텍스트 준비
        subtitle_text = ""
        if args.subtitle:
            subtitle_text = load_subtitle_text(args.subtitle)
        elif args.text:
            subtitle_text = args.text

        if not subtitle_text and args.action in [
            "summarize", "generate_tags", "generate_description", "all"
        ]:
            logger.warning(
                "자막 텍스트가 없습니다. --subtitle 또는 --text 옵션을 사용하세요."
            )

        # Gemini 모델 초기화
        model = get_gemini_model(
            model_name=args.model,
            location=location,
            project_id=project_id,
        )

        results = {}

        # 액션 실행
        if args.action == "summarize":
            if not subtitle_text:
                logger.error("--subtitle 또는 --text 옵션이 필요합니다.")
                return 1
            results = summarize_video(
                model=model,
                subtitle_text=subtitle_text,
                language=args.language,
                temperature=args.temperature,
            )

        elif args.action == "generate_tags":
            if not subtitle_text:
                logger.error("--subtitle 또는 --text 옵션이 필요합니다.")
                return 1
            results = generate_tags(
                model=model,
                subtitle_text=subtitle_text,
                title=args.title,
                num_tags=args.num_tags,
                language=args.language,
                temperature=args.temperature,
            )

        elif args.action == "thumbnail_desc":
            if not args.title:
                logger.error("--title 옵션이 필요합니다.")
                return 1
            results = generate_thumbnail_description(
                model=model,
                title=args.title,
                summary=args.summary,
                subtitle_text=subtitle_text or None,
                temperature=args.temperature,
            )

        elif args.action == "optimize_title":
            if not args.title:
                logger.error("--title 옵션이 필요합니다.")
                return 1
            results = optimize_title(
                model=model,
                original_title=args.title,
                subtitle_text=subtitle_text or None,
                temperature=args.temperature,
            )

        elif args.action == "generate_description":
            if not subtitle_text:
                logger.error("--subtitle 또는 --text 옵션이 필요합니다.")
                return 1
            results = generate_video_description(
                model=model,
                subtitle_text=subtitle_text,
                title=args.title,
                language=args.language,
                temperature=args.temperature,
            )

        elif args.action == "analyze_comments":
            if not args.comments:
                logger.error("--comments 옵션이 필요합니다.")
                return 1
            comment_list = [c.strip() for c in args.comments.split(",")]
            results = analyze_comments(
                model=model,
                comments=comment_list,
                title=args.title,
                temperature=args.temperature,
            )

        elif args.action == "all":
            results = run_all_actions(
                model=model,
                subtitle_text=subtitle_text,
                title=args.title,
                language=args.language,
                temperature=args.temperature,
                num_tags=args.num_tags,
            )

        # 결과 출력
        print_results(results)

        # JSON 파일로 저장
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"결과 저장: {args.output}")
            print(f"\n결과 파일: {args.output}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"파일 오류: {e}")
        return 1
    except ValueError as e:
        logger.error(f"설정 오류: {e}")
        return 1
    except Exception as e:
        logger.error(f"생성 중 오류 발생: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
