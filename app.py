# -*- coding: utf-8 -*-
"""
Streamlit YOLO 검사 전용 프로그램

목적
----
VSCode에서 학습한 YOLO best.pt 모델을 GitHub 저장소의 models 폴더에 올리고,
Streamlit Cloud에서 그 모델을 불러와 사용자가 업로드한 이미지를 검사합니다.

권장 GitHub 구조
----------------
repository/
├── app.py
├── requirements.txt
└── models/
    └── best.pt
"""

from __future__ import annotations

import hashlib
import io
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from google import genai
from google.genai import types


# ============================================================
# 1. Streamlit 기본 설정
# ============================================================

st.set_page_config(
    page_title="YOLO 이미지 검사",
    page_icon="🔍",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
TEMP_DIR = APP_DIR / ".streamlit_temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

st.title("🔍 YOLO 이미지 검사 프로그램")
st.caption(
    "GitHub에 저장된 학습 완료 모델을 불러와 업로드한 이미지를 검사합니다."
)


# ============================================================
# 2. 공통 함수
# ============================================================

def discover_local_models(model_dir: Path) -> list[Path]:
    """GitHub 저장소의 models 폴더에서 .pt 모델을 찾습니다."""
    if not model_dir.exists():
        return []

    return sorted(
        path
        for path in model_dir.rglob("*.pt")
        if path.is_file()
    )


def is_github_lfs_pointer(path: Path) -> bool:
    """
    파일이 실제 모델이 아니라 Git LFS 포인터 텍스트인지 확인합니다.
    """
    try:
        if path.stat().st_size > 2048:
            return False

        header = path.read_text(encoding="utf-8", errors="ignore")[:200]
        return "git-lfs.github.com/spec" in header
    except Exception:
        return False


def download_model_from_url(
    model_url: str,
    github_token: str | None = None,
) -> Path:
    """
    GitHub Raw URL 또는 직접 다운로드 가능한 URL에서 모델을 내려받습니다.

    비공개 GitHub 저장소라면 Streamlit Secrets에
    GITHUB_TOKEN을 등록해 사용할 수 있습니다.
    """
    model_url = model_url.strip()

    if not model_url:
        raise ValueError("모델 URL을 입력하세요.")

    url_hash = hashlib.sha256(model_url.encode("utf-8")).hexdigest()[:16]
    destination = TEMP_DIR / f"github_model_{url_hash}.pt"

    if destination.exists() and destination.stat().st_size > 1024:
        return destination

    headers = {}

    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    response = requests.get(
        model_url,
        headers=headers,
        timeout=180,
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    if "text/html" in content_type:
        raise RuntimeError(
            "모델 파일이 아니라 HTML 페이지가 내려왔습니다. "
            "GitHub 파일 화면 주소가 아닌 Raw 주소를 사용하세요."
        )

    destination.write_bytes(response.content)

    if destination.stat().st_size < 1024:
        text = destination.read_text(encoding="utf-8", errors="ignore")

        if "git-lfs.github.com/spec" in text:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                "Git LFS 포인터 파일만 내려받았습니다. "
                "실제 모델 파일을 일반 GitHub 파일로 올리거나 "
                "GitHub Release 자산의 직접 다운로드 주소를 사용하세요."
            )

    return destination


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path_text: str) -> YOLO:
    """YOLO 모델을 한 번만 메모리에 올려 재사용합니다."""
    model_path = Path(model_path_text)

    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾지 못했습니다: {model_path}")

    if is_github_lfs_pointer(model_path):
        raise RuntimeError(
            "현재 파일은 실제 모델이 아니라 Git LFS 포인터입니다."
        )

    return YOLO(str(model_path))


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """PIL 이미지를 YOLO 입력용 RGB numpy 배열로 변환합니다."""
    return np.array(image.convert("RGB"))


def run_prediction(
    model: YOLO,
    images: list[Image.Image],
    confidence: float,
    iou: float,
    image_size: int,
    device: str | int,
) -> list[Any]:
    """여러 이미지를 YOLO로 예측합니다."""
    sources = [pil_to_rgb_array(image) for image in images]

    return model.predict(
        source=sources,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        device=device,
        verbose=False,
        save=False,
    )


def result_to_annotated_image(result: Any) -> Image.Image:
    """
    Ultralytics의 plot 결과(BGR)를 Streamlit 표시용 RGB 이미지로 변환합니다.
    """
    plotted_bgr = result.plot()
    plotted_rgb = plotted_bgr[:, :, ::-1]
    return Image.fromarray(plotted_rgb)


def extract_detection_rows(
    results: list[Any],
    filenames: list[str],
) -> list[dict[str, Any]]:
    """예측 결과를 표 형태로 정리합니다."""
    rows: list[dict[str, Any]] = []

    for image_index, (result, filename) in enumerate(
        zip(results, filenames),
        start=1,
    ):
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            rows.append(
                {
                    "이미지 번호": image_index,
                    "파일명": filename,
                    "검출 번호": "-",
                    "클래스": "검출 없음",
                    "신뢰도": "-",
                    "x1": "-",
                    "y1": "-",
                    "x2": "-",
                    "y2": "-",
                }
            )
            continue

        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()
        coordinates = boxes.xyxy.detach().cpu().numpy()
        names = result.names

        for detection_index, (class_id, confidence, xyxy) in enumerate(
            zip(class_ids, confidences, coordinates),
            start=1,
        ):
            rows.append(
                {
                    "이미지 번호": image_index,
                    "파일명": filename,
                    "검출 번호": detection_index,
                    "클래스": names.get(int(class_id), str(class_id)),
                    "신뢰도": round(float(confidence), 4),
                    "x1": round(float(xyxy[0]), 1),
                    "y1": round(float(xyxy[1]), 1),
                    "x2": round(float(xyxy[2]), 1),
                    "y2": round(float(xyxy[3]), 1),
                }
            )

    return rows


def create_contact_sheet(
    images: list[Image.Image],
    labels: list[str],
    columns: int = 3,
    thumb_width: int = 480,
    thumb_height: int = 360,
    margin: int = 16,
    label_height: int = 32,
) -> Image.Image:
    """여러 검사 결과 이미지를 한 장으로 합칩니다."""
    if not images:
        raise ValueError("모음 이미지로 만들 결과가 없습니다.")

    columns = max(1, columns)
    rows = math.ceil(len(images) / columns)

    cell_width = thumb_width + margin * 2
    cell_height = thumb_height + label_height + margin * 2

    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        (245, 245, 245),
    )

    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (image, label) in enumerate(zip(images, labels)):
        row = index // columns
        column = index % columns

        x0 = column * cell_width + margin
        y0 = row * cell_height + margin

        thumbnail = image.copy().convert("RGB")
        thumbnail.thumbnail((thumb_width, thumb_height))

        paste_x = x0 + (thumb_width - thumbnail.width) // 2
        paste_y = y0 + (thumb_height - thumbnail.height) // 2

        sheet.paste(thumbnail, (paste_x, paste_y))
        draw.text(
            (x0, y0 + thumb_height + 7),
            f"{index + 1}. {label}",
            fill=(20, 20, 20),
            font=font,
        )

    return sheet


def image_to_jpeg_bytes(image: Image.Image, quality: int = 95) -> bytes:
    """PIL 이미지를 JPEG 바이트로 변환합니다."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def normalize_gemini_model_name(model_name: str) -> str:
    """
    models/gemini-... 형태로 입력해도 SDK에서 사용할 수 있도록 정리합니다.
    """
    normalized = model_name.strip()

    if normalized.startswith("models/"):
        normalized = normalized.removeprefix("models/")

    return normalized


def analyze_yolo_result_with_gemini(
    api_key: str,
    model_name: str,
    original_image: Image.Image,
    annotated_image: Image.Image,
    filename: str,
    detection_rows: list[dict[str, Any]],
    confidence_threshold: float,
    user_request: str,
) -> str:
    """
    원본 이미지, YOLO 표시 이미지, 검출 수치를 Gemini에 전달하여
    품질검사 관점의 보조 해석을 생성합니다.
    """
    if not api_key.strip():
        raise ValueError("Gemini API 키를 입력하세요.")

    normalized_model = normalize_gemini_model_name(model_name)

    if not normalized_model:
        raise ValueError("Gemini 모델명을 입력하세요.")

    client = genai.Client(api_key=api_key.strip())

    detected_rows = [
        row
        for row in detection_rows
        if row.get("파일명") == filename
    ]

    detection_summary = []

    for row in detected_rows:
        detection_summary.append(
            {
                "검출 번호": row.get("검출 번호"),
                "클래스": row.get("클래스"),
                "신뢰도": row.get("신뢰도"),
                "좌표": [
                    row.get("x1"),
                    row.get("y1"),
                    row.get("x2"),
                    row.get("y2"),
                ],
            }
        )

    prompt = f"""
당신은 반도체 및 정밀가공품의 비전검사 결과를 검토하는 품질 엔지니어입니다.

아래에는 같은 검사 이미지의 원본과 YOLO 검출 결과 이미지가 제공됩니다.

파일명: {filename}
YOLO 신뢰도 기준: {confidence_threshold:.2f}
YOLO 검출 결과:
{detection_summary}

다음 순서로 한국어로 분석하세요.

1. YOLO 검출 결과 요약
2. 검출된 위치와 형상 설명
3. 이미지에서 육안으로 확인되는 이상 특징
4. 가능한 원인 가설
5. 추가로 확인할 측정·공정 항목
6. 자동검사 결과 사용 시 주의사항

중요:
- YOLO 검출 결과를 사실로 단정하지 말고, 오검출 가능성을 함께 검토하세요.
- 이미지에서 보이지 않는 재료, 공정, 치수 정보를 단정하지 마세요.
- 검출이 없으면 '정상 확정'이라고 하지 말고, 현재 모델과 임계값에서 검출되지 않았다고 표현하세요.
- 불량의 최종 판정은 사내 검사 기준과 측정 결과를 함께 확인해야 합니다.

사용자 추가 요청:
{user_request.strip() or "추가 요청 없음"}
"""

    original_bytes = image_to_jpeg_bytes(
        original_image,
        quality=92,
    )
    annotated_bytes = image_to_jpeg_bytes(
        annotated_image,
        quality=92,
    )

    try:
        response = client.models.generate_content(
            model=normalized_model,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=original_bytes,
                    mime_type="image/jpeg",
                ),
                types.Part.from_bytes(
                    data=annotated_bytes,
                    mime_type="image/jpeg",
                ),
            ],
        )

    except Exception as error:
        error_text = str(error)

        if "404" in error_text or "NOT_FOUND" in error_text:
            raise RuntimeError(
                "선택한 Gemini 모델을 현재 API 키에서 사용할 수 없습니다. "
                "왼쪽 모델 입력을 'gemini-3.1-flash-lite'로 변경하거나 "
                "Google AI Studio에서 사용 가능한 모델명을 확인하세요.\n\n"
                f"현재 요청 모델: {normalized_model}\n"
                f"원본 오류: {error_text}"
            ) from error

        raise

    if not response.text:
        raise RuntimeError(
            "Gemini에서 분석 텍스트를 받지 못했습니다."
        )

    return response.text


def analyze_all_yolo_results_with_gemini(
    api_key: str,
    model_name: str,
    original_images: list[Image.Image],
    annotated_images: list[Image.Image],
    filenames: list[str],
    detection_rows: list[dict[str, Any]],
    confidence_threshold: float,
    user_request: str,
) -> str:
    """
    전체 검사 이미지와 전체 검출 데이터를 Gemini에 전달하여
    데이터셋 전체 기준의 종합 분석을 생성합니다.

    이미지가 너무 많으면 API 입력 부담을 줄이기 위해
    최대 12장까지 대표 이미지를 전달하고,
    모든 이미지의 검출 수치는 표 형태로 함께 전달합니다.
    """
    if not api_key.strip():
        raise ValueError("Gemini API 키를 입력하세요.")

    normalized_model = normalize_gemini_model_name(model_name)

    if not normalized_model:
        raise ValueError("Gemini 모델명을 입력하세요.")

    client = genai.Client(api_key=api_key.strip())

    # 전체 파일별 검출 요약 생성
    file_summaries = []

    for filename in filenames:
        rows = [
            row
            for row in detection_rows
            if row.get("파일명") == filename
        ]

        valid_rows = [
            row
            for row in rows
            if row.get("클래스") != "검출 없음"
        ]

        confidences = [
            float(row["신뢰도"])
            for row in valid_rows
            if isinstance(row.get("신뢰도"), (int, float))
        ]

        file_summaries.append(
            {
                "파일명": filename,
                "검출 수": len(valid_rows),
                "검출 클래스": [
                    row.get("클래스")
                    for row in valid_rows
                ],
                "최대 신뢰도": (
                    round(max(confidences), 4)
                    if confidences
                    else None
                ),
                "평균 신뢰도": (
                    round(sum(confidences) / len(confidences), 4)
                    if confidences
                    else None
                ),
            }
        )

    total_detected_images = sum(
        1
        for summary in file_summaries
        if summary["검출 수"] > 0
    )

    total_boxes = sum(
        summary["검출 수"]
        for summary in file_summaries
    )

    class_counts: dict[str, int] = {}

    for row in detection_rows:
        class_name = row.get("클래스")

        if class_name and class_name != "검출 없음":
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    prompt = f"""
당신은 반도체 및 정밀가공품의 비전검사 결과를 검토하는 품질 엔지니어입니다.

이번 요청은 개별 이미지 1장이 아니라 전체 검사 데이터 기준의 종합 분석입니다.

전체 검사 이미지 수: {len(filenames)}
검출이 발생한 이미지 수: {total_detected_images}
전체 검출 박스 수: {total_boxes}
YOLO 신뢰도 기준: {confidence_threshold:.2f}
클래스별 검출 수: {class_counts}

파일별 검사 요약:
{file_summaries}

전체 검출 상세 데이터:
{detection_rows}

반드시 다음 순서로 한국어로 분석하세요.

1. 전체 검사 결과 요약
2. 검출 빈도가 높은 클래스와 이미지 유형
3. 신뢰도가 높은 검출과 낮은 검출의 분포
4. 반복적으로 나타나는 위치 또는 형상 경향
5. 오검출 가능성이 높은 사례
6. 추가로 확인할 측정·공정 항목
7. 모델 개선을 위한 데이터 수집 및 재학습 제안
8. 최종 품질 판단 시 주의사항

중요:
- 검출이 없는 이미지를 정상 확정이라고 표현하지 마세요.
- YOLO 검출 결과와 Gemini의 이미지 해석을 최종 품질 판정으로 단정하지 마세요.
- 전체 데이터에서 반복되는 경향과 예외 사례를 구분해 설명하세요.
- 이미지에서 보이지 않는 재료, 치수, 공정 조건은 사실처럼 단정하지 마세요.

사용자 추가 요청:
{user_request.strip() or "추가 요청 없음"}
"""

    # 전체 이미지를 모두 보내면 입력 크기가 지나치게 커질 수 있으므로
    # 최대 12장의 대표 이미지를 균등 간격으로 선택합니다.
    max_images_for_ai = 12

    if len(filenames) <= max_images_for_ai:
        selected_indices = list(range(len(filenames)))
    else:
        selected_indices = sorted(
            {
                round(
                    index * (len(filenames) - 1)
                    / (max_images_for_ai - 1)
                )
                for index in range(max_images_for_ai)
            }
        )

    contents: list[Any] = [prompt]

    for index in selected_indices:
        contents.append(
            f"대표 이미지 {index + 1}: {filenames[index]} - 원본"
        )
        contents.append(
            types.Part.from_bytes(
                data=image_to_jpeg_bytes(
                    original_images[index],
                    quality=88,
                ),
                mime_type="image/jpeg",
            )
        )

        contents.append(
            f"대표 이미지 {index + 1}: {filenames[index]} - YOLO 결과"
        )
        contents.append(
            types.Part.from_bytes(
                data=image_to_jpeg_bytes(
                    annotated_images[index],
                    quality=88,
                ),
                mime_type="image/jpeg",
            )
        )

    try:
        response = client.models.generate_content(
            model=normalized_model,
            contents=contents,
        )

    except Exception as error:
        error_text = str(error)

        if "404" in error_text or "NOT_FOUND" in error_text:
            raise RuntimeError(
                "선택한 Gemini 모델을 현재 API 키에서 사용할 수 없습니다. "
                "왼쪽 모델 입력을 'gemini-3.1-flash-lite'로 변경하거나 "
                "Google AI Studio에서 사용 가능한 모델명을 확인하세요.\n\n"
                f"현재 요청 모델: {normalized_model}\n"
                f"원본 오류: {error_text}"
            ) from error

        raise

    if not response.text:
        raise RuntimeError(
            "Gemini에서 전체 데이터 분석 텍스트를 받지 못했습니다."
        )

    return response.text


def build_result_zip(
    annotated_images: list[Image.Image],
    filenames: list[str],
    detection_rows: list[dict[str, Any]],
    contact_sheet: Image.Image | None,
) -> bytes:
    """개별 결과 이미지, CSV, 모음 이미지를 ZIP으로 묶습니다."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:
        for index, (image, filename) in enumerate(
            zip(annotated_images, filenames),
            start=1,
        ):
            output_name = f"{index:03d}_{Path(filename).stem}_result.jpg"
            zip_file.writestr(
                output_name,
                image_to_jpeg_bytes(image),
            )

        dataframe = pd.DataFrame(detection_rows)
        zip_file.writestr(
            "detection_results.csv",
            dataframe.to_csv(index=False).encode("utf-8-sig"),
        )

        if contact_sheet is not None:
            zip_file.writestr(
                "prediction_contact_sheet.jpg",
                image_to_jpeg_bytes(contact_sheet),
            )

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ============================================================
# 3. 모델 설정
# ============================================================

st.sidebar.header("모델 설정")

model_source = st.sidebar.radio(
    "모델 불러오기 방식",
    [
        "GitHub 저장소의 models 폴더",
        "GitHub Raw URL",
    ],
)

selected_model_path: Path | None = None

if model_source == "GitHub 저장소의 models 폴더":
    local_models = discover_local_models(MODEL_DIR)

    if not local_models:
        st.sidebar.error(
            "models 폴더에서 .pt 모델을 찾지 못했습니다."
        )
        st.sidebar.code(
            "repository/\n"
            "├── app.py\n"
            "├── requirements.txt\n"
            "└── models/\n"
            "    └── best.pt"
        )
    else:
        model_labels = [
            str(path.relative_to(APP_DIR))
            for path in local_models
        ]

        selected_label = st.sidebar.selectbox(
            "사용할 모델",
            model_labels,
        )

        selected_model_path = APP_DIR / selected_label

else:
    model_url = st.sidebar.text_input(
        "모델 Raw URL",
        placeholder=(
            "https://raw.githubusercontent.com/"
            "사용자/저장소/main/models/best.pt"
        ),
    )

    try:
        github_token = st.secrets.get("GITHUB_TOKEN", None)
    except Exception:
        github_token = None

    if model_url:
        try:
            with st.sidebar.status(
                "GitHub 모델 확인 중...",
                expanded=False,
            ) as status:
                selected_model_path = download_model_from_url(
                    model_url=model_url,
                    github_token=github_token,
                )
                status.update(
                    label="GitHub 모델 준비 완료",
                    state="complete",
                )
        except Exception as error:
            st.sidebar.error(str(error))

st.sidebar.divider()

confidence = st.sidebar.slider(
    "신뢰도 기준",
    min_value=0.01,
    max_value=1.00,
    value=0.25,
    step=0.01,
    help=(
        "낮추면 더 많은 후보를 검출하고, "
        "높이면 확실한 검출만 표시합니다."
    ),
)

iou_threshold = st.sidebar.slider(
    "중복 박스 제거 기준(IoU)",
    min_value=0.10,
    max_value=0.95,
    value=0.45,
    step=0.05,
)

image_size = st.sidebar.selectbox(
    "검사 입력 크기",
    [320, 416, 640, 800, 1024],
    index=2,
    help=(
        "작은 결함은 큰 입력 크기가 유리할 수 있지만 "
        "검사 시간이 증가합니다."
    ),
)

device = 0 if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    st.sidebar.success(
        f"GPU 사용: {torch.cuda.get_device_name(0)}"
    )
else:
    st.sidebar.info("Streamlit Cloud CPU로 검사합니다.")

st.sidebar.divider()
st.sidebar.header("Gemini 결과 분석")

gemini_api_key = st.sidebar.text_input(
    "Gemini API 키",
    type="password",
    help=(
        "현재 Streamlit 세션에서만 사용합니다. "
        "코드와 GitHub 저장소에는 저장하지 않습니다."
    ),
)

gemini_model_name = st.sidebar.text_input(
    "Gemini 모델명",
    value="gemini-3.1-flash-lite",
    help=(
        "Gemini 3.1 Flash-Lite의 API 모델명은 gemini-3.1-flash-lite 입니다."
    ),
)

gemini_user_request = st.sidebar.text_area(
    "추가 분석 요청",
    value=(
        "검출 위치와 신뢰도를 품질팀 관점에서 해석하고 "
        "추가 확인이 필요한 항목을 알려주세요."
    ),
)


# ============================================================
# 4. 이미지 업로드 및 검사
# ============================================================

st.subheader("1. 검사 이미지 등록")

uploaded_files = st.file_uploader(
    "검사할 이미지를 한 장 또는 여러 장 선택하세요.",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    accept_multiple_files=True,
)

images: list[Image.Image] = []
filenames: list[str] = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        images.append(image)
        filenames.append(uploaded_file.name)

    st.success(f"{len(images)}장의 이미지를 등록했습니다.")

    preview_columns = st.columns(min(4, len(images)))

    for index, image in enumerate(images[:4]):
        with preview_columns[index]:
            st.image(
                image,
                caption=filenames[index],
                use_container_width=True,
            )

    if len(images) > 4:
        st.caption(f"나머지 {len(images) - 4}장은 검사 결과에서 확인할 수 있습니다.")


st.subheader("2. YOLO 검사 실행")

if selected_model_path is not None:
    model_size_mb = selected_model_path.stat().st_size / 1024**2

    st.write("**선택 모델**")
    st.code(str(selected_model_path))
    st.caption(f"모델 크기: {model_size_mb:.1f} MB")
else:
    st.warning("먼저 왼쪽 모델 설정에서 사용할 모델을 선택하세요.")

run_button = st.button(
    "이미지 검사 시작",
    type="primary",
    disabled=(selected_model_path is None or not images),
    use_container_width=True,
)

if run_button:
    try:
        with st.spinner("YOLO 모델을 불러오고 이미지를 검사하고 있습니다."):
            model = load_yolo_model(str(selected_model_path))

            results = run_prediction(
                model=model,
                images=images,
                confidence=confidence,
                iou=iou_threshold,
                image_size=image_size,
                device=device,
            )

            annotated_images = [
                result_to_annotated_image(result)
                for result in results
            ]

            detection_rows = extract_detection_rows(
                results=results,
                filenames=filenames,
            )

            st.session_state["inference_results"] = {
                "original_images": images,
                "annotated_images": annotated_images,
                "filenames": filenames,
                "detection_rows": detection_rows,
                "model_path": str(selected_model_path),
                "confidence_threshold": float(confidence),
            }

            # 새 검사 결과가 생성되면 이전 Gemini 해석은 초기화합니다.
            st.session_state["gemini_analysis"] = None

        st.success("이미지 검사가 완료되었습니다.")

    except Exception as error:
        st.exception(error)


# ============================================================
# 5. 검사 결과
# ============================================================

saved_results = st.session_state.get("inference_results")

if saved_results:
    original_images = saved_results["original_images"]
    annotated_images = saved_results["annotated_images"]
    result_filenames = saved_results["filenames"]
    detection_rows = saved_results["detection_rows"]

    st.divider()
    st.subheader("3. 검사 결과")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    detected_image_count = len(
        {
            row["파일명"]
            for row in detection_rows
            if row["클래스"] != "검출 없음"
        }
    )

    total_detection_count = sum(
        1
        for row in detection_rows
        if row["클래스"] != "검출 없음"
    )

    summary_col1.metric(
        "검사 이미지",
        len(annotated_images),
    )
    summary_col2.metric(
        "검출 이미지",
        detected_image_count,
    )
    summary_col3.metric(
        "전체 검출 박스",
        total_detection_count,
    )

    result_tabs = st.tabs(
        [
            "개별 결과",
            "전체 모음",
            "검출 데이터",
            "Gemini 분석",
            "결과 다운로드",
        ]
    )

    with result_tabs[0]:
        selected_index = st.number_input(
            "확인할 이미지 번호",
            min_value=1,
            max_value=len(annotated_images),
            value=1,
            step=1,
        ) - 1

        st.image(
            annotated_images[selected_index],
            caption=(
                f"{selected_index + 1} / {len(annotated_images)} "
                f"— {result_filenames[selected_index]}"
            ),
            use_container_width=True,
        )

        selected_rows = [
            row
            for row in detection_rows
            if row["파일명"] == result_filenames[selected_index]
        ]

        st.dataframe(
            pd.DataFrame(selected_rows),
            use_container_width=True,
            hide_index=True,
        )

    with result_tabs[1]:
        contact_columns = st.slider(
            "한 줄에 표시할 이미지 수",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
        )

        contact_sheet = create_contact_sheet(
            images=annotated_images,
            labels=result_filenames,
            columns=contact_columns,
        )

        st.image(
            contact_sheet,
            caption="전체 검사 결과 모음",
            use_container_width=True,
        )

    with result_tabs[2]:
        dataframe = pd.DataFrame(detection_rows)

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = dataframe.to_csv(
            index=False,
        ).encode("utf-8-sig")

        st.download_button(
            "검출 결과 CSV 다운로드",
            data=csv_bytes,
            file_name="detection_results.csv",
            mime="text/csv",
        )

    with result_tabs[3]:
        st.markdown("### Gemini API 검사 결과 분석")

        analysis_mode = st.radio(
            "분석 범위",
            [
                "개별 이미지 기준",
                "전체 데이터 기준",
            ],
            horizontal=True,
        )

        if analysis_mode == "개별 이미지 기준":
            st.info(
                "선택한 이미지 1장의 원본, YOLO 검출 결과, 클래스·신뢰도·좌표를 "
                "Gemini에 전달해 보조 분석을 생성합니다."
            )

            gemini_target_index = st.selectbox(
                "Gemini로 분석할 이미지",
                options=list(range(len(result_filenames))),
                format_func=lambda index: (
                    f"{index + 1}. {result_filenames[index]}"
                ),
                key="gemini_target_index",
            )

            gemini_col1, gemini_col2 = st.columns(2)

            with gemini_col1:
                st.image(
                    original_images[gemini_target_index],
                    caption="원본 이미지",
                    use_container_width=True,
                )

            with gemini_col2:
                st.image(
                    annotated_images[gemini_target_index],
                    caption="YOLO 검출 결과",
                    use_container_width=True,
                )

            current_filename = result_filenames[gemini_target_index]

            current_rows = [
                row
                for row in detection_rows
                if row["파일명"] == current_filename
            ]

            st.dataframe(
                pd.DataFrame(current_rows),
                use_container_width=True,
                hide_index=True,
            )

            if st.button(
                "선택한 결과를 Gemini로 분석",
                type="primary",
                disabled=not gemini_api_key.strip(),
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Gemini가 선택 이미지와 YOLO 검사 결과를 분석하고 있습니다."
                    ):
                        analysis_text = analyze_yolo_result_with_gemini(
                            api_key=gemini_api_key,
                            model_name=gemini_model_name,
                            original_image=original_images[gemini_target_index],
                            annotated_image=annotated_images[gemini_target_index],
                            filename=current_filename,
                            detection_rows=detection_rows,
                            confidence_threshold=float(
                                saved_results.get(
                                    "confidence_threshold",
                                    confidence,
                                )
                            ),
                            user_request=gemini_user_request,
                        )

                    st.session_state["gemini_analysis"] = {
                        "mode": "개별 이미지",
                        "filename": current_filename,
                        "text": analysis_text,
                    }

                    st.success("Gemini 개별 분석이 완료되었습니다.")

                except Exception as error:
                    st.exception(error)

        else:
            st.info(
                "전체 이미지의 검출 수, 클래스, 신뢰도, 좌표를 모두 전달하고, "
                "대표 원본·결과 이미지를 함께 보내 전체 경향을 분석합니다."
            )

            total_detected_images = len(
                {
                    row["파일명"]
                    for row in detection_rows
                    if row["클래스"] != "검출 없음"
                }
            )

            total_boxes = sum(
                1
                for row in detection_rows
                if row["클래스"] != "검출 없음"
            )

            all_col1, all_col2, all_col3 = st.columns(3)

            all_col1.metric(
                "전체 이미지",
                len(result_filenames),
            )
            all_col2.metric(
                "검출 이미지",
                total_detected_images,
            )
            all_col3.metric(
                "전체 박스",
                total_boxes,
            )

            st.dataframe(
                pd.DataFrame(detection_rows),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "이미지가 12장을 초과하면 전체 수치 데이터는 모두 전달하고, "
                "대표 이미지는 최대 12장을 균등하게 선택해 Gemini에 전달합니다."
            )

            if st.button(
                "전체 데이터를 Gemini로 종합 분석",
                type="primary",
                disabled=not gemini_api_key.strip(),
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Gemini가 전체 검사 데이터와 대표 이미지를 종합 분석하고 있습니다."
                    ):
                        analysis_text = analyze_all_yolo_results_with_gemini(
                            api_key=gemini_api_key,
                            model_name=gemini_model_name,
                            original_images=original_images,
                            annotated_images=annotated_images,
                            filenames=result_filenames,
                            detection_rows=detection_rows,
                            confidence_threshold=float(
                                saved_results.get(
                                    "confidence_threshold",
                                    confidence,
                                )
                            ),
                            user_request=gemini_user_request,
                        )

                    st.session_state["gemini_analysis"] = {
                        "mode": "전체 데이터",
                        "filename": "전체 검사 데이터",
                        "text": analysis_text,
                    }

                    st.success("Gemini 전체 데이터 분석이 완료되었습니다.")

                except Exception as error:
                    st.exception(error)

        gemini_analysis = st.session_state.get(
            "gemini_analysis"
        )

        if gemini_analysis:
            st.markdown(
                f"#### 분석 범위: {gemini_analysis.get('mode', '분석')}"
            )
            st.markdown(
                f"**분석 대상:** {gemini_analysis['filename']}"
            )
            st.markdown(gemini_analysis["text"])

    with result_tabs[4]:
        contact_sheet_for_zip = create_contact_sheet(
            images=annotated_images,
            labels=result_filenames,
            columns=3,
        )

        result_zip = build_result_zip(
            annotated_images=annotated_images,
            filenames=result_filenames,
            detection_rows=detection_rows,
            contact_sheet=contact_sheet_for_zip,
        )

        st.download_button(
            "전체 검사 결과 ZIP 다운로드",
            data=result_zip,
            file_name="yolo_inspection_results.zip",
            mime="application/zip",
            use_container_width=True,
        )

        st.caption(
            "ZIP에는 개별 결과 이미지, 전체 모음 이미지, 검출 결과 CSV가 포함됩니다."
        )


# ============================================================
# 6. 사용 안내
# ============================================================

with st.expander("GitHub 모델 연결 방법"):
    st.markdown(
        """
        ### 가장 간단한 방법

        VSCode에서 학습 후 생성된 `best.pt`를 GitHub 저장소의
        `models` 폴더에 업로드합니다.

        ```text
        repository
        ├── app.py
        ├── requirements.txt
        └── models
            └── best.pt
        ```

        앱을 다시 열면 왼쪽의 **사용할 모델** 목록에 자동으로 나타납니다.

        ### 모델 파일이 여러 개인 경우

        ```text
        models
        ├── screw_v1.pt
        ├── screw_v2.pt
        └── screw_v3.pt
        ```

        형태로 올리면 Streamlit 화면에서 모델을 선택할 수 있습니다.

        ### Gemini 분석

        왼쪽 사이드바에 Gemini API 키와 모델명을 입력하면,
        개별 이미지 또는 전체 검사 데이터 기준으로 원본 이미지와
        YOLO 검출 결과를 함께 분석할 수 있습니다.

        기본 모델명:

        ```text
        gemini-3.1-flash-lite
        ```

        API 키는 현재 실행 세션에서만 사용되며 GitHub에 저장되지 않습니다.

        ### 주의

        일반 GitHub 저장소는 단일 파일 100MB 제한이 있습니다.
        작은 YOLO 모델은 보통 저장할 수 있지만, 큰 모델은
        GitHub Release나 별도 모델 저장소 사용이 필요할 수 있습니다.

        Gemini의 설명은 보조 분석이며 최종 품질 판정을 대신하지 않습니다.
        """
    )
