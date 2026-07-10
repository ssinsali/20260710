# -*- coding: utf-8 -*-
"""
GitHub 모델 연동 Patch 이상 탐지 + Gemini 해석 Streamlit 앱

앱 동작
-------
1. GitHub 저장소의 models/patch_anomaly 폴더에서 정상 특징 모델을 자동 검색
2. 사용자가 검사 이미지를 드래그앤드롭
3. Patch 기반 최근접 이웃 로직으로 이상 점수와 Heatmap 생성
4. 좌측에 입력한 Gemini API 키로 선택 결과를 보조 해석
5. 결과 이미지와 CSV를 ZIP으로 다운로드

GitHub 권장 구조
---------------
repository/
├── app.py
├── requirements.txt
└── models/
    └── patch_anomaly/
        ├── screw_memory.npy
        └── screw_settings.json

중요
----
- C:\\VisionAI 같은 사용자 PC 경로를 사용하지 않습니다.
- Streamlit Cloud가 복제한 GitHub 저장소 내부 파일만 사용합니다.
- Gemini 해석은 자동 검사 결과를 설명하는 보조 기능이며 최종 품질 판정을 대신하지 않습니다.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. Streamlit 화면 설정
# ============================================================

st.set_page_config(
    page_title="Patch 이상 검사",
    page_icon="🔬",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_ROOT = APP_DIR / "models" / "patch_anomaly"

st.title("🔬 Patch 기반 이상 검사")
st.caption(
    "GitHub에 저장된 정상 특징 모델로 업로드 이미지를 검사하고 Gemini로 결과를 해석합니다."
)


# ============================================================
# 2. 업로드 영역 디자인
# ============================================================

st.markdown(
    """
    <style>
    [data-testid="stFileUploaderDropzone"] {
        min-height: 210px;
        border: 2px dashed #5f7fff;
        border-radius: 16px;
        background-color: rgba(95, 127, 255, 0.06);
        display: flex;
        align-items: center;
        justify-content: center;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #315cff;
        background-color: rgba(95, 127, 255, 0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. GitHub 저장 모델 검색 및 불러오기
# ============================================================

def discover_memory_models(model_root: Path) -> list[dict[str, Path | str]]:
    """
    GitHub 저장소에서 *_memory.npy와 대응하는 *_settings.json을 찾습니다.

    예:
    screw_memory.npy
    screw_settings.json
    """
    if not model_root.exists():
        return []

    models: list[dict[str, Path | str]] = []

    for memory_path in sorted(model_root.rglob("*_memory.npy")):
        prefix = memory_path.name.removesuffix("_memory.npy")
        settings_path = memory_path.with_name(f"{prefix}_settings.json")

        if settings_path.exists():
            models.append(
                {
                    "name": prefix,
                    "memory_path": memory_path,
                    "settings_path": settings_path,
                }
            )

    return models


@st.cache_resource(show_spinner=False)
def load_memory_and_nearest(
    memory_path_text: str,
    settings_path_text: str,
    memory_modified_time: float,
) -> tuple[np.ndarray, NearestNeighbors, dict[str, Any]]:
    """
    정상 특징 모델을 불러와 최근접 이웃 검색기를 구성합니다.

    memory_modified_time은 GitHub 모델 파일이 변경됐을 때
    Streamlit 캐시가 자동으로 갱신되게 하기 위한 값입니다.
    """
    del memory_modified_time

    memory_path = Path(memory_path_text)
    settings_path = Path(settings_path_text)

    memory = np.load(memory_path, allow_pickle=False)

    if memory.ndim != 2 or memory.shape[1] != 3:
        raise ValueError(
            "정상 특징 파일 형식이 올바르지 않습니다. "
            "예상 형식은 (전체 패치 수, 3)입니다."
        )

    settings = json.loads(
        settings_path.read_text(encoding="utf-8")
    )

    required_keys = {
        "image_width",
        "image_height",
        "patch_size",
        "canny_low",
        "canny_high",
    }

    missing_keys = required_keys - set(settings)

    if missing_keys:
        raise ValueError(
            "설정 파일에서 필요한 항목을 찾지 못했습니다: "
            + ", ".join(sorted(missing_keys))
        )

    nearest = NearestNeighbors(
        n_neighbors=1,
        algorithm="auto",
        metric="euclidean",
    )
    nearest.fit(memory)

    return memory, nearest, settings


# ============================================================
# 4. 이미지 처리와 Patch 특징 추출
# ============================================================

def uploaded_image_to_gray(
    uploaded_file: Any,
    image_size: tuple[int, int],
) -> np.ndarray:
    """업로드 이미지를 지정된 크기의 0~1 흑백 영상으로 변환합니다."""
    file_bytes = np.asarray(
        bytearray(uploaded_file.getvalue()),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"이미지를 읽지 못했습니다: {uploaded_file.name}"
        )

    image = cv2.resize(
        image,
        image_size,
        interpolation=cv2.INTER_AREA,
    )

    return image.astype(np.float32) / 255.0


def patch_features(
    image: np.ndarray,
    patch_size: int,
    canny_low: int,
    canny_high: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    각 패치에서 평균 밝기, 표준편차, 에지 비율을 추출합니다.
    """
    if patch_size <= 0:
        raise ValueError("패치 크기는 1 이상이어야 합니다.")

    if (
        patch_size > image.shape[0]
        or patch_size > image.shape[1]
    ):
        raise ValueError(
            f"패치 크기 {patch_size}가 이미지 크기 {image.shape}보다 큽니다."
        )

    features: list[list[float]] = []
    locations: list[tuple[int, int]] = []

    for y in range(
        0,
        image.shape[0] - patch_size + 1,
        patch_size,
    ):
        for x in range(
            0,
            image.shape[1] - patch_size + 1,
            patch_size,
        ):
            roi = image[
                y:y + patch_size,
                x:x + patch_size,
            ]

            roi_uint8 = (roi * 255).astype(np.uint8)

            edges = cv2.Canny(
                roi_uint8,
                canny_low,
                canny_high,
            )

            features.append(
                [
                    float(roi.mean()),
                    float(roi.std()),
                    float(edges.mean() / 255.0),
                ]
            )
            locations.append((x, y))

    return np.asarray(features, dtype=np.float32), locations


def inspect_image(
    image: np.ndarray,
    nearest: NearestNeighbors,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """한 장의 이미지에서 이상 점수와 Heatmap을 계산합니다."""
    patch_size = int(settings["patch_size"])

    features, locations = patch_features(
        image=image,
        patch_size=patch_size,
        canny_low=int(settings["canny_low"]),
        canny_high=int(settings["canny_high"]),
    )

    distances, _ = nearest.kneighbors(features)
    patch_scores = distances.ravel()

    raw_heatmap = np.zeros_like(
        image,
        dtype=np.float32,
    )

    for score, (x, y) in zip(
        patch_scores,
        locations,
    ):
        raw_heatmap[
            y:y + patch_size,
            x:x + patch_size,
        ] = float(score)

    normalized_heatmap = cv2.normalize(
        raw_heatmap,
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    max_index = int(np.argmax(patch_scores))
    max_x, max_y = locations[max_index]

    return {
        "image": image,
        "heatmap": normalized_heatmap,
        "raw_heatmap": raw_heatmap,
        "patch_scores": patch_scores,
        "max_score": float(patch_scores.max()),
        "mean_score": float(patch_scores.mean()),
        "max_location": (
            max_x,
            max_y,
            max_x + patch_size,
            max_y + patch_size,
        ),
    }


# ============================================================
# 5. 결과 이미지 생성
# ============================================================

def apply_heatmap(
    gray_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.50,
) -> Image.Image:
    """흑백 이미지 위에 Jet Heatmap을 겹칩니다."""
    gray_uint8 = np.clip(
        gray_image * 255,
        0,
        255,
    ).astype(np.uint8)

    gray_bgr = cv2.cvtColor(
        gray_uint8,
        cv2.COLOR_GRAY2BGR,
    )

    heat_uint8 = np.clip(
        heatmap * 255,
        0,
        255,
    ).astype(np.uint8)

    heat_bgr = cv2.applyColorMap(
        heat_uint8,
        cv2.COLORMAP_JET,
    )

    overlay_bgr = cv2.addWeighted(
        gray_bgr,
        1.0 - alpha,
        heat_bgr,
        alpha,
        0,
    )

    overlay_rgb = cv2.cvtColor(
        overlay_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(overlay_rgb)


def gray_to_pil(gray_image: np.ndarray) -> Image.Image:
    """0~1 흑백 배열을 PIL RGB 이미지로 변환합니다."""
    gray_uint8 = np.clip(
        gray_image * 255,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(gray_uint8).convert("RGB")


def create_result_panel(
    original: Image.Image,
    overlay: Image.Image,
    filename: str,
    max_score: float,
    threshold: float,
    judgment: str,
) -> Image.Image:
    """원본과 Heatmap 결과를 한 장으로 합칩니다."""
    width = max(original.width, overlay.width)
    image_height = max(original.height, overlay.height)
    header_height = 60

    panel = Image.new(
        "RGB",
        (width * 2, image_height + header_height),
        (245, 245, 245),
    )

    panel.paste(original.resize((width, image_height)), (0, header_height))
    panel.paste(overlay.resize((width, image_height)), (width, header_height))

    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    draw.text(
        (10, 8),
        f"Original: {filename}",
        fill=(20, 20, 20),
        font=font,
    )

    draw.text(
        (width + 10, 8),
        (
            f"{judgment} | score={max_score:.6f} "
            f"| threshold={threshold:.6f}"
        ),
        fill=(20, 20, 20),
        font=font,
    )

    return panel


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_contact_sheet(
    records: list[dict[str, Any]],
    columns: int = 3,
) -> Image.Image:
    """여러 검사 결과를 한 장의 격자 이미지로 합칩니다."""
    if not records:
        raise ValueError("모음 이미지로 만들 결과가 없습니다.")

    columns = max(1, columns)
    rows = math.ceil(len(records) / columns)

    thumb_width = 520
    thumb_height = 280
    margin = 14
    label_height = 36

    cell_width = thumb_width + margin * 2
    cell_height = thumb_height + label_height + margin * 2

    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        (245, 245, 245),
    )

    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, record in enumerate(records):
        row = index // columns
        column = index % columns

        x0 = column * cell_width + margin
        y0 = row * cell_height + margin

        thumbnail = record["result_panel"].copy()
        thumbnail.thumbnail((thumb_width, thumb_height))

        paste_x = x0 + (thumb_width - thumbnail.width) // 2
        paste_y = y0 + (thumb_height - thumbnail.height) // 2

        sheet.paste(thumbnail, (paste_x, paste_y))

        draw.text(
            (x0, y0 + thumb_height + 8),
            (
                f"{index + 1}. {record['filename']} | "
                f"{record['judgment']} | "
                f"{record['max_score']:.4f}"
            ),
            fill=(20, 20, 20),
            font=font,
        )

    return sheet


# ============================================================
# 6. Gemini 이미지 해석
# ============================================================

def analyze_with_gemini(
    api_key: str,
    model_name: str,
    record: dict[str, Any],
    additional_request: str,
) -> str:
    """
    Gemini에 결과 이미지와 검사 수치를 전달하여 보조 해석을 생성합니다.
    """
    if not api_key.strip():
        raise ValueError("Gemini API 키를 입력하세요.")

    client = genai.Client(
        api_key=api_key.strip(),
    )

    result_png = image_to_png_bytes(
        record["result_panel"]
    )

    prompt = f"""
당신은 반도체 및 정밀가공품의 비전 검사 결과를 검토하는 품질 엔지니어입니다.

다음 Patch 기반 이상 탐지 결과를 해석하세요.

파일명: {record['filename']}
알고리즘: 정상 이미지 Patch 특징 기반 최근접 이웃 이상 탐지
특징: 평균 밝기, 밝기 표준편차, Canny 에지 비율
최대 이상 점수: {record['max_score']:.6f}
평균 이상 점수: {record['mean_score']:.6f}
판정 임계값: {record['threshold']:.6f}
자동 판정: {record['judgment']}
최대 이상 영역 좌표(x1, y1, x2, y2): {record['max_location']}

아래 순서로 한국어로 설명하세요.

1. 자동 판정 요약
2. Heatmap에서 이상 반응이 집중된 위치
3. 원본과 Heatmap을 함께 봤을 때 관찰되는 특징
4. 가능한 원인 가설
5. 추가로 확인할 측정 또는 공정 항목
6. 최종 판정 시 주의사항

중요:
- 이미지와 점수만으로 실제 불량 종류나 원인을 확정하지 마세요.
- 화면으로 확인할 수 없는 재료 특성이나 공정 조건을 사실처럼 단정하지 마세요.
- Heatmap은 정상 특징과 다른 위치이지 반드시 실제 결함 위치라는 뜻은 아닙니다.

사용자 추가 요청:
{additional_request.strip() or "추가 요청 없음"}
"""

    response = client.models.generate_content(
        model=model_name,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=result_png,
                mime_type="image/png",
            ),
        ],
    )

    if not response.text:
        raise RuntimeError(
            "Gemini에서 텍스트 응답을 받지 못했습니다."
        )

    return response.text


# ============================================================
# 7. 결과 ZIP 생성
# ============================================================

def build_result_zip(
    records: list[dict[str, Any]],
) -> bytes:
    """결과 이미지와 CSV를 ZIP으로 묶습니다."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:
        csv_rows: list[dict[str, Any]] = []

        for index, record in enumerate(
            records,
            start=1,
        ):
            result_name = (
                f"{index:03d}_"
                f"{Path(record['filename']).stem}_result.png"
            )

            zip_file.writestr(
                result_name,
                image_to_png_bytes(
                    record["result_panel"]
                ),
            )

            csv_rows.append(
                {
                    "파일명": record["filename"],
                    "최대 이상 점수": record["max_score"],
                    "평균 이상 점수": record["mean_score"],
                    "임계값": record["threshold"],
                    "판정": record["judgment"],
                    "최대 이상 영역": record["max_location"],
                }
            )

        dataframe = pd.DataFrame(csv_rows)

        zip_file.writestr(
            "anomaly_results.csv",
            dataframe.to_csv(
                index=False,
            ).encode("utf-8-sig"),
        )

        contact_sheet = create_contact_sheet(
            records=records,
            columns=3,
        )

        zip_file.writestr(
            "prediction_contact_sheet.png",
            image_to_png_bytes(contact_sheet),
        )

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ============================================================
# 8. 세션 상태
# ============================================================

if "inspection_records" not in st.session_state:
    st.session_state.inspection_records = []

if "gemini_result" not in st.session_state:
    st.session_state.gemini_result = None


# ============================================================
# 9. 사이드바
# ============================================================

st.sidebar.header("모델 및 검사 설정")

available_models = discover_memory_models(
    MODEL_ROOT
)

selected_model_info: dict[str, Any] | None = None
memory: np.ndarray | None = None
nearest: NearestNeighbors | None = None
settings: dict[str, Any] | None = None

if not available_models:
    st.sidebar.error(
        "GitHub 저장소에서 정상 특징 모델을 찾지 못했습니다."
    )
    st.sidebar.code(
        "models/patch_anomaly/\n"
        "├── screw_memory.npy\n"
        "└── screw_settings.json"
    )

    st.error(
        "GitHub 저장소의 models/patch_anomaly 폴더에 "
        "정상 특징 모델 파일을 올려야 합니다."
    )

else:
    model_names = [
        str(model["name"])
        for model in available_models
    ]

    selected_model_name = st.sidebar.selectbox(
        "사용할 정상 특징 모델",
        model_names,
    )

    selected_model_info = next(
        model
        for model in available_models
        if model["name"] == selected_model_name
    )

    try:
        memory_path = Path(
            selected_model_info["memory_path"]
        )
        settings_path = Path(
            selected_model_info["settings_path"]
        )

        memory, nearest, settings = load_memory_and_nearest(
            memory_path_text=str(memory_path),
            settings_path_text=str(settings_path),
            memory_modified_time=memory_path.stat().st_mtime,
        )

        st.sidebar.success(
            f"모델 준비 완료: {selected_model_name}"
        )
        st.sidebar.caption(
            f"정상 특징 수: {memory.shape[0]:,}개"
        )

        with st.sidebar.expander(
            "모델 설정값 확인"
        ):
            st.json(settings)

    except Exception as error:
        st.sidebar.exception(error)

threshold = st.sidebar.number_input(
    "PASS/FAIL 임계값",
    min_value=0.0,
    value=0.20,
    step=0.01,
    format="%.4f",
    help=(
        "정상과 불량 이미지 점수 분포를 비교하여 "
        "검증한 임계값을 입력하세요."
    ),
)

heatmap_alpha = st.sidebar.slider(
    "Heatmap 투명도",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)

st.sidebar.divider()
st.sidebar.header("Gemini 결과 해석")

gemini_api_key = st.sidebar.text_input(
    "Gemini API 키",
    type="password",
    help=(
        "현재 세션에서만 사용하며 코드나 GitHub 저장소에는 저장하지 않습니다."
    ),
)

gemini_model = st.sidebar.text_input(
    "Gemini 모델",
    value="gemini-2.5-flash",
    help="이미지 입력을 지원하는 Gemini 모델명을 입력하세요.",
)

additional_request = st.sidebar.text_area(
    "추가 분석 요청",
    value=(
        "Heatmap과 이상 점수를 품질팀 관점에서 해석하고 "
        "추가 확인 항목을 알려주세요."
    ),
)


# ============================================================
# 10. 검사 이미지 업로드
# ============================================================

st.subheader("1. 검사 이미지 등록")

uploaded_files = st.file_uploader(
    "이미지 파일을 이곳에 드래그앤드롭하세요.",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    accept_multiple_files=True,
    help="한 장 또는 여러 장을 동시에 등록할 수 있습니다.",
)

if uploaded_files:
    st.success(
        f"{len(uploaded_files)}장의 이미지를 등록했습니다."
    )

    preview_columns = st.columns(
        min(4, len(uploaded_files))
    )

    for index, uploaded_file in enumerate(
        uploaded_files[:4]
    ):
        with preview_columns[index]:
            st.image(
                uploaded_file,
                caption=uploaded_file.name,
                use_container_width=True,
            )


# ============================================================
# 11. 검사 실행
# ============================================================

st.subheader("2. 코드 로직으로 이상 검사")

inspection_disabled = (
    not uploaded_files
    or nearest is None
    or settings is None
)

if st.button(
    "이미지 검사 시작",
    type="primary",
    disabled=inspection_disabled,
    use_container_width=True,
):
    try:
        records: list[dict[str, Any]] = []

        progress = st.progress(
            0,
            text="검사 이미지를 처리하고 있습니다.",
        )

        image_size = (
            int(settings["image_width"]),
            int(settings["image_height"]),
        )

        for index, uploaded_file in enumerate(
            uploaded_files,
            start=1,
        ):
            gray_image = uploaded_image_to_gray(
                uploaded_file=uploaded_file,
                image_size=image_size,
            )

            result = inspect_image(
                image=gray_image,
                nearest=nearest,
                settings=settings,
            )

            judgment = (
                "FAIL"
                if result["max_score"] >= float(threshold)
                else "PASS"
            )

            original_pil = gray_to_pil(
                result["image"]
            )

            overlay_pil = apply_heatmap(
                gray_image=result["image"],
                heatmap=result["heatmap"],
                alpha=float(heatmap_alpha),
            )

            result_panel = create_result_panel(
                original=original_pil,
                overlay=overlay_pil,
                filename=uploaded_file.name,
                max_score=result["max_score"],
                threshold=float(threshold),
                judgment=judgment,
            )

            records.append(
                {
                    "filename": uploaded_file.name,
                    "max_score": result["max_score"],
                    "mean_score": result["mean_score"],
                    "max_location": result["max_location"],
                    "threshold": float(threshold),
                    "judgment": judgment,
                    "result_panel": result_panel,
                }
            )

            progress.progress(
                index / len(uploaded_files),
                text=(
                    f"검사 중 {index}/{len(uploaded_files)}: "
                    f"{uploaded_file.name}"
                ),
            )

        st.session_state.inspection_records = records
        st.session_state.gemini_result = None

        st.success("전체 이미지 검사가 완료되었습니다.")

    except Exception as error:
        st.exception(error)


# ============================================================
# 12. 검사 결과
# ============================================================

records = st.session_state.inspection_records

if records:
    st.divider()
    st.subheader("3. 검사 결과")

    dataframe = pd.DataFrame(
        [
            {
                "파일명": record["filename"],
                "최대 이상 점수": round(
                    record["max_score"],
                    6,
                ),
                "평균 이상 점수": round(
                    record["mean_score"],
                    6,
                ),
                "임계값": record["threshold"],
                "판정": record["judgment"],
                "최대 이상 영역": str(
                    record["max_location"]
                ),
            }
            for record in records
        ]
    )

    pass_count = int(
        (dataframe["판정"] == "PASS").sum()
    )
    fail_count = int(
        (dataframe["판정"] == "FAIL").sum()
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "검사 이미지",
        len(records),
    )
    metric2.metric(
        "PASS",
        pass_count,
    )
    metric3.metric(
        "FAIL",
        fail_count,
    )

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )

    selected_filename = st.selectbox(
        "상세 확인 이미지",
        [
            record["filename"]
            for record in records
        ],
    )

    selected_record = next(
        record
        for record in records
        if record["filename"] == selected_filename
    )

    st.image(
        selected_record["result_panel"],
        caption=(
            f"{selected_record['filename']} | "
            f"{selected_record['judgment']} | "
            f"Score={selected_record['max_score']:.6f}"
        ),
        use_container_width=True,
    )

    st.markdown("### 전체 결과 모음")

    contact_columns = st.slider(
        "한 줄에 표시할 결과 수",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )

    contact_sheet = create_contact_sheet(
        records=records,
        columns=contact_columns,
    )

    st.image(
        contact_sheet,
        caption="전체 검사 결과",
        use_container_width=True,
    )


# ============================================================
# 13. Gemini 해석
# ============================================================

if records:
    st.divider()
    st.subheader("4. Gemini API 결과 해석")

    st.info(
        "좌측에 Gemini API 키를 입력한 후 아래 버튼을 누르세요. "
        "Gemini는 코드 로직의 결과 이미지와 점수를 보조 해석합니다."
    )

    if st.button(
        "선택 결과를 Gemini로 해석",
        type="primary",
        disabled=not gemini_api_key.strip(),
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Gemini가 검사 결과를 해석하고 있습니다."
            ):
                gemini_text = analyze_with_gemini(
                    api_key=gemini_api_key,
                    model_name=gemini_model.strip(),
                    record=selected_record,
                    additional_request=additional_request,
                )

            st.session_state.gemini_result = {
                "filename": selected_record["filename"],
                "text": gemini_text,
            }

        except Exception as error:
            st.exception(error)

    gemini_result = st.session_state.gemini_result

    if gemini_result:
        st.success(
            f"Gemini 해석 완료: {gemini_result['filename']}"
        )
        st.markdown(
            gemini_result["text"]
        )


# ============================================================
# 14. 결과 다운로드
# ============================================================

if records:
    st.divider()
    st.subheader("5. 결과 다운로드")

    result_zip = build_result_zip(
        records=records
    )

    st.download_button(
        "전체 검사 결과 ZIP 다운로드",
        data=result_zip,
        file_name="patch_anomaly_results.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.caption(
        "ZIP에는 개별 결과 이미지, 전체 모음 이미지, 검사 결과 CSV가 포함됩니다."
    )
