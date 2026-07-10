# -*- coding: utf-8 -*-
"""
위치 기반 Patch 이상 탐지 + Gemini 보조 해석 Streamlit 앱

GitHub 구조
-----------
repository/
├── app.py
├── requirements.txt
└── models/
    └── patch_anomaly/
        ├── screw_position_memory.npy
        └── screw_position_settings.json
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


st.set_page_config(
    page_title="위치 기반 Patch 이상 검사",
    page_icon="🔬",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_ROOT = APP_DIR / "models" / "patch_anomaly"

st.title("🔬 위치 기반 Patch 이상 검사")
st.caption(
    "검사 Patch를 동일 위치의 정상 Patch와 비교하여 이상 점수와 Heatmap을 생성합니다."
)


def discover_models(model_root: Path) -> list[dict[str, Any]]:
    if not model_root.exists():
        return []

    models: list[dict[str, Any]] = []

    for memory_path in sorted(
        model_root.rglob("*_position_memory.npy")
    ):
        prefix = memory_path.name.removesuffix(
            "_position_memory.npy"
        )
        settings_path = memory_path.with_name(
            f"{prefix}_position_settings.json"
        )

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
def load_model(
    memory_path_text: str,
    settings_path_text: str,
    modified_time: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    del modified_time

    memory = np.load(
        memory_path_text,
        allow_pickle=False,
    )

    settings = json.loads(
        Path(settings_path_text).read_text(
            encoding="utf-8"
        )
    )

    if memory.ndim != 3 or memory.shape[2] != 3:
        raise ValueError(
            "현재 Memory 파일은 위치 기반 형식이 아닙니다. "
            "새 모델 생성기로 다시 생성하세요. "
            "예상 형식: (정상 이미지 수, Patch 수, 3)"
        )

    if settings.get("memory_type") != "position_based":
        raise ValueError(
            "설정 파일이 위치 기반 모델 형식이 아닙니다."
        )

    return memory, settings


def uploaded_to_gray(
    uploaded_file: Any,
    image_size: tuple[int, int],
) -> np.ndarray:
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

            edges = cv2.Canny(
                (roi * 255).astype(np.uint8),
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

    return np.asarray(
        features,
        dtype=np.float32,
    ), locations


def aggregate_image_score(
    patch_scores: np.ndarray,
    top_ratio: float,
) -> float:
    top_count = max(
        1,
        int(math.ceil(len(patch_scores) * top_ratio)),
    )

    top_scores = np.partition(
        patch_scores,
        -top_count,
    )[-top_count:]

    return float(top_scores.mean())


def inspect_image(
    image: np.ndarray,
    position_memory: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    patch_size = int(settings["patch_size"])

    features, locations = patch_features(
        image=image,
        patch_size=patch_size,
        canny_low=int(settings["canny_low"]),
        canny_high=int(settings["canny_high"]),
    )

    if features.shape[0] != position_memory.shape[1]:
        raise ValueError(
            "검사 이미지의 Patch 수와 모델의 Patch 수가 다릅니다."
        )

    # 동일 위치 Patch끼리만 비교
    differences = (
        position_memory
        - features[np.newaxis, :, :]
    )

    distances = np.linalg.norm(
        differences,
        axis=2,
    )

    patch_scores = distances.min(axis=0)

    top_ratio = float(
        settings.get("top_ratio", 0.05)
    )

    image_score = aggregate_image_score(
        patch_scores,
        top_ratio,
    )

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

    # 화면 표시용 정규화이며 판정에는 raw 점수를 사용합니다.
    display_heatmap = cv2.normalize(
        raw_heatmap,
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    max_index = int(
        np.argmax(patch_scores)
    )
    max_x, max_y = locations[max_index]

    return {
        "image": image,
        "heatmap": display_heatmap,
        "patch_scores": patch_scores,
        "image_score": image_score,
        "max_patch_score": float(
            patch_scores.max()
        ),
        "mean_patch_score": float(
            patch_scores.mean()
        ),
        "max_location": (
            max_x,
            max_y,
            max_x + patch_size,
            max_y + patch_size,
        ),
    }


def gray_to_pil(image: np.ndarray) -> Image.Image:
    image_uint8 = np.clip(
        image * 255,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(
        image_uint8
    ).convert("RGB")


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float,
) -> Image.Image:
    gray_uint8 = np.clip(
        image * 255,
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

    overlay = cv2.addWeighted(
        gray_bgr,
        1.0 - alpha,
        heat_bgr,
        alpha,
        0,
    )

    return Image.fromarray(
        cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB,
        )
    )


def make_panel(
    original: Image.Image,
    overlay: Image.Image,
    filename: str,
    score: float,
    threshold: float,
    judgment: str,
) -> Image.Image:
    width = max(
        original.width,
        overlay.width,
    )
    height = max(
        original.height,
        overlay.height,
    )

    header = 55

    panel = Image.new(
        "RGB",
        (width * 2, height + header),
        (245, 245, 245),
    )

    panel.paste(
        original.resize((width, height)),
        (0, header),
    )
    panel.paste(
        overlay.resize((width, height)),
        (width, header),
    )

    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    draw.text(
        (8, 8),
        f"Original: {filename}",
        fill=(20, 20, 20),
        font=font,
    )

    draw.text(
        (width + 8, 8),
        (
            f"{judgment} | score={score:.6f} "
            f"| threshold={threshold:.6f}"
        ),
        fill=(20, 20, 20),
        font=font,
    )

    return panel


def image_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def analyze_with_gemini(
    api_key: str,
    model_name: str,
    record: dict[str, Any],
) -> str:
    client = genai.Client(
        api_key=api_key.strip()
    )

    normalized_name = model_name.strip()

    if normalized_name.startswith("models/"):
        normalized_name = normalized_name.removeprefix(
            "models/"
        )

    prompt = f"""
당신은 비전 검사 결과를 검토하는 품질 엔지니어입니다.

파일명: {record['filename']}
검사 방식: 동일 위치 정상 Patch 비교
이미지 점수: {record['image_score']:.6f}
최대 Patch 점수: {record['max_patch_score']:.6f}
평균 Patch 점수: {record['mean_patch_score']:.6f}
임계값: {record['threshold']:.6f}
자동 판정: {record['judgment']}
최대 이상 위치: {record['max_location']}

원본과 Heatmap을 참고해 다음을 한국어로 설명하세요.

1. 판정 요약
2. 이상 반응 위치
3. 육안으로 보이는 특징
4. 가능한 원인 가설
5. 추가 확인 항목

이미지와 점수만으로 원인을 확정하지 마세요.
"""

    response = client.models.generate_content(
        model=normalized_name,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=image_bytes(
                    record["panel"]
                ),
                mime_type="image/png",
            ),
        ],
    )

    return response.text or "응답 텍스트가 없습니다."


def build_zip(
    records: list[dict[str, Any]],
) -> bytes:
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as zip_file:
        rows = []

        for index, record in enumerate(
            records,
            start=1,
        ):
            zip_file.writestr(
                (
                    f"{index:03d}_"
                    f"{Path(record['filename']).stem}_result.png"
                ),
                image_bytes(record["panel"]),
            )

            rows.append(
                {
                    "파일명": record["filename"],
                    "이미지 점수": record["image_score"],
                    "최대 Patch 점수": record["max_patch_score"],
                    "평균 Patch 점수": record["mean_patch_score"],
                    "임계값": record["threshold"],
                    "판정": record["judgment"],
                    "최대 이상 위치": record["max_location"],
                }
            )

        dataframe = pd.DataFrame(rows)

        zip_file.writestr(
            "anomaly_results.csv",
            dataframe.to_csv(
                index=False
            ).encode("utf-8-sig"),
        )

    buffer.seek(0)
    return buffer.getvalue()


if "records" not in st.session_state:
    st.session_state.records = []

if "gemini_text" not in st.session_state:
    st.session_state.gemini_text = None


models = discover_models(MODEL_ROOT)

position_memory = None
settings = None

st.sidebar.header("검사 모델")

if not models:
    st.sidebar.error(
        "위치 기반 모델 파일이 없습니다."
    )
    st.sidebar.code(
        "models/patch_anomaly/\n"
        "├── screw_position_memory.npy\n"
        "└── screw_position_settings.json"
    )
else:
    selected_name = st.sidebar.selectbox(
        "모델 선택",
        [model["name"] for model in models],
    )

    selected = next(
        model
        for model in models
        if model["name"] == selected_name
    )

    try:
        memory_path = Path(
            selected["memory_path"]
        )

        position_memory, settings = load_model(
            str(memory_path),
            str(selected["settings_path"]),
            memory_path.stat().st_mtime,
        )

        st.sidebar.success(
            f"{selected_name} 모델 준비 완료"
        )

        with st.sidebar.expander(
            "정상 점수와 설정 확인"
        ):
            st.json(settings)

    except Exception as error:
        st.sidebar.exception(error)


recommended_threshold = (
    float(
        settings.get(
            "recommended_threshold",
            0.20,
        )
    )
    if settings
    else 0.20
)

threshold = st.sidebar.number_input(
    "PASS/FAIL 임계값",
    min_value=0.0,
    value=recommended_threshold,
    step=max(
        recommended_threshold / 20,
        0.001,
    ),
    format="%.6f",
)

st.sidebar.caption(
    f"모델 권장 임계값: {recommended_threshold:.6f}"
)

heatmap_alpha = st.sidebar.slider(
    "Heatmap 투명도",
    0.10,
    0.90,
    0.50,
    0.05,
)

st.sidebar.divider()
st.sidebar.header("Gemini 해석")

gemini_key = st.sidebar.text_input(
    "Gemini API 키",
    type="password",
)

gemini_model = st.sidebar.text_input(
    "Gemini 모델명",
    value="gemini-flash-latest",
)


uploaded_files = st.file_uploader(
    "검사 이미지를 드래그앤드롭하세요.",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    accept_multiple_files=True,
)

disabled = (
    not uploaded_files
    or position_memory is None
    or settings is None
)

if st.button(
    "이미지 검사 시작",
    type="primary",
    disabled=disabled,
    use_container_width=True,
):
    try:
        records = []

        for uploaded_file in uploaded_files:
            gray = uploaded_to_gray(
                uploaded_file,
                (
                    int(settings["image_width"]),
                    int(settings["image_height"]),
                ),
            )

            result = inspect_image(
                gray,
                position_memory,
                settings,
            )

            judgment = (
                "FAIL"
                if result["image_score"] >= threshold
                else "PASS"
            )

            original = gray_to_pil(
                result["image"]
            )
            overlay = overlay_heatmap(
                result["image"],
                result["heatmap"],
                heatmap_alpha,
            )

            panel = make_panel(
                original,
                overlay,
                uploaded_file.name,
                result["image_score"],
                threshold,
                judgment,
            )

            records.append(
                {
                    "filename": uploaded_file.name,
                    "image_score": result["image_score"],
                    "max_patch_score": result["max_patch_score"],
                    "mean_patch_score": result["mean_patch_score"],
                    "max_location": result["max_location"],
                    "threshold": threshold,
                    "judgment": judgment,
                    "panel": panel,
                }
            )

        st.session_state.records = records
        st.session_state.gemini_text = None

    except Exception as error:
        st.exception(error)


records = st.session_state.records

if records:
    dataframe = pd.DataFrame(
        [
            {
                "파일명": record["filename"],
                "이미지 점수": round(
                    record["image_score"],
                    6,
                ),
                "최대 Patch 점수": round(
                    record["max_patch_score"],
                    6,
                ),
                "임계값": round(
                    record["threshold"],
                    6,
                ),
                "판정": record["judgment"],
            }
            for record in records
        ]
    )

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )

    selected_filename = st.selectbox(
        "상세 결과",
        [record["filename"] for record in records],
    )

    selected_record = next(
        record
        for record in records
        if record["filename"] == selected_filename
    )

    st.image(
        selected_record["panel"],
        use_container_width=True,
    )

    if st.button(
        "선택 결과 Gemini 해석",
        disabled=not gemini_key.strip(),
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Gemini가 결과를 해석하고 있습니다."
            ):
                st.session_state.gemini_text = analyze_with_gemini(
                    gemini_key,
                    gemini_model,
                    selected_record,
                )
        except Exception as error:
            st.exception(error)

    if st.session_state.gemini_text:
        st.markdown(
            st.session_state.gemini_text
        )

    st.download_button(
        "검사 결과 ZIP 다운로드",
        data=build_zip(records),
        file_name="position_patch_results.zip",
        mime="application/zip",
        use_container_width=True,
    )
