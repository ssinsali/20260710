# -*- coding: utf-8 -*-
"""
Patch 기반 이상 탐지 + OpenAI 해석 Streamlit 프로그램

기능
----
1. C:\VisionAI\datasets\screw\train\good의 정상 이미지로 특징 메모리 생성
2. 정상 특징과 검사 이미지 패치를 최근접 이웃 방식으로 비교
3. 이상 점수와 Heatmap 생성
4. 검사 이미지를 드래그앤드롭으로 여러 장 등록
5. 선택적으로 OpenAI API 키를 입력하여 수치와 이미지를 함께 해석
6. 정상 특징 모델을 파일로 저장하고 다시 불러오기

기본 데이터셋 구조
------------------
C:\VisionAI
└── datasets
    └── screw
        ├── train
        │   └── good
        └── test
            ├── good
            ├── scratch_head
            ├── scratch_neck
            ├── thread_side
            └── thread_top
"""

from __future__ import annotations

import base64
import io
import json
import math
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from PIL import Image
from sklearn.neighbors import NearestNeighbors


# ============================================================
# 1. Streamlit 화면 설정
# ============================================================

st.set_page_config(
    page_title="Patch 이상 탐지 검사",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Patch 기반 이상 탐지 검사")
st.caption(
    "정상 이미지의 패치 특징과 검사 이미지를 비교하여 이상 위치와 점수를 표시합니다."
)


# ============================================================
# 2. Matplotlib 한글 설정
# ============================================================

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. 기본 경로
# ============================================================

DEFAULT_ROOT = Path(r"C:\VisionAI")


# ============================================================
# 4. 공통 이미지 함수
# ============================================================

def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    한글 경로에서도 안전하게 이미지를 읽습니다.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"이미지를 찾지 못했습니다: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)

    if image is None:
        raise RuntimeError(f"이미지를 읽지 못했습니다: {path}")

    return image


def read_gray_from_path(
    path: str | Path,
    size: tuple[int, int],
) -> np.ndarray:
    """
    로컬 경로의 이미지를 흑백으로 읽고 0~1 범위로 변환합니다.
    """
    image = imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def read_gray_from_upload(
    uploaded_file: Any,
    size: tuple[int, int],
) -> np.ndarray:
    """
    Streamlit에 업로드된 이미지를 흑백으로 읽습니다.
    """
    file_bytes = np.asarray(
        bytearray(uploaded_file.getvalue()),
        dtype=np.uint8,
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise RuntimeError(
            f"업로드 이미지를 읽지 못했습니다: {uploaded_file.name}"
        )

    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


# ============================================================
# 5. 패치 특징 추출
# ============================================================

def patch_features(
    image: np.ndarray,
    patch: int,
    canny_low: int,
    canny_high: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    이미지를 patch x patch 영역으로 나누고 특징 3개를 계산합니다.

    특징
    ----
    1. 평균 밝기
    2. 밝기 표준편차
    3. 에지 비율
    """
    if patch <= 0:
        raise ValueError("패치 크기는 1 이상이어야 합니다.")

    if patch > image.shape[0] or patch > image.shape[1]:
        raise ValueError(
            f"패치 크기 {patch}가 이미지 크기 {image.shape}보다 큽니다."
        )

    features: list[list[float]] = []
    locations: list[tuple[int, int]] = []

    for y in range(0, image.shape[0] - patch + 1, patch):
        for x in range(0, image.shape[1] - patch + 1, patch):
            roi = image[y:y + patch, x:x + patch]
            roi_uint8 = (roi * 255).astype(np.uint8)

            edge = cv2.Canny(
                roi_uint8,
                canny_low,
                canny_high,
            )

            features.append(
                [
                    float(roi.mean()),
                    float(roi.std()),
                    float(edge.mean() / 255.0),
                ]
            )

            locations.append((x, y))

    return np.array(features, dtype=np.float32), locations


# ============================================================
# 6. 데이터셋 경로 검색
# ============================================================

def find_categories(dataset_root: Path) -> list[Path]:
    """
    train/good와 test 폴더를 가진 카테고리를 찾습니다.
    """
    if not dataset_root.exists():
        return []

    return sorted(
        [
            path
            for path in dataset_root.iterdir()
            if (
                path.is_dir()
                and (path / "train" / "good").exists()
                and (path / "test").exists()
            )
        ]
    )


def collect_normal_paths(
    category_path: Path,
    max_images: int,
) -> list[Path]:
    """
    train/good 폴더의 정상 이미지를 가져옵니다.
    """
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    paths = sorted(
        [
            path
            for path in (category_path / "train" / "good").iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ]
    )

    if max_images > 0:
        paths = paths[:max_images]

    return paths


# ============================================================
# 7. 정상 특징 메모리 생성
# ============================================================

def build_memory_model(
    normal_paths: list[Path],
    image_size: tuple[int, int],
    patch: int,
    canny_low: int,
    canny_high: int,
) -> tuple[np.ndarray, NearestNeighbors]:
    """
    정상 이미지 전체에서 패치 특징을 추출하고
    최근접 이웃 모델을 생성합니다.
    """
    if not normal_paths:
        raise RuntimeError("정상 학습 이미지가 없습니다.")

    memory_list: list[np.ndarray] = []

    progress = st.progress(
        0,
        text="정상 이미지 특징을 생성하고 있습니다.",
    )

    for index, path in enumerate(normal_paths, start=1):
        image = read_gray_from_path(path, image_size)

        features, _ = patch_features(
            image=image,
            patch=patch,
            canny_low=canny_low,
            canny_high=canny_high,
        )

        memory_list.append(features)

        progress.progress(
            index / len(normal_paths),
            text=(
                f"정상 이미지 처리 중 "
                f"{index}/{len(normal_paths)}: {path.name}"
            ),
        )

    memory = np.vstack(memory_list)

    nearest = NearestNeighbors(
        n_neighbors=1,
        algorithm="auto",
        metric="euclidean",
    )

    nearest.fit(memory)

    progress.progress(
        1.0,
        text="정상 특징 모델 생성이 완료되었습니다.",
    )

    return memory, nearest


# ============================================================
# 8. 모델 저장 / 불러오기
# ============================================================

def save_memory_model(
    model_dir: Path,
    category_name: str,
    memory: np.ndarray,
    nearest: NearestNeighbors,
    settings: dict[str, Any],
) -> dict[str, Path]:
    """
    정상 특징 데이터와 최근접 이웃 모델을 파일로 저장합니다.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    memory_path = model_dir / f"{category_name}_memory.npy"
    nearest_path = model_dir / f"{category_name}_nearest_model.joblib"
    settings_path = model_dir / f"{category_name}_settings.json"

    np.save(memory_path, memory)
    joblib.dump(nearest, nearest_path)

    settings_path.write_text(
        json.dumps(
            settings,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "memory_path": memory_path,
        "nearest_path": nearest_path,
        "settings_path": settings_path,
    }


def load_memory_model(
    model_dir: Path,
    category_name: str,
) -> tuple[np.ndarray, NearestNeighbors, dict[str, Any]]:
    """
    저장된 정상 특징 모델을 불러옵니다.
    """
    memory_path = model_dir / f"{category_name}_memory.npy"
    nearest_path = model_dir / f"{category_name}_nearest_model.joblib"
    settings_path = model_dir / f"{category_name}_settings.json"

    missing = [
        path
        for path in [memory_path, nearest_path, settings_path]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "저장된 모델 파일을 찾지 못했습니다:\n"
            + "\n".join(str(path) for path in missing)
        )

    memory = np.load(memory_path)
    nearest = joblib.load(nearest_path)
    settings = json.loads(
        settings_path.read_text(encoding="utf-8")
    )

    return memory, nearest, settings


# ============================================================
# 9. 이상 탐지
# ============================================================

def get_anomaly_result(
    image: np.ndarray,
    nearest: NearestNeighbors,
    patch: int,
    canny_low: int,
    canny_high: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    검사 이미지의 Heatmap과 이상 점수를 계산합니다.

    반환값
    ------
    image:
        0~1 범위의 흑백 이미지

    heatmap:
        0~1 범위의 이상 위치 지도

    max_score:
        가장 이상한 패치의 점수

    mean_score:
        전체 패치 평균 점수
    """
    features, locations = patch_features(
        image=image,
        patch=patch,
        canny_low=canny_low,
        canny_high=canny_high,
    )

    distances, _ = nearest.kneighbors(features)
    scores = distances.ravel()

    raw_heatmap = np.zeros_like(
        image,
        dtype=np.float32,
    )

    for score, (x, y) in zip(scores, locations):
        raw_heatmap[
            y:y + patch,
            x:x + patch
        ] = float(score)

    heatmap = cv2.normalize(
        raw_heatmap,
        None,
        0,
        1,
        cv2.NORM_MINMAX,
    )

    return (
        image,
        heatmap,
        float(scores.max()),
        float(scores.mean()),
    )


# ============================================================
# 10. 결과 이미지 생성
# ============================================================

def create_result_figure(
    image: np.ndarray,
    heatmap: np.ndarray,
    filename: str,
    max_score: float,
    threshold: float,
) -> Image.Image:
    """
    원본과 Heatmap을 나란히 배치한 결과 이미지를 생성합니다.
    """
    judgment = "FAIL" if max_score >= threshold else "PASS"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4),
    )

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"검사 이미지\n{filename}")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(
        heatmap,
        cmap="jet",
        alpha=0.5,
    )
    axes[1].set_title(
        f"{judgment} | Score={max_score:.4f}\n"
        f"Threshold={threshold:.4f}"
    )
    axes[1].axis("off")

    plt.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def pil_image_to_data_url(image: Image.Image) -> str:
    """
    PIL 이미지를 OpenAI 이미지 입력용 data URL로 변환합니다.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# ============================================================
# 11. OpenAI 결과 해석
# ============================================================

def analyze_with_openai(
    api_key: str,
    model_name: str,
    result_image: Image.Image,
    filename: str,
    max_score: float,
    mean_score: float,
    threshold: float,
    judgment: str,
    user_instruction: str,
) -> str:
    """
    수치 결과와 Heatmap 이미지를 OpenAI 모델에 전달하여
    검사 결과를 한국어로 해석합니다.
    """
    if not api_key.strip():
        raise ValueError("OpenAI API 키를 입력하세요.")

    client = OpenAI(api_key=api_key.strip())

    prompt = f"""
당신은 반도체 및 정밀가공품의 비전 검사 결과를 검토하는 품질 엔지니어입니다.

다음 결과를 참고하여 한국어로 분석하세요.

파일명: {filename}
알고리즘: 패치 기반 최근접 이웃 이상 탐지
최대 이상 점수: {max_score:.6f}
평균 이상 점수: {mean_score:.6f}
판정 임계값: {threshold:.6f}
자동 판정: {judgment}

반드시 다음 순서로 답하세요.

1. 자동 판정 요약
2. Heatmap에서 이상이 집중된 위치
3. 관찰 가능한 형상 또는 표면 특징
4. 가능한 원인 가설
5. 추가 확인이 필요한 항목
6. 최종 품질 판단 시 주의사항

주의:
- Heatmap과 점수만으로 실제 불량 원인을 확정하지 마세요.
- 이미지로 확인할 수 없는 재료, 공정, 치수 정보를 추측해서 단정하지 마세요.
- 자동 판정과 육안 관찰이 다르면 그 차이를 분명히 적으세요.

사용자 추가 요청:
{user_instruction.strip() or "추가 요청 없음"}
"""

    response = client.responses.create(
        model=model_name.strip(),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": pil_image_to_data_url(
                            result_image
                        ),
                    },
                ],
            }
        ],
    )

    return response.output_text


# ============================================================
# 12. 결과 ZIP 생성
# ============================================================

def build_result_zip(
    result_records: list[dict[str, Any]],
) -> bytes:
    """
    결과 이미지와 CSV를 ZIP 파일로 만듭니다.
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:

        csv_rows = []

        for index, record in enumerate(
            result_records,
            start=1,
        ):
            image_buffer = io.BytesIO()

            record["result_image"].save(
                image_buffer,
                format="PNG",
            )

            zip_file.writestr(
                f"{index:03d}_{Path(record['filename']).stem}_result.png",
                image_buffer.getvalue(),
            )

            csv_rows.append(
                {
                    "파일명": record["filename"],
                    "최대 이상 점수": record["max_score"],
                    "평균 이상 점수": record["mean_score"],
                    "임계값": record["threshold"],
                    "판정": record["judgment"],
                }
            )

        dataframe = pd.DataFrame(csv_rows)

        zip_file.writestr(
            "anomaly_results.csv",
            dataframe.to_csv(
                index=False
            ).encode("utf-8-sig"),
        )

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ============================================================
# 13. 세션 상태
# ============================================================

if "memory" not in st.session_state:
    st.session_state.memory = None

if "nearest" not in st.session_state:
    st.session_state.nearest = None

if "model_settings" not in st.session_state:
    st.session_state.model_settings = None

if "result_records" not in st.session_state:
    st.session_state.result_records = []


# ============================================================
# 14. 사이드바 설정
# ============================================================

with st.sidebar:
    st.header("프로젝트 설정")

    root_text = st.text_input(
        "프로젝트 ROOT",
        value=str(DEFAULT_ROOT),
        help=r"기본값: C:\VisionAI",
    )

    ROOT = Path(root_text)
    DATASET_ROOT = ROOT / "datasets"
    MODEL_DIR = ROOT / "models" / "patch_anomaly"

    st.caption("데이터셋 검색 위치")
    st.code(str(DATASET_ROOT))

    st.caption("모델 저장 위치")
    st.code(str(MODEL_DIR))

    st.divider()

    st.header("특징 추출 설정")

    image_width = st.selectbox(
        "입력 이미지 너비",
        [64, 128, 256, 512],
        index=1,
    )

    image_height = st.selectbox(
        "입력 이미지 높이",
        [64, 128, 256, 512],
        index=1,
    )

    patch_size = st.selectbox(
        "패치 크기",
        [8, 16, 32, 64],
        index=1,
    )

    canny_low = st.number_input(
        "Canny 낮은 임계값",
        min_value=0,
        max_value=255,
        value=40,
        step=5,
    )

    canny_high = st.number_input(
        "Canny 높은 임계값",
        min_value=0,
        max_value=255,
        value=120,
        step=5,
    )

    max_normal_images = st.number_input(
        "사용할 정상 이미지 최대 수",
        min_value=1,
        value=80,
        step=1,
    )

    threshold = st.number_input(
        "PASS/FAIL 이상 점수 임계값",
        min_value=0.0,
        value=0.20,
        step=0.01,
        format="%.4f",
        help=(
            "이 값은 데이터에 맞게 정상/불량 점수 분포를 보고 조정해야 합니다."
        ),
    )


# ============================================================
# 15. 탭 구성
# ============================================================

tab_model, tab_inspection, tab_ai, tab_download = st.tabs(
    [
        "1️⃣ 정상 모델 생성",
        "2️⃣ 이미지 검사",
        "3️⃣ API 결과 해석",
        "4️⃣ 결과 다운로드",
    ]
)


# ============================================================
# 탭 1: 정상 모델 생성
# ============================================================

with tab_model:
    st.subheader("정상 이미지 특징 모델 생성")

    categories = find_categories(DATASET_ROOT)

    if not categories:
        st.error(
            "사용 가능한 데이터셋 카테고리를 찾지 못했습니다."
        )
        st.code(
            str(
                DATASET_ROOT
                / "screw"
                / "train"
                / "good"
            )
        )
    else:
        category_names = [path.name for path in categories]

        selected_category_name = st.selectbox(
            "데이터셋 카테고리",
            category_names,
        )

        category_path = next(
            path
            for path in categories
            if path.name == selected_category_name
        )

        normal_paths = collect_normal_paths(
            category_path=category_path,
            max_images=int(max_normal_images),
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "정상 이미지 수",
            len(normal_paths),
        )

        patches_per_image = (
            int(image_width) // int(patch_size)
        ) * (
            int(image_height) // int(patch_size)
        )

        col2.metric(
            "이미지당 패치 수",
            patches_per_image,
        )

        col3.metric(
            "예상 전체 패치 수",
            len(normal_paths) * patches_per_image,
        )

        st.write("**정상 이미지 경로**")
        st.code(
            str(category_path / "train" / "good")
        )

        build_col, load_col = st.columns(2)

        with build_col:
            if st.button(
                "정상 모델 생성",
                type="primary",
                use_container_width=True,
            ):
                try:
                    memory, nearest = build_memory_model(
                        normal_paths=normal_paths,
                        image_size=(
                            int(image_width),
                            int(image_height),
                        ),
                        patch=int(patch_size),
                        canny_low=int(canny_low),
                        canny_high=int(canny_high),
                    )

                    settings = {
                        "category_name": selected_category_name,
                        "image_width": int(image_width),
                        "image_height": int(image_height),
                        "patch_size": int(patch_size),
                        "canny_low": int(canny_low),
                        "canny_high": int(canny_high),
                        "normal_image_count": len(normal_paths),
                    }

                    saved_paths = save_memory_model(
                        model_dir=MODEL_DIR,
                        category_name=selected_category_name,
                        memory=memory,
                        nearest=nearest,
                        settings=settings,
                    )

                    st.session_state.memory = memory
                    st.session_state.nearest = nearest
                    st.session_state.model_settings = settings

                    st.success("정상 특징 모델 생성 및 저장 완료")

                    st.write("**저장된 파일**")

                    for name, path in saved_paths.items():
                        st.code(f"{name}: {path}")

                except Exception as error:
                    st.exception(error)

        with load_col:
            if st.button(
                "저장된 모델 불러오기",
                use_container_width=True,
            ):
                try:
                    memory, nearest, settings = load_memory_model(
                        model_dir=MODEL_DIR,
                        category_name=selected_category_name,
                    )

                    st.session_state.memory = memory
                    st.session_state.nearest = nearest
                    st.session_state.model_settings = settings

                    st.success("저장된 모델을 불러왔습니다.")
                    st.json(settings)

                except Exception as error:
                    st.exception(error)

        if st.session_state.nearest is not None:
            st.info(
                f"현재 사용 가능한 정상 특징 수: "
                f"{st.session_state.memory.shape[0]:,}개"
            )


# ============================================================
# 탭 2: 이미지 검사
# ============================================================

with tab_inspection:
    st.subheader("검사 이미지 드래그앤드롭")

    if st.session_state.nearest is None:
        st.warning(
            "먼저 1번 탭에서 정상 모델을 생성하거나 불러오세요."
        )

    uploaded_files = st.file_uploader(
        "검사할 이미지를 이곳에 끌어다 놓으세요.",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(
            f"{len(uploaded_files)}장의 이미지를 등록했습니다."
        )

    if st.button(
        "이상 탐지 검사 시작",
        type="primary",
        disabled=(
            st.session_state.nearest is None
            or not uploaded_files
        ),
        use_container_width=True,
    ):
        try:
            settings = st.session_state.model_settings or {
                "image_width": int(image_width),
                "image_height": int(image_height),
                "patch_size": int(patch_size),
                "canny_low": int(canny_low),
                "canny_high": int(canny_high),
            }

            result_records = []

            progress = st.progress(
                0,
                text="검사 이미지를 처리하고 있습니다.",
            )

            for index, uploaded_file in enumerate(
                uploaded_files,
                start=1,
            ):
                image = read_gray_from_upload(
                    uploaded_file=uploaded_file,
                    size=(
                        int(settings["image_width"]),
                        int(settings["image_height"]),
                    ),
                )

                (
                    result_image_array,
                    heatmap,
                    max_score,
                    mean_score,
                ) = get_anomaly_result(
                    image=image,
                    nearest=st.session_state.nearest,
                    patch=int(settings["patch_size"]),
                    canny_low=int(settings["canny_low"]),
                    canny_high=int(settings["canny_high"]),
                )

                judgment = (
                    "FAIL"
                    if max_score >= float(threshold)
                    else "PASS"
                )

                result_image = create_result_figure(
                    image=result_image_array,
                    heatmap=heatmap,
                    filename=uploaded_file.name,
                    max_score=max_score,
                    threshold=float(threshold),
                )

                result_records.append(
                    {
                        "filename": uploaded_file.name,
                        "image": result_image_array,
                        "heatmap": heatmap,
                        "max_score": max_score,
                        "mean_score": mean_score,
                        "threshold": float(threshold),
                        "judgment": judgment,
                        "result_image": result_image,
                    }
                )

                progress.progress(
                    index / len(uploaded_files),
                    text=(
                        f"검사 중 {index}/{len(uploaded_files)}: "
                        f"{uploaded_file.name}"
                    ),
                )

            st.session_state.result_records = result_records
            st.success("전체 이미지 검사가 완료되었습니다.")

        except Exception as error:
            st.exception(error)

    result_records = st.session_state.result_records

    if result_records:
        st.divider()
        st.subheader("검사 결과")

        result_dataframe = pd.DataFrame(
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
                }
                for record in result_records
            ]
        )

        pass_count = int(
            (result_dataframe["판정"] == "PASS").sum()
        )
        fail_count = int(
            (result_dataframe["판정"] == "FAIL").sum()
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "전체 검사 수",
            len(result_dataframe),
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
            result_dataframe,
            use_container_width=True,
            hide_index=True,
        )

        selected_result_name = st.selectbox(
            "상세 확인 이미지",
            [
                record["filename"]
                for record in result_records
            ],
        )

        selected_record = next(
            record
            for record in result_records
            if record["filename"] == selected_result_name
        )

        st.image(
            selected_record["result_image"],
            caption=(
                f"{selected_record['filename']} | "
                f"{selected_record['judgment']}"
            ),
            use_container_width=True,
        )


# ============================================================
# 탭 3: OpenAI API 결과 해석
# ============================================================

with tab_ai:
    st.subheader("OpenAI API 기반 검사 결과 해석")

    st.info(
        "API 해석은 자동 이상 탐지 결과를 설명하는 보조 기능입니다. "
        "최종 품질 판정은 실제 제품 기준, 공정 정보 및 측정 결과와 함께 검토하세요."
    )

    openai_api_key = st.text_input(
        "OpenAI API 키",
        type="password",
        help=(
            "입력한 키는 이 앱의 현재 실행에서만 사용하며 코드 파일에 저장하지 않습니다."
        ),
    )

    openai_model = st.text_input(
        "OpenAI 모델 이름",
        value="gpt-4.1-mini",
        help="이미지 입력을 지원하는 모델 이름을 입력하세요.",
    )

    user_instruction = st.text_area(
        "추가 분석 요청",
        value=(
            "Heatmap 위치와 점수를 바탕으로 품질팀 관점에서 "
            "확인해야 할 내용을 설명해 주세요."
        ),
    )

    if not st.session_state.result_records:
        st.warning(
            "먼저 2번 탭에서 이미지 검사를 실행하세요."
        )
    else:
        ai_target_name = st.selectbox(
            "API로 해석할 결과 이미지",
            [
                record["filename"]
                for record in st.session_state.result_records
            ],
            key="ai_target",
        )

        ai_target = next(
            record
            for record in st.session_state.result_records
            if record["filename"] == ai_target_name
        )

        st.image(
            ai_target["result_image"],
            caption=ai_target_name,
            use_container_width=True,
        )

        if st.button(
            "API로 검사 결과 해석",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "이미지와 이상 점수를 분석하고 있습니다."
                ):
                    ai_result = analyze_with_openai(
                        api_key=openai_api_key,
                        model_name=openai_model,
                        result_image=ai_target["result_image"],
                        filename=ai_target["filename"],
                        max_score=ai_target["max_score"],
                        mean_score=ai_target["mean_score"],
                        threshold=ai_target["threshold"],
                        judgment=ai_target["judgment"],
                        user_instruction=user_instruction,
                    )

                st.success("API 분석 완료")
                st.markdown(ai_result)

            except Exception as error:
                st.exception(error)


# ============================================================
# 탭 4: 결과 다운로드
# ============================================================

with tab_download:
    st.subheader("검사 결과 저장")

    if not st.session_state.result_records:
        st.warning(
            "다운로드할 검사 결과가 없습니다."
        )
    else:
        result_zip = build_result_zip(
            st.session_state.result_records
        )

        st.download_button(
            "검사 결과 ZIP 다운로드",
            data=result_zip,
            file_name="patch_anomaly_results.zip",
            mime="application/zip",
            use_container_width=True,
        )

        st.caption(
            "ZIP에는 이미지별 Heatmap 결과와 anomaly_results.csv가 포함됩니다."
        )

        result_dataframe = pd.DataFrame(
            [
                {
                    "파일명": record["filename"],
                    "최대 이상 점수": record["max_score"],
                    "평균 이상 점수": record["mean_score"],
                    "임계값": record["threshold"],
                    "판정": record["judgment"],
                }
                for record in st.session_state.result_records
            ]
        )

        st.download_button(
            "검사 결과 CSV 다운로드",
            data=result_dataframe.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="anomaly_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
