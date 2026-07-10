# -*- coding: utf-8 -*-
"""
Vision AI YOLO 통합 Streamlit 프로그램

실행 방법
---------
1. Windows 명령 프롬프트에서 프로젝트 폴더로 이동
   cd /d C:\VisionAI

2. Streamlit 실행
   streamlit run vision_ai_streamlit_app.py

주요 기능
---------
1. MVTec 형식 데이터셋 확인
2. Ground Truth 마스크를 이용한 YOLO 데이터셋 생성
3. YOLO 모델 학습
4. 검증 이미지 예측
5. 개별 예측 이미지 확인
6. 여러 예측 이미지를 한 장으로 합친 모음 이미지 저장
7. 예측 박스의 신뢰도와 좌표 확인
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from shutil import copy2, rmtree
from typing import Any

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# ============================================================
# 1. Streamlit 기본 설정
# ============================================================

st.set_page_config(
    page_title="Vision AI YOLO 통합 프로그램",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Vision AI YOLO 통합 프로그램")
st.caption(
    "데이터셋 확인 → YOLO 데이터셋 생성 → 모델 학습 → 검증 예측 → 결과 확인"
)


# ============================================================
# 2. 공통 함수
# ============================================================

def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    한글 또는 특수문자가 포함된 Windows 경로에서도 이미지를 안전하게 읽습니다.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"이미지 파일이 없습니다: {path}")

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)

    if image is None:
        raise FileNotFoundError(f"이미지를 읽지 못했습니다: {path}")

    return image


def find_mvtec_root(root: Path) -> Path | None:
    """프로젝트 ROOT 아래의 datasets 폴더를 찾습니다."""
    dataset_root = root / "datasets"
    return dataset_root if dataset_root.exists() else None


def list_mvtec_categories(mvtec_root: Path | None) -> list[Path]:
    """
    train/good와 test 폴더를 가진 데이터셋 카테고리를 찾습니다.
    """
    if mvtec_root is None:
        return []

    categories: list[Path] = []

    for path in sorted(mvtec_root.iterdir()):
        if (
            path.is_dir()
            and (path / "train" / "good").exists()
            and (path / "test").exists()
        ):
            categories.append(path)

    return categories


def categories_with_masks(categories: list[Path]) -> list[Path]:
    """ground_truth 폴더가 있는 카테고리만 반환합니다."""
    return [
        category
        for category in categories
        if (category / "ground_truth").exists()
    ]


def bbox_from_mask(mask_path: Path) -> tuple[int, int, int, int] | None:
    """
    마스크의 모든 흰색 영역을 포함하는 하나의 Bounding Box를 계산합니다.
    """
    mask = imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def write_label(
    label_path: Path,
    bbox: tuple[int, int, int, int],
    shape: tuple[int, ...],
    class_id: int,
) -> None:
    """
    Bounding Box를 YOLO 형식으로 변환하여 txt 파일로 저장합니다.

    YOLO 형식:
    class_id center_x center_y width height
    """
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox

    # 픽셀 한 개짜리 영역도 최소한 포함하도록 +1을 적용합니다.
    box_width_px = max(1, x2 - x1 + 1)
    box_height_px = max(1, y2 - y1 + 1)

    center_x = ((x1 + x2 + 1) / 2) / w
    center_y = ((y1 + y2 + 1) / 2) / h
    box_width = box_width_px / w
    box_height = box_height_px / h

    label_path.write_text(
        f"{class_id} "
        f"{center_x:.6f} "
        f"{center_y:.6f} "
        f"{box_width:.6f} "
        f"{box_height:.6f}\n",
        encoding="utf-8",
    )


def create_yolo_dataset(
    category_path: Path,
    output_dir: Path,
    max_images_per_defect: int,
    val_interval: int,
    class_id: int,
    class_name: str,
    overwrite: bool,
) -> dict[str, Any]:
    """
    test 이미지와 ground_truth 마스크를 연결하여 YOLO 데이터셋을 생성합니다.
    """
    test_dir = category_path / "test"
    mask_root = category_path / "ground_truth"

    if not category_path.exists():
        raise FileNotFoundError(f"데이터셋 폴더가 없습니다: {category_path}")

    if not test_dir.exists():
        raise FileNotFoundError(f"test 폴더가 없습니다: {test_dir}")

    if not mask_root.exists():
        raise FileNotFoundError(f"ground_truth 폴더가 없습니다: {mask_root}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"출력 폴더가 이미 존재합니다: {output_dir}\n"
                "기존 폴더 삭제 허용을 선택하거나 다른 폴더명을 사용하세요."
            )
        rmtree(output_dir)

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    defect_dirs = [
        path
        for path in sorted(test_dir.iterdir())
        if path.is_dir() and path.name != "good"
    ]

    planned_images: list[tuple[Path, Path, str]] = []

    for defect_dir in defect_dirs:
        for image_path in sorted(defect_dir.glob("*.png"))[:max_images_per_defect]:
            mask_path = (
                mask_root
                / defect_dir.name
                / f"{image_path.stem}_mask.png"
            )
            planned_images.append((image_path, mask_path, defect_dir.name))

    progress = st.progress(0, text="YOLO 데이터셋 생성을 준비하고 있습니다.")
    status = st.empty()

    image_index = 0
    train_count = 0
    val_count = 0
    missing_mask_count = 0
    empty_mask_count = 0
    defect_summary: dict[str, int] = {}

    total = max(1, len(planned_images))

    for task_index, (image_path, mask_path, defect_name) in enumerate(planned_images):
        status.write(
            f"처리 중: **{defect_name} / {image_path.name}** "
            f"({task_index + 1}/{len(planned_images)})"
        )

        if not mask_path.exists():
            missing_mask_count += 1
            progress.progress(
                (task_index + 1) / total,
                text="마스크 파일 누락 항목을 건너뛰는 중입니다.",
            )
            continue

        bbox = bbox_from_mask(mask_path)

        if bbox is None:
            empty_mask_count += 1
            progress.progress(
                (task_index + 1) / total,
                text="빈 마스크 항목을 건너뛰는 중입니다.",
            )
            continue

        split = "val" if image_index % val_interval == 0 else "train"

        target_image = (
            output_dir / "images" / split / f"{image_index:04d}.png"
        )
        target_label = (
            output_dir / "labels" / split / f"{image_index:04d}.txt"
        )

        copy2(image_path, target_image)
        original_image = imread(image_path)

        write_label(
            target_label,
            bbox,
            original_image.shape,
            class_id,
        )

        if split == "train":
            train_count += 1
        else:
            val_count += 1

        defect_summary[defect_name] = defect_summary.get(defect_name, 0) + 1
        image_index += 1

        progress.progress(
            (task_index + 1) / total,
            text="이미지와 YOLO 라벨을 생성하고 있습니다.",
        )

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {output_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  {class_id}: {class_name}\n",
        encoding="utf-8",
    )

    progress.progress(1.0, text="YOLO 데이터셋 생성이 완료되었습니다.")
    status.empty()

    return {
        "work": output_dir,
        "yaml_path": yaml_path,
        "train_count": train_count,
        "val_count": val_count,
        "total_count": image_index,
        "missing_mask_count": missing_mask_count,
        "empty_mask_count": empty_mask_count,
        "defect_summary": defect_summary,
    }


def get_train_device() -> int | str:
    """CUDA 사용 가능 시 첫 번째 GPU, 그렇지 않으면 CPU를 반환합니다."""
    return 0 if torch.cuda.is_available() else "cpu"


def train_yolo_model(
    yaml_path: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    amp: bool,
    device: int | str,
    project_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    """YOLO 모델을 학습하고 best.pt 경로를 반환합니다."""
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml 파일이 없습니다: {yaml_path}")

    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)

    train_result = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        amp=amp,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )

    save_dir = Path(train_result.save_dir)
    best_model_path = save_dir / "weights" / "best.pt"

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"학습은 종료되었지만 best.pt를 찾지 못했습니다: {best_model_path}"
        )

    return {
        "train_result": train_result,
        "save_dir": save_dir,
        "best_model_path": best_model_path,
    }


def predict_validation_images(
    best_model_path: Path,
    source_dir: Path,
    output_project: Path,
    output_name: str,
    confidence: float,
    imgsz: int,
    device: int | str,
) -> dict[str, Any]:
    """학습된 모델로 검증 이미지를 예측하고 결과를 저장합니다."""
    if not best_model_path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {best_model_path}")

    if not source_dir.exists():
        raise FileNotFoundError(f"검증 이미지 폴더가 없습니다: {source_dir}")

    trained_model = YOLO(str(best_model_path))

    results = trained_model.predict(
        source=str(source_dir),
        save=True,
        conf=confidence,
        imgsz=imgsz,
        device=device,
        project=str(output_project),
        name=output_name,
        exist_ok=True,
    )

    if results:
        result_dir = Path(results[0].save_dir)
    else:
        result_dir = output_project / output_name

    return {
        "results": results,
        "result_dir": result_dir,
    }


def find_result_images(result_dir: Path) -> list[Path]:
    """예측 결과 폴더에서 지원되는 이미지 파일을 모두 찾습니다."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if not result_dir.exists():
        return []

    return sorted(
        path
        for path in result_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    )


def create_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    columns: int = 4,
    thumb_width: int = 360,
    thumb_height: int = 280,
    margin: int = 16,
    label_height: int = 34,
    background_value: int = 245,
) -> Path:
    """
    여러 개의 개별 예측 이미지를 한 장의 모음 이미지로 합쳐 저장합니다.

    사용자가 말한 '다른 이미지들이 한 파일에 정리된 결과'를 생성하는 기능입니다.
    """
    if not image_paths:
        raise ValueError("모음 이미지로 만들 예측 이미지가 없습니다.")

    columns = max(1, columns)
    rows = math.ceil(len(image_paths) / columns)

    cell_width = thumb_width + margin * 2
    cell_height = thumb_height + label_height + margin * 2

    sheet_width = columns * cell_width
    sheet_height = rows * cell_height

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        (background_value, background_value, background_value),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, image_path in enumerate(image_paths):
        row = index // columns
        col = index % columns

        x0 = col * cell_width + margin
        y0 = row * cell_height + margin

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_width, thumb_height))

            paste_x = x0 + (thumb_width - image.width) // 2
            paste_y = y0 + (thumb_height - image.height) // 2
            sheet.paste(image, (paste_x, paste_y))

        label = f"{index + 1}. {image_path.name}"
        draw.text(
            (x0, y0 + thumb_height + 8),
            label,
            fill=(20, 20, 20),
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)

    return output_path


def result_box_records(results: list[Any], limit: int = 5) -> list[dict[str, Any]]:
    """예측 결과에서 박스 수, 신뢰도, 좌표를 표 형식으로 정리합니다."""
    records: list[dict[str, Any]] = []

    for image_index, result in enumerate(results[:limit]):
        box_count = len(result.boxes)

        if box_count == 0:
            records.append(
                {
                    "이미지 번호": image_index,
                    "파일명": Path(result.path).name,
                    "박스 번호": "-",
                    "신뢰도": "-",
                    "x1": "-",
                    "y1": "-",
                    "x2": "-",
                    "y2": "-",
                }
            )
            continue

        confidences = result.boxes.conf.detach().cpu().numpy()
        coordinates = result.boxes.xyxy.detach().cpu().numpy()

        for box_index, (confidence, xyxy) in enumerate(
            zip(confidences, coordinates),
            start=1,
        ):
            records.append(
                {
                    "이미지 번호": image_index,
                    "파일명": Path(result.path).name,
                    "박스 번호": box_index,
                    "신뢰도": round(float(confidence), 4),
                    "x1": round(float(xyxy[0]), 1),
                    "y1": round(float(xyxy[1]), 1),
                    "x2": round(float(xyxy[2]), 1),
                    "y2": round(float(xyxy[3]), 1),
                }
            )

    return records


def open_folder(path: Path) -> None:
    """
    Windows에서 폴더를 엽니다.
    원격 서버에서 Streamlit을 실행할 때는 서버 컴퓨터의 폴더가 열립니다.
    """
    if os.name != "nt":
        raise OSError("이 기능은 현재 Windows에서만 지원합니다.")

    if not path.exists():
        raise FileNotFoundError(f"폴더가 없습니다: {path}")

    os.startfile(str(path))  # type: ignore[attr-defined]


# ============================================================
# 3. 세션 상태 초기화
# ============================================================

session_defaults = {
    "dataset_info": None,
    "train_info": None,
    "prediction_info": None,
    "current_image_index": 0,
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# 4. 사이드바 공통 설정
# ============================================================

with st.sidebar:
    st.header("공통 설정")

    root_text = st.text_input(
        "프로젝트 기준 폴더(ROOT)",
        value=r"C:\VisionAI",
        help="예: C:\\VisionAI",
    )
    ROOT = Path(root_text)

    st.divider()

    st.subheader("GPU 상태")

    if torch.cuda.is_available():
        st.success("CUDA GPU 사용 가능")
        st.write(f"GPU: `{torch.cuda.get_device_name(0)}`")
        st.write(f"PyTorch: `{torch.__version__}`")
        st.write(f"PyTorch CUDA: `{torch.version.cuda}`")
        st.write(
            "현재 할당 메모리: "
            f"`{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB`"
        )
    else:
        st.warning("GPU를 사용할 수 없어 CPU로 실행합니다.")
        st.write(f"PyTorch: `{torch.__version__}`")


# ============================================================
# 5. 프로그램 탭
# ============================================================

tab_dataset, tab_create, tab_train, tab_predict, tab_result = st.tabs(
    [
        "1️⃣ 데이터셋 확인",
        "2️⃣ YOLO 데이터셋 생성",
        "3️⃣ YOLO 학습",
        "4️⃣ 검증 예측",
        "5️⃣ 결과 확인",
    ]
)


# ============================================================
# 탭 1: 데이터셋 확인
# ============================================================

with tab_dataset:
    st.subheader("데이터셋 폴더 및 카테고리 확인")

    if st.button("데이터셋 검색", type="primary", key="search_dataset"):
        try:
            mvtec_root = find_mvtec_root(ROOT)
            categories = list_mvtec_categories(mvtec_root)
            mask_categories = categories_with_masks(categories)

            if mvtec_root is None:
                raise FileNotFoundError(
                    f"datasets 폴더를 찾지 못했습니다: {ROOT / 'datasets'}"
                )

            if not categories:
                raise RuntimeError(
                    "train/good 및 test 구조를 가진 카테고리를 찾지 못했습니다."
                )

            st.session_state.dataset_info = {
                "mvtec_root": mvtec_root,
                "categories": categories,
                "mask_categories": mask_categories,
            }

        except Exception as error:
            st.exception(error)

    dataset_info = st.session_state.dataset_info

    if dataset_info:
        st.success("데이터셋을 정상적으로 확인했습니다.")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**프로젝트 기준 폴더**")
            st.code(str(ROOT))

            st.write("**데이터셋 폴더**")
            st.code(str(dataset_info["mvtec_root"]))

        with col2:
            st.write("**사용 가능한 카테고리**")
            st.write(
                [path.name for path in dataset_info["categories"]]
            )

            st.write("**Ground Truth가 있는 카테고리**")
            st.write(
                [path.name for path in dataset_info["mask_categories"]]
            )


# ============================================================
# 탭 2: YOLO 데이터셋 생성
# ============================================================

with tab_create:
    st.subheader("Ground Truth 마스크 → YOLO 데이터셋 변환")

    mvtec_root = find_mvtec_root(ROOT)
    categories = list_mvtec_categories(mvtec_root)
    mask_categories = categories_with_masks(categories)

    if not mask_categories:
        st.warning(
            "ground_truth 폴더가 있는 데이터셋을 찾지 못했습니다. "
            "먼저 ROOT 경로와 데이터셋 구조를 확인하세요."
        )
    else:
        category_names = [path.name for path in mask_categories]

        selected_category_name = st.selectbox(
            "데이터셋 카테고리",
            category_names,
        )
        selected_category = next(
            path
            for path in mask_categories
            if path.name == selected_category_name
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            max_images_per_defect = st.number_input(
                "불량 유형별 최대 이미지 수",
                min_value=1,
                value=30,
                step=1,
                help="각 불량 폴더에서 최대 몇 장을 사용할지 정합니다.",
            )

        with col2:
            val_interval = st.number_input(
                "검증 이미지 선택 간격",
                min_value=2,
                value=5,
                step=1,
                help="5이면 약 20%, 4이면 약 25%, 10이면 약 10%가 검증용입니다.",
            )

        with col3:
            class_id = st.number_input(
                "클래스 번호",
                min_value=0,
                value=0,
                step=1,
            )

        class_name = st.text_input(
            "클래스 이름",
            value="defect",
        )

        output_folder_name = st.text_input(
            "YOLO 데이터셋 출력 폴더명",
            value="notebook_yolo_dataset",
        )

        work = (
            ROOT
            / "easy_code_samples"
            / "outputs"
            / output_folder_name
        )

        st.write("**생성 예정 위치**")
        st.code(str(work))

        overwrite = st.checkbox(
            "기존 출력 폴더가 있으면 삭제하고 다시 생성",
            value=False,
            help="체크하면 기존 폴더와 내부 파일이 즉시 삭제됩니다.",
        )

        if st.button(
            "YOLO 데이터셋 생성",
            type="primary",
            key="create_yolo_dataset",
        ):
            try:
                dataset_info = create_yolo_dataset(
                    category_path=selected_category,
                    output_dir=work,
                    max_images_per_defect=int(max_images_per_defect),
                    val_interval=int(val_interval),
                    class_id=int(class_id),
                    class_name=class_name.strip() or "defect",
                    overwrite=overwrite,
                )

                st.session_state.dataset_info = {
                    **(st.session_state.dataset_info or {}),
                    **dataset_info,
                    "selected_category": selected_category,
                }

                st.success("YOLO 데이터셋 생성이 완료되었습니다.")

            except Exception as error:
                st.exception(error)

        created_info = st.session_state.dataset_info

        if created_info and created_info.get("yaml_path"):
            metric_cols = st.columns(5)

            metric_cols[0].metric(
                "학습 이미지",
                created_info["train_count"],
            )
            metric_cols[1].metric(
                "검증 이미지",
                created_info["val_count"],
            )
            metric_cols[2].metric(
                "전체 생성",
                created_info["total_count"],
            )
            metric_cols[3].metric(
                "마스크 없음",
                created_info["missing_mask_count"],
            )
            metric_cols[4].metric(
                "빈 마스크",
                created_info["empty_mask_count"],
            )

            st.write("**data.yaml**")
            st.code(str(created_info["yaml_path"]))

            if created_info["defect_summary"]:
                st.write("**불량 유형별 생성 수**")
                st.json(created_info["defect_summary"])


# ============================================================
# 탭 3: YOLO 학습
# ============================================================

with tab_train:
    st.subheader("YOLO 모델 학습")

    default_yaml = (
        st.session_state.dataset_info.get("yaml_path")
        if st.session_state.dataset_info
        and st.session_state.dataset_info.get("yaml_path")
        else ROOT
        / "easy_code_samples"
        / "outputs"
        / "notebook_yolo_dataset"
        / "data.yaml"
    )

    yaml_text = st.text_input(
        "data.yaml 경로",
        value=str(default_yaml),
    )
    yaml_path = Path(yaml_text)

    col1, col2, col3 = st.columns(3)

    with col1:
        model_name = st.selectbox(
            "사전 학습 모델",
            [
                "yolov8n.pt",
                "yolov8s.pt",
                "yolo11n.pt",
                "yolo11s.pt",
            ],
            index=1,
            help=(
                "n은 가볍고 빠르며, s는 상대적으로 정확도가 높지만 "
                "학습 시간이 더 길어집니다."
            ),
        )

        epochs = st.number_input(
            "Epochs",
            min_value=1,
            value=50,
            step=1,
        )

    with col2:
        imgsz = st.selectbox(
            "입력 이미지 크기",
            [320, 416, 640, 800, 1024],
            index=2,
            help="작은 불량은 큰 입력 크기가 유리하지만 GPU 메모리 사용량이 증가합니다.",
        )

        batch = st.number_input(
            "Batch 크기",
            min_value=1,
            value=4,
            step=1,
            help="메모리 부족이 발생하면 4 → 2 → 1 순서로 낮추세요.",
        )

    with col3:
        workers = st.number_input(
            "DataLoader workers",
            min_value=0,
            value=0,
            step=1,
            help="Windows Streamlit 환경에서는 우선 0이 안정적입니다.",
        )

        amp = st.checkbox(
            "AMP 혼합 정밀도 사용",
            value=True,
            help="GPU 메모리 사용량과 학습 시간을 줄일 수 있습니다.",
        )

    training_project = (
        ROOT / "easy_code_samples" / "outputs" / "yolo_training"
    )

    run_name = st.text_input(
        "학습 실행 이름",
        value="screw_yolo_train",
    )

    train_device = get_train_device()

    st.info(
        "사용 장치: "
        + (
            f"GPU 0 — {torch.cuda.get_device_name(0)}"
            if train_device == 0
            else "CPU"
        )
    )

    st.warning(
        "학습 중에는 현재 Streamlit 화면이 대기 상태처럼 보일 수 있습니다. "
        "터미널에는 Epoch 진행 상황이 계속 출력됩니다."
    )

    if st.button(
        "YOLO 학습 시작",
        type="primary",
        key="start_training",
    ):
        try:
            with st.spinner(
                "YOLO 모델을 학습하고 있습니다. 터미널의 진행 로그도 확인하세요."
            ):
                train_info = train_yolo_model(
                    yaml_path=yaml_path,
                    model_name=model_name,
                    epochs=int(epochs),
                    imgsz=int(imgsz),
                    batch=int(batch),
                    workers=int(workers),
                    amp=amp,
                    device=train_device,
                    project_dir=training_project,
                    run_name=run_name.strip() or "screw_yolo_train",
                )

            st.session_state.train_info = train_info
            st.success("YOLO 학습이 완료되었습니다.")

        except Exception as error:
            st.exception(error)

    train_info = st.session_state.train_info

    if train_info:
        st.write("**학습 결과 저장 폴더**")
        st.code(str(train_info["save_dir"]))

        st.write("**가장 성능이 좋은 모델(best.pt)**")
        st.code(str(train_info["best_model_path"]))


# ============================================================
# 탭 4: 검증 예측
# ============================================================

with tab_predict:
    st.subheader("학습 모델로 검증 이미지 예측")

    default_best_model = (
        st.session_state.train_info["best_model_path"]
        if st.session_state.train_info
        else ROOT
        / "easy_code_samples"
        / "outputs"
        / "yolo_training"
        / "screw_yolo_train"
        / "weights"
        / "best.pt"
    )

    default_val_dir = (
        st.session_state.dataset_info["work"] / "images" / "val"
        if st.session_state.dataset_info
        and st.session_state.dataset_info.get("work")
        else ROOT
        / "easy_code_samples"
        / "outputs"
        / "notebook_yolo_dataset"
        / "images"
        / "val"
    )

    best_model_text = st.text_input(
        "best.pt 경로",
        value=str(default_best_model),
    )

    val_dir_text = st.text_input(
        "검증 이미지 폴더",
        value=str(default_val_dir),
    )

    col1, col2 = st.columns(2)

    with col1:
        confidence = st.slider(
            "검출 신뢰도 기준",
            min_value=0.01,
            max_value=1.00,
            value=0.05,
            step=0.01,
            help="오검출이 많으면 0.25 이상으로 올려보세요.",
        )

    with col2:
        prediction_imgsz = st.selectbox(
            "예측 입력 이미지 크기",
            [320, 416, 640, 800, 1024],
            index=2,
            key="prediction_imgsz",
        )

    prediction_project = (
        ROOT
        / "easy_code_samples"
        / "outputs"
        / "notebook_yolo_dataset"
        / "predictions"
    )

    prediction_name = st.text_input(
        "예측 결과 폴더명",
        value="result",
    )

    if st.button(
        "검증 이미지 예측 시작",
        type="primary",
        key="start_prediction",
    ):
        try:
            with st.spinner("검증 이미지를 예측하고 결과를 저장하고 있습니다."):
                prediction_info = predict_validation_images(
                    best_model_path=Path(best_model_text),
                    source_dir=Path(val_dir_text),
                    output_project=prediction_project,
                    output_name=prediction_name.strip() or "result",
                    confidence=float(confidence),
                    imgsz=int(prediction_imgsz),
                    device=get_train_device(),
                )

            st.session_state.prediction_info = prediction_info
            st.session_state.current_image_index = 0
            st.success("검증 이미지 예측이 완료되었습니다.")

        except Exception as error:
            st.exception(error)

    prediction_info = st.session_state.prediction_info

    if prediction_info:
        results = prediction_info["results"]
        result_dir = prediction_info["result_dir"]

        st.metric("검사한 이미지 수", len(results))

        st.write("**예측 결과 저장 위치**")
        st.code(str(result_dir))


# ============================================================
# 탭 5: 결과 확인
# ============================================================

with tab_result:
    st.subheader("예측 결과 이미지 및 검출 정보 확인")

    prediction_info = st.session_state.prediction_info

    manual_result_dir = st.text_input(
        "예측 결과 이미지 폴더",
        value=(
            str(prediction_info["result_dir"])
            if prediction_info
            else str(
                ROOT
                / "easy_code_samples"
                / "outputs"
                / "notebook_yolo_dataset"
                / "predictions"
                / "result"
            )
        ),
    )

    result_dir = Path(manual_result_dir)
    result_images = find_result_images(result_dir)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.metric("찾은 개별 결과 이미지", len(result_images))

    with col_b:
        if st.button("결과 폴더 열기", key="open_result_folder"):
            try:
                open_folder(result_dir)
            except Exception as error:
                st.error(str(error))

    if not result_images:
        st.warning(
            "결과 이미지를 찾지 못했습니다. "
            "검증 예측을 먼저 실행하거나 결과 폴더 경로를 확인하세요."
        )

    else:
        st.markdown("### 개별 이미지 보기")

        max_index = len(result_images) - 1
        current_index = min(
            st.session_state.current_image_index,
            max_index,
        )

        nav1, nav2, nav3, nav4, nav5 = st.columns(
            [1, 1, 2, 1, 1]
        )

        with nav1:
            if st.button("⏮ 처음", use_container_width=True):
                current_index = 0

        with nav2:
            if st.button("◀ 이전", use_container_width=True):
                current_index = max(0, current_index - 1)

        with nav3:
            selected_number = st.number_input(
                "이미지 번호",
                min_value=1,
                max_value=len(result_images),
                value=current_index + 1,
                step=1,
                label_visibility="collapsed",
            )
            current_index = int(selected_number) - 1

        with nav4:
            if st.button("다음 ▶", use_container_width=True):
                current_index = min(max_index, current_index + 1)

        with nav5:
            if st.button("마지막 ⏭", use_container_width=True):
                current_index = max_index

        st.session_state.current_image_index = current_index

        selected_image = result_images[current_index]

        st.image(
            str(selected_image),
            caption=(
                f"{current_index + 1} / {len(result_images)} — "
                f"{selected_image.name}"
            ),
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("### 여러 결과 이미지를 한 장으로 정리")

        st.info(
            "기존 YOLO 예측은 개별 이미지로 저장됩니다. "
            "아래 기능을 실행하면 개별 결과를 격자 형태의 한 장짜리 모음 이미지로 추가 저장합니다."
        )

        montage_col1, montage_col2, montage_col3 = st.columns(3)

        with montage_col1:
            montage_columns = st.number_input(
                "한 줄에 배치할 이미지 수",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
            )

        with montage_col2:
            montage_width = st.number_input(
                "개별 이미지 최대 너비",
                min_value=120,
                max_value=1000,
                value=360,
                step=20,
            )

        with montage_col3:
            montage_height = st.number_input(
                "개별 이미지 최대 높이",
                min_value=120,
                max_value=1000,
                value=280,
                step=20,
            )

        montage_path = result_dir / "prediction_contact_sheet.jpg"

        if st.button(
            "한 장짜리 결과 모음 이미지 생성",
            type="primary",
            key="create_montage",
        ):
            try:
                created_montage = create_contact_sheet(
                    image_paths=result_images,
                    output_path=montage_path,
                    columns=int(montage_columns),
                    thumb_width=int(montage_width),
                    thumb_height=int(montage_height),
                )
                st.success(f"모음 이미지를 저장했습니다: {created_montage}")
            except Exception as error:
                st.exception(error)

        if montage_path.exists():
            st.image(
                str(montage_path),
                caption="전체 예측 결과 모음",
                use_container_width=True,
            )

            with montage_path.open("rb") as file:
                st.download_button(
                    "모음 이미지 다운로드",
                    data=file,
                    file_name=montage_path.name,
                    mime="image/jpeg",
                )

        st.markdown("---")
        st.markdown("### 검출 박스 정보")

        if prediction_info and prediction_info.get("results"):
            display_limit = st.number_input(
                "정보를 표시할 이미지 수",
                min_value=1,
                max_value=len(prediction_info["results"]),
                value=min(5, len(prediction_info["results"])),
                step=1,
            )

            records = result_box_records(
                prediction_info["results"],
                limit=int(display_limit),
            )

            st.dataframe(
                records,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "x1, y1은 박스 왼쪽 위 좌표이고 x2, y2는 오른쪽 아래 좌표입니다."
            )
        else:
            st.info(
                "현재 실행 세션의 예측 결과 객체가 없습니다. "
                "박스 신뢰도와 좌표를 보려면 4번 탭에서 예측을 실행하세요. "
                "저장된 이미지만 불러온 경우에는 이미지 확인과 모음 이미지 생성만 가능합니다."
            )
