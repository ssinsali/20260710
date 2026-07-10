ChatGPT Plus








오늘 오후 1:07
tensor([[-5.2859e-01,  7.2061e-01, -8.4023e-01, -2.7197e-02, -3.6002e-01,
          1.5844e+00,  7.7546e-01, -5.2523e-01,  1.1261e+00, -1.1343e+00],
        [ 2.1166e-01, -1.1468e-01, -1.2342e+00,  3.1759e-01, -2.1307e-01,
          4.3343e-01, -8.4606e-01,  8.4971e-01,  7.5319e-02, -2.1639e-01],
        [ 3.2463e-01, -1.3989e+00, -4.6666e-01,  7.6455e-01,  1.2484e-02,
         -9.8920e-01, -1.8561e+00, -2.9582e-01,  5.4987e-01, -8.3207e-03],
        [ 1.7319e+00,  7.9282e-02, -1.4089e+00,  7.7606e-01,  4.1472e-01,
          4.4303e-01, -1.2733e-01,  1.0533e+00, -3.3615e-01,  4.8253e-01],
        [ 1.1455e+00,  1.2798e+00,  5.0352e-02, -1.7343e-01,  1.7567e+00,
         -8.7226e-04,  1.1256e+00,  1.4179e+00,  1.2947e+00, -7.9190e-01],
        [-7.2030e-01, -7.4252e-01, -6.1159e-01, -3.9859e-01, -1.3082e-01,
          5.8267e-02, -4.2801e-01, -9.3092e-02,  4.5628e-02, -1.5612e+00],
        [ 5.0075e-01,  1.1678e+00, -1.1927e+00, -2.1579e+00,  2.7672e-01,
          2.0281e-01, -1.6005e+00, -5.8652e-01,  2.2948e+00,  1.4169e-03],
        [ 1.2921e+00,  7.4553e-01,  1.0692e+00,  2.5072e-01, -5.5736e-01,
         -5.4389e-01, -1.0248e+00, -1.1759e+00, -1.0230e+00, -8.5682e-02],
        [-4.6934e-01,  4.9813e-02, -8.2421e-01,  1.2738e-01, -7.3231e-01,
          4.9694e-01,  7.8397e-01,  7.8466e-01, -5.2119e-01, -2.4340e+00],
        [-1.4243e+00, -8.6748e-01,  1.2735e+00,  3.7627e-01, -8.1527e-01,
          2.5990e-01,  2.1126e+00,  4.7153e-01,  2.2727e+00,  4.7587e-01]],
       device='cuda:0')


붙여넣은 텍스트 (1)(3).txt
문서
내 pc에서 돌렸는데 이렇게 에러가 나와.

여기이후에 어떻게 해?

스트림릿에서 만들기해서 가져올때 이렇게 됬어


맞습니다. 지금 구조라면 학습까지 한 뒤, 학습된 best.pt 모델을 저장하고 나중에 다시 불러와서 새 이미지 검사에 사용하는 기능이 반드시 있어야 합니다.

현재 프로그램은 학습 직후 같은 실행 세션 안에서는 best.pt를 찾아 예측할 수 있지만, Streamlit Cloud에서는 앱이 재부팅되거나 세션이 종료되면 서버 내부 파일이 사라질 수 있습니다. 그래서 실제 사용을 위해서는 아래 두 기능이 필요합니다.

필요한 기능
1. 학습 완료 모델 다운로드
학습이 끝나면 생성되는 파일:

weights/best.pt
를 Streamlit 화면에서 바로 다운로드할 수 있어야 합니다.

예:

with open(best_model_path, "rb") as file:
    st.download_button(
        "학습 완료 모델 다운로드",
        data=file,
        file_name="best.pt",
        mime="application/octet-stream",
    )
이렇게 내려받아 PC에 보관하면 됩니다.

2. 저장한 모델 다시 업로드
다음에 앱을 열었을 때 학습을 다시 하지 않고, PC에 저장한 best.pt 파일을 올려서 새 이미지를 검사할 수 있어야 합니다.

예:

uploaded_model = st.file_uploader(
    "학습 완료 모델 업로드",
    type=["pt"],
)
업로드 후 서버 임시 폴더에 저장합니다.

model_path = ROOT / "uploaded_models" / uploaded_model.name
model_path.parent.mkdir(parents=True, exist_ok=True)
model_path.write_bytes(uploaded_model.getbuffer())
그다음:

trained_model = YOLO(str(model_path))
로 불러오면 됩니다.

실제 프로그램 흐름
지금 프로그램은 아래 구조로 바뀌는 것이 맞습니다.

1. 데이터셋 ZIP 등록
2. YOLO 데이터셋 생성
3. 모델 학습
4. best.pt 다운로드
5. 나중에 best.pt 다시 업로드
6. 검사할 새 이미지 업로드
7. 업로드한 이미지 예측
8. 결과 이미지와 신뢰도 확인
즉, 학습용 프로그램과 검사용 프로그램이 하나로 합쳐진 형태입니다.

중요한 점
Streamlit Cloud에 학습 결과를 그냥 저장해두는 방식은 안전하지 않습니다.

/mount/src/...
아래에 생성된 파일은 앱 재부팅, 재배포, 서버 초기화 시 사라질 수 있습니다.

따라서 가장 현실적인 방식은:

학습 완료
→ best.pt 다운로드
→ PC 또는 회사 서버에 보관
→ 검사할 때 best.pt 업로드
입니다.

실무에서는 모델 파일을 아래처럼 버전 관리하는 편이 좋습니다.

models
├── screw_yolo_v1_20260710.pt
├── screw_yolo_v2_20260715.pt
└── screw_yolo_v3_20260720.pt
파일명에 다음 정보를 넣으면 좋습니다.

제품명_모델버전_학습일자.pt
예:

screw_defect_v1_20260710.pt
현재 통합 코드에는 이 부분이 빠져 있으므로, 다음 수정에서는 최소한 아래를 넣어야 합니다.

학습 완료 best.pt 다운로드 버튼

기존 best.pt 업로드 기능

검사할 이미지 여러 장 업로드

업로드한 모델로 새 이미지 예측

예측 결과 ZIP 다운로드

결과 모음 이미지 다운로드

모델명과 학습일자 표시

김그룹장님이 이해한 방향이 맞습니다. 지금 코드는 학습 흐름은 있지만, 학습 결과를 보존하고 재사용하는 실사용 단계가 아직 완성되지 않은 상태입니다.


아니 깃허브에 저장되게 해줘야 하는거 아니야?

맞습니다. 학습이 끝난 best.pt를 GitHub 저장소에 자동으로 저장하도록 만들 수 있습니다.

다만 Streamlit Cloud가 GitHub에 파일을 쓰려면 권한이 필요합니다. 앱 코드에 비밀번호처럼 토큰을 직접 적으면 안 되고, GitHub 토큰을 Streamlit의 Secrets에 넣은 뒤 GitHub REST API로 best.pt를 커밋해야 합니다. GitHub의 Repository Contents API는 저장소 파일 생성·교체를 지원합니다. 
 Streamlit Community Cloud는 비밀키를 코드 밖에서 관리하는 Secrets 기능을 제공합니다. 

최종 동작 방식
YOLO 학습 완료
    ↓
best.pt 생성
    ↓
GitHub 저장 버튼 클릭
    ↓
models/screw_defect_v1.pt로 GitHub에 업로드
    ↓
다음 실행 때 GitHub의 모델을 자동으로 내려받아 예측
저장소 구조는 이렇게 됩니다.

20260710
├── app.py
├── requirements.txt
├── README.md
└── models
    ├── screw_defect_v1.pt
    └── screw_defect_v2.pt
다만 모델 크기를 확인해야 합니다
일반 Git 저장소는 단일 파일을 100MB까지 강제로 제한하며, 큰 파일은 Git LFS 사용이 권장됩니다. 

보통 YOLO의 대략적인 모델 크기는 선택한 모델에 따라 다릅니다.

YOLOv8n best.pt → 비교적 작음
YOLOv8s best.pt → 보통 수십 MB 수준
더 큰 모델       → 100MB를 넘을 가능성 있음
따라서 현재 사용 중인 yolov8s.pt 기반 모델은 일반 GitHub 업로드가 가능한 경우가 많지만, 실제 생성된 best.pt 크기를 검사한 뒤 업로드하도록 해야 합니다.

필요한 설정
1. GitHub 토큰 만들기
GitHub에서 Fine-grained personal access token을 만들고, 대상 저장소 20260710에 대해 최소한 다음 권한을 줍니다.

Repository access:
ssinsali/20260710

Permissions:
Contents → Read and write
토큰은 GitHub 코드나 requirements.txt에 적지 않습니다.

2. Streamlit Secrets에 등록
Streamlit 앱 오른쪽 아래에서:

Manage app
→ Settings
→ Secrets
아래 내용을 등록합니다.

GITHUB_TOKEN = "발급받은_토큰"
GITHUB_OWNER = "ssinsali"
GITHUB_REPO = "20260710"
GITHUB_BRANCH = "main"
Streamlit 앱 설정의 Secrets에서 이 값을 저장할 수 있습니다. 

코드에 추가할 핵심 기능
import base64
import requests
import streamlit as st
from pathlib import Path


def upload_model_to_github(
    local_model_path: Path,
    github_path: str,
    commit_message: str,
):
    token = st.secrets["GITHUB_TOKEN"]
    owner = st.secrets["GITHUB_OWNER"]
    repo = st.secrets["GITHUB_REPO"]
    branch = st.secrets.get("GITHUB_BRANCH", "main")

    api_url = (
        f"https://api.github.com/repos/"
        f"{owner}/{repo}/contents/{github_path}"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    file_bytes = local_model_path.read_bytes()
    encoded_content = base64.b64encode(file_bytes).decode("utf-8")

    # 같은 파일이 이미 있으면 SHA를 받아와 교체합니다.
    existing_response = requests.get(
        api_url,
        headers=headers,
        params={"ref": branch},
        timeout=30,
    )

    payload = {
        "message": commit_message,
        "content": encoded_content,
        "branch": branch,
    }

    if existing_response.status_code == 200:
        payload["sha"] = existing_response.json()["sha"]

    response = requests.put(
        api_url,
        headers=headers,
        json=payload,
        timeout=120,
    )

    if response.status_code not in (200, 201):
        raise RuntimeError(
            f"GitHub 저장 실패\n"
            f"상태 코드: {response.status_code}\n"
            f"응답: {response.text}"
        )

    return response.json()
학습 완료 화면에 다음 버튼을 붙이면 됩니다.

model_filename = st.text_input(
    "GitHub에 저장할 모델 파일명",
    value="screw_defect_v1.pt",
)

if st.button("학습 모델을 GitHub에 저장"):
    best_model_path = st.session_state.train_info["best_model_path"]

    file_size_mb = best_model_path.stat().st_size / 1024**2

    if file_size_mb >= 100:
        st.error(
            f"모델 크기가 {file_size_mb:.1f}MB입니다. "
            "일반 GitHub 저장 한도 100MB를 초과합니다."
        )
    else:
        result = upload_model_to_github(
            local_model_path=best_model_path,
            github_path=f"models/{model_filename}",
            commit_message=f"학습 모델 저장: {model_filename}",
        )

        st.success("학습 모델을 GitHub에 저장했습니다.")
GitHub 모델 다시 불러오기
앱 시작 시 저장된 모델을 내려받아 예측에 사용할 수도 있습니다.

def download_model_from_github(
    github_path: str,
    local_path: Path,
):
    token = st.secrets["GITHUB_TOKEN"]
    owner = st.secrets["GITHUB_OWNER"]
    repo = st.secrets["GITHUB_REPO"]
    branch = st.secrets.get("GITHUB_BRANCH", "main")

    api_url = (
        f"https://api.github.com/repos/"
        f"{owner}/{repo}/contents/{github_path}"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.raw+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(
        api_url,
        headers=headers,
        params={"ref": branch},
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"GitHub 모델 불러오기 실패: {response.status_code}"
        )

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(response.content)

    return local_path
그리고:

model_path = download_model_from_github(
    github_path="models/screw_defect_v1.pt",
    local_path=ROOT / "downloaded_models" / "screw_defect_v1.pt",
)

trained_model = YOLO(str(model_path))
로 사용할 수 있습니다.

꼭 알아둘 점
GitHub에 모델을 저장하면 앱이 다시 배포될 수 있습니다. 저장소에 새 커밋이 생기기 때문입니다. 따라서 학습 완료 후 저장 버튼을 누르면 잠시 후 Streamlit 앱이 재시작될 수 있습니다.

또한 requests를 사용하므로 requirements.txt에 아래 줄도 넣는 것이 안전합니다.

requests
이번 수정 방향은 다음처럼 잡는 것이 맞습니다.

데이터셋 ZIP 등록
→ 모델 학습
→ best.pt GitHub 저장
→ GitHub 모델 목록 조회
→ 저장된 모델 선택
→ 새 검사 이미지 업로드
→ 선택한 모델로 예측
현재 앱을 이 구조로 다시 통합해야 합니다.


그럼 스트리밋에서는 vs코드에서 학습된 결과의 모델을 깃허브에서 파일을 연동해서 쓰게 바꿔주고, 각 파일 이미지를 업로드해서 결과를 보도록 전체 프로그램을 수정해줘.

전체 프로그램을 학습 기능 없이, GitHub에 저장된 학습 완료 모델을 불러와 이미지 검사만 하는 구조로 다시 만들었습니다.

app.py

requirements.txt

GitHub 저장소는 아래 구조로 맞추면 됩니다.

20260710
├── app.py
├── requirements.txt
└── models
    └── best.pt
VSCode에서 학습한 모델은 보통 다음 위치에 있습니다.

runs/detect/train/weights/best.pt
또는 기존 코드 기준으로:

easy_code_samples/outputs/yolo_training/.../weights/best.pt
이 best.pt를 GitHub 저장소의 models 폴더에 올리면 됩니다.

앱에서는 자동으로:

models/best.pt
를 찾아 왼쪽 모델 선택 메뉴에 표시합니다.

새 프로그램의 흐름은 다음과 같습니다.

GitHub의 best.pt 모델 불러오기
→ 검사 이미지 여러 장 업로드
→ YOLO 검사 실행
→ 개별 결과 확인
→ 전체 결과 모음 확인
→ 검출 좌표·신뢰도 표 확인
→ 결과 ZIP 다운로드
결과 ZIP에는 아래 파일들이 포함됩니다.

개별 검출 결과 이미지
prediction_contact_sheet.jpg
detection_results.csv
GitHub에 기존 app.py와 requirements.txt를 새 파일로 교체하고, models/best.pt를 추가한 뒤 커밋하면 Streamlit 앱에 반영됩니다. 


라이브러리
/
app.py


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
                "annotated_images": annotated_images,
                "filenames": filenames,
                "detection_rows": detection_rows,
                "model_path": str(selected_model_path),
            }

        st.success("이미지 검사가 완료되었습니다.")

    except Exception as error:
        st.exception(error)


# ============================================================
# 5. 검사 결과
# ============================================================

saved_results = st.session_state.get("inference_results")

if saved_results:
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

        ### 주의

        일반 GitHub 저장소는 단일 파일 100MB 제한이 있습니다.
        작은 YOLO 모델은 보통 저장할 수 있지만, 큰 모델은
        GitHub Release나 별도 모델 저장소 사용이 필요할 수 있습니다.
        """
    )
