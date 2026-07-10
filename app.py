"""
GitHub / Streamlit Cloud 배포용 진입점 (이 파일 1개만 실행).

실행: streamlit run app.py
Main file path (Streamlit Cloud): app.py

데이터 원본: data/event_log.csv
  → GitHub에서 CSV를 수정 후 push 하면 앱 재배포 시 자동 반영됩니다.

로컬 PC에서는 기존 파일을 그대로 사용하세요.
  python "드릴 파손 카운트.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
_MAIN_SCRIPT = _APP_DIR / "드릴 파손 카운트.py"
_EVENT_CSV = _APP_DIR / "data" / "event_log.csv"

if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


def _install_csv_data_source() -> None:
    """BreakageTracker 생성 시 저장소 CSV를 이벤트 로그 소스로 사용."""
    from breakage_tracker import BreakageTracker

    _orig_init = BreakageTracker.__init__

    def _init_with_csv(self, data_path: Path, *_args, **_kwargs) -> None:
        _orig_init(self, data_path)
        if not _EVENT_CSV.exists():
            return
        mtime = _EVENT_CSV.stat().st_mtime
        if st.session_state.get("event_csv_mtime") == mtime:
            return
        self.replace_from_event_csv(_EVENT_CSV)
        st.session_state.event_csv_mtime = mtime

    BreakageTracker.__init__ = _init_with_csv  # type: ignore[method-assign]


def _run_main_app() -> None:
    if not _MAIN_SCRIPT.exists():
        st.error(f"메인 스크립트를 찾을 수 없습니다: {_MAIN_SCRIPT.name}")
        st.stop()
    source = _MAIN_SCRIPT.read_text(encoding="utf-8")
    code = compile(source, str(_MAIN_SCRIPT), "exec")
    exec(code, {"__name__": "__streamlit__", "__file__": str(_MAIN_SCRIPT)})


_install_csv_data_source()
_run_main_app()
