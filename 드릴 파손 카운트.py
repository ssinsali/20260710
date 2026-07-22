"""
드릴 설비별 파손(빨간색) 발생 횟수를 추적·시각화합니다.
실행: python "드릴 파손 카운트.py"  (브라우저 자동 열림)
"""
from __future__ import annotations

import os
from datetime import date
import sys
from io import StringIO
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from breakage_stats import (
    breakage_type_options,
    count_by_column,
    count_drill_lot_grouped,
    daily_trend,
    drill_lot_group_detail,
    drill_usage_by_equipment,
    equipment_counts,
    equipment_counts_dataframe,
    filter_by_breakage_type,
    filter_by_work_end_range,
    format_year_month,
    iter_year_months,
    monthly_trend,
    month_bounds_to_dates,
    product_equipment_matrix,
    records_to_dataframe,
    summary_metrics,
    work_end_month_bounds,
    yearly_trend,
)
from breakage_tracker import BreakageTracker
from data_import import (
    event_log_template_dataframe,
    is_event_log_format,
    parse_count_table,
    parse_event_log_table,
    read_table_file,
    template_dataframe,
)
from equipment_layout import (
    ALL_EQUIPMENT_IDS,
    DENSE_COLS,
    DRA_DIVIDER_AFTER,
    DRA_EXTRA_GAP_AFTER,
    DRA_GRID,
    DRA_ROW_SPACING,
    FPAS3_SIDEBAR_SLOTS,
    MAP_LEGEND_ITEMS,
    STANDARD_COLS,
    STATUS_COLORS,
    STATUS_LABELS,
    EquipmentStatus,
    effective_equipment_status,
    sidebar_occupies_row,
)
from image_analyzer import analyze_dashboard_image

_DATA_PATH = _APP_DIR / "data" / "breakage_data.json"

st.set_page_config(page_title="드릴 파손 카운트", page_icon="🔴", layout="wide")


def _get_tracker() -> BreakageTracker:
    if "tracker" not in st.session_state:
        st.session_state.tracker = BreakageTracker(_DATA_PATH)
    return st.session_state.tracker


def _shutdown_app(tracker: BreakageTracker) -> None:
    """Streamlit 서버 프로세스 종료."""
    tracker.save()
    os._exit(0)


def _cell_html(eq_id: str, counts: dict[str, int]) -> str:
    count = counts.get(eq_id, 0)
    display = effective_equipment_status(count, EquipmentStatus.NORMAL.value)
    color = STATUS_COLORS[display]
    count_html = (
        f'<div class="cnt" style="color:{color};">×{count}</div>' if count > 0 else ""
    )
    return f"""
    <div class="cell">
        <div class="dot" style="background:{color};"></div>
        <div class="lbl">{eq_id}</div>
        {count_html}
    </div>
    """


def _side_items_html(items: list[dict[str, str]], counts: dict[str, int]) -> str:
    parts: list[str] = []
    for item in items:
        if item.get("type") == "equipment":
            parts.append(_cell_html(item["id"], counts))
        elif item.get("type") == "label":
            parts.append(f'<div class="side-label">{item["text"]}</div>')
    return "".join(parts)


def _render_dashboard_map(counts: dict[str, int]) -> str:
    board: list[str] = []

    def _row_html(row: list[str | None], spacing: str) -> str:
        cols = DENSE_COLS if spacing == "dense" else STANDARD_COLS
        cells: list[str] = []
        for col_idx in range(cols):
            eq_id = row[col_idx] if col_idx < len(row) else None
            if eq_id is None:
                cells.append('<div class="cell empty"></div>')
            else:
                cells.append(_cell_html(eq_id, counts))
        return (
            f'<div class="dra-row {spacing}" '
            f'style="grid-template-columns:repeat({cols},1fr)">'
            f'{"".join(cells)}</div>'
        )

    for row_idx, row in enumerate(DRA_GRID):
        spacing = DRA_ROW_SPACING[row_idx] if row_idx < len(DRA_ROW_SPACING) else "standard"
        slot_start = sidebar_occupies_row(row_idx)

        divider_html = ""
        if row_idx < len(DRA_DIVIDER_AFTER) and DRA_DIVIDER_AFTER[row_idx]:
            thick = " thick" if row_idx == 8 else ""
            divider_html = f'<div class="row-divider{thick}"></div>'

        extra_gap_html = ""
        if row_idx < len(DRA_EXTRA_GAP_AFTER) and DRA_EXTRA_GAP_AFTER[row_idx]:
            extra_gap_html = '<div class="row-extra-gap"></div>'

        board.append(
            f'<div class="main-cell">{_row_html(row, spacing)}{divider_html}{extra_gap_html}</div>'
        )

        if slot_start == row_idx:
            span, items, valign = FPAS3_SIDEBAR_SLOTS[row_idx]
            board.append(
                f'<div class="side-cell" style="grid-row:span {span}">'
                f'<div class="side-inner side-valign-{valign}">'
                f'{_side_items_html(items, counts)}</div></div>'
            )
        elif slot_start is None:
            board.append('<div class="side-cell side-empty"></div>')

    return f"""
  <style>
    .dash-wrap {{
      background: #1a1a1a;
      padding: 10px 12px;
      border-radius: 6px;
      font-family: sans-serif;
      width: 100%;
      box-sizing: border-box;
    }}
    .dash-board {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 76px;
      width: 100%;
      column-gap: 0;
      align-items: stretch;
    }}
    .main-cell {{
      min-width: 0;
    }}
    .dra-row {{
      display: grid;
      width: 100%;
    }}
    .bottom-block {{ width: 100%; }}
    .row-divider {{
      height: 1px;
      background: #6b7280;
      margin: 3px 0 4px 0;
      opacity: 0.85;
      width: 100%;
    }}
    .row-divider.thick {{
      height: 2px;
      margin: 5px 0 6px 0;
      background: #9ca3af;
    }}
    .row-extra-gap {{
      height: 22px;
      width: 100%;
    }}
    .side-cell {{
      border-left: 1px solid #6b7280;
      display: flex;
      align-items: stretch;
      justify-content: center;
      padding: 0 4px;
      min-height: 30px;
    }}
    .side-cell.side-empty {{
      border-left: 1px solid #6b7280;
      min-height: 4px;
    }}
    .side-inner {{
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 3px;
      width: 100%;
      flex: 1;
      min-height: 0;
    }}
    .side-inner.side-valign-start {{ justify-content: flex-start; }}
    .side-inner.side-valign-end {{ justify-content: flex-end; }}
    .side-inner.side-valign-center {{ justify-content: center; }}
    .cell {{
      text-align: center;
      padding: 3px 0;
      box-sizing: border-box;
      min-width: 0;
    }}
    .cell.empty {{ visibility: hidden; }}
    .dot {{
      width: 16px;
      height: 16px;
      margin: 0 auto;
      border: 1px solid #444;
      border-radius: 2px;
    }}
    .lbl {{
      font-size: 11px;
      color: #d1d5db;
      margin-top: 3px;
      white-space: nowrap;
    }}
    .cnt {{
      font-size: 9px;
      color: #fbbf24;
    }}
    .side-label {{
      font-size: 10px;
      color: #d1d5db;
      text-align: center;
      line-height: 1.2;
    }}
    .side-cell .cell {{ width: 100%; }}
  </style>
  <div class="dash-wrap">
    <div class="dash-board">{"".join(board)}</div>
  </div>
  """


def _detail_df(tracker: BreakageTracker) -> pd.DataFrame:
    return records_to_dataframe(tracker.state.detail_records)


def _filtered_by_breakage_type(tracker: BreakageTracker) -> pd.DataFrame:
    raw = _detail_df(tracker)
    mode = st.session_state.get("breakage_type_mode", "전체 포함")
    selected = st.session_state.get("breakage_type_selected", [])
    return filter_by_breakage_type(raw, mode, selected)


def _display_detail_df(tracker: BreakageTracker) -> pd.DataFrame:
    """파손 형태 + 조회 기간 필터가 적용된 표시용 데이터."""
    df = _filtered_by_breakage_type(tracker)
    start = st.session_state.get("global_period_start")
    end = st.session_state.get("global_period_end")
    if start is not None and end is not None:
        return filter_by_work_end_range(df, start, end)
    return df


def _filtered_equipment_counts(tracker: BreakageTracker) -> dict[str, int]:
    return equipment_counts(_display_detail_df(tracker))


def _breakage_type_filter_active() -> bool:
    mode = st.session_state.get("breakage_type_mode", "전체 포함")
    if mode == "전체 포함":
        return False
    return bool(st.session_state.get("breakage_type_selected"))


def _period_filter_active() -> bool:
    return not st.session_state.get("global_period_full", True)


def _display_filter_active() -> bool:
    return _breakage_type_filter_active() or _period_filter_active()


def _show_period_filter_banner() -> None:
    if st.session_state.get("global_period_full", True):
        return
    start = st.session_state.get("global_period_start")
    end = st.session_state.get("global_period_end")
    if start is None or end is None:
        return
    st.info(f"조회 기간 필터: **{start.strftime('%Y년 %m월')}** ~ **{end.strftime('%Y년 %m월')}**")


def _empty_filter_reason(tracker: BreakageTracker) -> str:
    if _detail_df(tracker).empty:
        return "no_data"
    if _filtered_by_breakage_type(tracker).empty:
        return "type"
    if _display_detail_df(tracker).empty:
        return "period"
    return ""


def _show_empty_filter_message(tracker: BreakageTracker) -> None:
    reason = _empty_filter_reason(tracker)
    if reason == "no_data":
        st.info(
            "이벤트 로그가 없습니다. **데이터 입력 → 이벤트 로그 업로드**에서 CSV를 반영하세요."
        )
    elif reason == "type":
        st.info("현재 파손 형태 필터 조건에 맞는 이벤트가 없습니다. 사이드바 필터를 확인하세요.")
    elif reason == "period":
        st.info("현재 조회 기간에 해당하는 이벤트가 없습니다. 상단 기간 설정을 확인하세요.")


def _show_breakage_type_filter_banner() -> None:
    mode = st.session_state.get("breakage_type_mode", "전체 포함")
    if mode == "전체 포함":
        return
    selected = st.session_state.get("breakage_type_selected", [])
    if mode == "선택만" and not selected:
        st.warning("파손 형태 **선택만** 모드입니다. 사이드바에서 표시할 형태를 선택하세요.")
        return
    if not selected and mode == "선택 제외":
        return
    labels = ", ".join(selected) if selected else "(없음)"
    if mode == "선택 제외":
        st.info(f"파손 형태 필터: **제외** — {labels}")
    else:
        st.info(f"파손 형태 필터: **선택만** — {labels}")


def _breakage_metrics(tracker: BreakageTracker) -> dict[str, int | float | bool]:
    """상단 요약 — 필터 적용 이벤트 로그 기준."""
    detail_df = _display_detail_df(tracker)
    sm = summary_metrics(detail_df)
    total_eq = len(ALL_EQUIPMENT_IDS)
    return {
        "total_events": sm["total_events"],
        "broken_equipment": sm["equipment_count"] if not detail_df.empty else 0,
        "intact_equipment": total_eq - (sm["equipment_count"] if not detail_df.empty else 0),
        "breakage_rate_pct": (
            (sm["equipment_count"] / total_eq * 100) if total_eq and not detail_df.empty else 0.0
        ),
        "has_events": not detail_df.empty,
    }


def _chart_top_n_selector() -> int | None:
    """1~10 또는 전체. 전체일 때 None."""
    choice = st.select_slider(
        "차트 상위 N개",
        options=[*range(1, 11), "전체"],
        value=10,
        key="stats_top_n",
    )
    if choice == "전체":
        st.caption("전체 항목 표시")
        return None
    return int(choice)


def _use_horizontal_bar(plot_df: pd.DataFrame, category_col: str) -> bool:
    if category_col not in plot_df.columns or plot_df.empty:
        return False
    lengths = plot_df[category_col].astype(str).str.len()
    return float(lengths.mean()) > 10 or int(lengths.max()) > 16


def _stats_bar_chart(
    data: pd.DataFrame,
    category_col: str,
    value_col: str = "파손 건수",
) -> None:
    plot_df = data[[category_col, value_col]].copy()
    n = len(plot_df)
    if n == 0:
        return
    horizontal = _use_horizontal_bar(plot_df, category_col)
    if horizontal:
        height = max(360, min(900, n * 28 + 80))
        chart = (
            alt.Chart(plot_df)
            .mark_bar(color="#60a5fa")
            .encode(
                y=alt.Y(
                    f"{category_col}:N",
                    sort="-x",
                    axis=alt.Axis(labelFontSize=13, labelLimit=0, title=None),
                ),
                x=alt.X(f"{value_col}:Q", title=value_col),
                tooltip=[category_col, value_col],
            )
            .properties(height=height)
        )
    else:
        chart = (
            alt.Chart(plot_df)
            .mark_bar(color="#60a5fa")
            .encode(
                x=alt.X(
                    f"{category_col}:N",
                    sort="-y",
                    axis=alt.Axis(
                        labelAngle=-45,
                        labelFontSize=13,
                        labelLimit=0,
                        labelOverlap=False,
                        title=None,
                    ),
                ),
                y=alt.Y(
                    f"{value_col}:Q",
                    title=value_col,
                    scale=alt.Scale(zero=True, nice=True),
                ),
                tooltip=[category_col, value_col],
            )
            .properties(height=340)
        )
    st.altair_chart(chart, use_container_width=True)


def _period_bar_chart(data: pd.DataFrame, category_col: str, value_col: str = "파손 건수") -> None:
    if data.empty:
        return
    plot_df = data[[category_col, value_col]].copy()
    label_angle = -45 if len(plot_df) > 8 else 0
    chart = (
        alt.Chart(plot_df)
        .mark_bar(color="#60a5fa")
        .encode(
            x=alt.X(
                f"{category_col}:N",
                sort=None,
                axis=alt.Axis(
                    labelAngle=label_angle,
                    labelFontSize=12,
                    labelLimit=0,
                    title=None,
                ),
            ),
            y=alt.Y(
                f"{value_col}:Q",
                title=value_col,
                scale=alt.Scale(zero=True, nice=True),
            ),
            tooltip=[category_col, value_col],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_global_period_selector(detail_df: pd.DataFrame) -> tuple[date, date] | None:
    """제목 아래 전역 조회 기간 (년·월)."""
    bounds = work_end_month_bounds(detail_df)
    if bounds is None:
        st.session_state.pop("global_period_start", None)
        st.session_state.pop("global_period_end", None)
        st.session_state["global_period_full"] = True
        st.caption("작업종료 일시가 없어 조회 기간을 설정할 수 없습니다.")
        return None

    min_ym, max_ym = bounds
    month_list = iter_year_months(min_ym, max_ym)
    labels = [format_year_month(y, m) for y, m in month_list]
    ym_by_label = {label: ym for label, ym in zip(labels, month_list)}

    if len(labels) == 1:
        start, end = month_bounds_to_dates(month_list[0], month_list[0])
        st.caption(f"조회 기간: {labels[0]}")
        is_full = True
    else:
        start_label, end_label = st.select_slider(
            "조회 기간 (년·월)",
            options=labels,
            value=(labels[0], labels[-1]),
            key="global_period_range",
        )
        start, end = month_bounds_to_dates(ym_by_label[start_label], ym_by_label[end_label])
        st.caption(f"선택 기간: {start_label} ~ {end_label}")
        is_full = start_label == labels[0] and end_label == labels[-1]

    st.session_state["global_period_start"] = start
    st.session_state["global_period_end"] = end
    st.session_state["global_period_full"] = is_full
    return start, end


def _render_breakage_trends(detail_df: pd.DataFrame) -> None:
    if detail_df.empty:
        st.caption("표시할 추이 데이터가 없습니다.")
        return
    start = st.session_state.get("global_period_start")
    end = st.session_state.get("global_period_end")

    st.subheader("년도별 파손 추이")
    yearly = yearly_trend(detail_df, range_start=start, range_end=end)
    if yearly.empty:
        st.caption("해당 기간 데이터가 없습니다.")
    else:
        _period_bar_chart(yearly, "년")

    st.subheader("월별 파손 추이")
    monthly = monthly_trend(detail_df, range_start=start, range_end=end)
    if monthly.empty:
        st.caption("해당 기간 데이터가 없습니다.")
    else:
        _period_bar_chart(monthly, "기간")

    st.subheader("일별 파손 추이")
    daily = daily_trend(detail_df, range_start=start, range_end=end)
    if daily.empty:
        st.caption("해당 기간 데이터가 없습니다.")
    else:
        _daily_trend_chart(daily)


def _daily_month_year_axis(chart_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = chart_df.copy()
    tmp["date_ts"] = pd.to_datetime(tmp["date"])
    month_rows: list[dict] = []
    for (_, month), grp in tmp.groupby([tmp["date_ts"].dt.year, tmp["date_ts"].dt.month], sort=True):
        start = grp["date_ts"].min()
        end = grp["date_ts"].max()
        mid = start + (end - start) / 2
        month_rows.append({"date": mid.strftime("%Y-%m-%d"), "월": f"{int(month)}월"})
    year_rows: list[dict] = []
    for year, grp in tmp.groupby(tmp["date_ts"].dt.year, sort=True):
        start = grp["date_ts"].min()
        end = grp["date_ts"].max()
        mid = start + (end - start) / 2
        year_rows.append({"date": mid.strftime("%Y-%m-%d"), "년": str(int(year))})
    return pd.DataFrame(month_rows), pd.DataFrame(year_rows)


def _daily_trend_chart(data: pd.DataFrame) -> None:
    plot_df = data.copy()
    if plot_df.empty:
        return
    chart_df = pd.DataFrame(
        {
            "date": pd.to_datetime(plot_df["날짜_원본"]).dt.strftime("%Y-%m-%d"),
            "일": plot_df["일"].astype(str),
            "날짜": plot_df["날짜"].astype(str),
            "파손 건수": plot_df["파손 건수"],
        }
    )
    ymax = float(chart_df["파손 건수"].max())
    ymax = ymax * 1.12 if ymax > 0 else 1.0
    x_scale = alt.Scale(type="time")
    month_df, year_df = _daily_month_year_axis(chart_df)

    line = (
        alt.Chart(chart_df)
        .mark_line(point=True, color="#60a5fa", strokeWidth=2)
        .encode(
            x=alt.X("date:T", scale=x_scale, axis=alt.Axis(labels=False, ticks=False, title=None)),
            y=alt.Y(
                "파손 건수:Q",
                title="파손 건수",
                scale=alt.Scale(domain=[0, ymax], nice=False),
            ),
            tooltip=[
                alt.Tooltip("날짜:N", title="날짜"),
                alt.Tooltip("파손 건수:Q", title="파손 건수"),
            ],
        )
        .properties(height=340)
    )
    day_axis = (
        alt.Chart(chart_df)
        .mark_text(align="center", baseline="middle", fontSize=11, color="#ffffff")
        .encode(
            x=alt.X(
                "date:T",
                scale=x_scale,
                axis=alt.Axis(
                    labels=False,
                    ticks=True,
                    tickSize=5,
                    title=None,
                    grid=False,
                    tickColor="#ffffff",
                    domainColor="#ffffff",
                ),
            ),
            text=alt.Text("일:N"),
        )
        .properties(height=28)
    )
    month_axis = (
        alt.Chart(month_df)
        .mark_text(align="center", baseline="middle", fontSize=12, color="#ffffff")
        .encode(
            x=alt.X(
                "date:T",
                scale=x_scale,
                axis=alt.Axis(labels=False, ticks=False, title=None, domain=False),
            ),
            text="월:N",
        )
        .properties(height=26)
    )
    year_axis = (
        alt.Chart(year_df)
        .mark_text(align="center", baseline="middle", fontSize=13, color="#ffffff")
        .encode(
            x=alt.X(
                "date:T",
                scale=x_scale,
                axis=alt.Axis(labels=False, ticks=False, title=None, domain=False),
            ),
            text="년:N",
        )
        .properties(height=26)
    )
    chart = alt.vconcat(line, day_axis, month_axis, year_axis, spacing=0).resolve_scale(x="shared")
    st.altair_chart(chart, use_container_width=True)


def _render_event_statistics(tracker: BreakageTracker) -> None:
    detail_df = _display_detail_df(tracker)
    if detail_df.empty:
        _show_empty_filter_message(tracker)
        return

    sm = summary_metrics(detail_df)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("총 파손 건수", f"{sm['total_events']} 건")
    with c2:
        st.metric("관련 설비", f"{sm['equipment_count']} 대")
    with c3:
        st.metric("제품 종류", f"{sm['product_count']} 종")
    with c4:
        st.metric("등록자", f"{sm['registrar_count']} 명")
    with c5:
        avg = sm["avg_drill_usage"]
        st.metric("평균 드릴사용량", f"{avg:,.0f}" if avg is not None else "-")

    if sm["date_min"] is not None and sm["date_max"] is not None:
        st.caption(
            f"분석 기간: {pd.Timestamp(sm['date_min']).strftime('%Y-%m-%d')} "
            f"~ {pd.Timestamp(sm['date_max']).strftime('%Y-%m-%d')}"
        )

    top_n = _chart_top_n_selector()

    st.subheader("파손 추이")
    _render_breakage_trends(detail_df)

    eq_all = equipment_counts_dataframe(detail_df)
    eq_active = eq_all[eq_all["파손 횟수"] > 0]
    st.subheader("설비별 파손 건수 (이벤트 로그 집계)")
    if eq_active.empty:
        st.caption("파손 이력 설비 없음")
    else:
        chart_eq = eq_active if top_n is None else eq_active.head(top_n)
        _stats_bar_chart(chart_eq, "설비", "파손 횟수")
        with st.expander("전체 설비 파손 건수"):
            st.dataframe(eq_all, width="stretch", hide_index=True)

    tab_eq, tab_prod, tab_reg, tab_lot, tab_tool, tab_usage, tab_type, tab_matrix = st.tabs(
        ["설비별", "제품별", "등록자별", "드릴 랏", "툴설명", "드릴사용량", "파손 형태", "제품×설비"]
    )

    with tab_eq:
        eq_df = count_by_column(detail_df, "설비", top_n)
        if eq_df.empty:
            st.caption("데이터 없음")
        else:
            _stats_bar_chart(eq_df, "설비")
            st.dataframe(eq_df, width="stretch", hide_index=True)

    with tab_prod:
        prod_df = count_by_column(detail_df, "제품코드", top_n)
        if prod_df.empty:
            st.caption("데이터 없음")
        else:
            _stats_bar_chart(prod_df, "제품코드")
            st.dataframe(prod_df, width="stretch", hide_index=True)

    with tab_reg:
        reg_df = count_by_column(detail_df, "등록자", top_n)
        if reg_df.empty:
            st.caption("데이터 없음")
        else:
            _stats_bar_chart(reg_df, "등록자")
            st.dataframe(reg_df, width="stretch", hide_index=True)

    with tab_lot:
        lot_df = count_drill_lot_grouped(detail_df, top_n)
        hole_df = count_by_column(detail_df, "브로큰 홀번호", top_n)
        if not lot_df.empty:
            st.caption("드릴 랏별 파손 (공통 글자·숫자 접두사 자동 합산)")
            _stats_bar_chart(lot_df, "드릴 랏(통합)")
            st.dataframe(lot_df, width="stretch", hide_index=True)
            with st.expander("통합 랏별 원본 드릴 랏"):
                st.dataframe(drill_lot_group_detail(detail_df, top_n), width="stretch", hide_index=True)
        if not hole_df.empty:
            st.caption("브로큰 홀번호별 파손")
            _stats_bar_chart(hole_df, "브로큰 홀번호")

    with tab_tool:
        tool_df = count_by_column(detail_df, "툴설명", top_n)
        if tool_df.empty:
            st.caption("데이터 없음")
        else:
            _stats_bar_chart(tool_df, "툴설명")
            st.dataframe(tool_df, width="stretch", hide_index=True)

    with tab_usage:
        usage_df = drill_usage_by_equipment(detail_df, top_n)
        if usage_df.empty:
            st.caption("드릴사용량 데이터가 없습니다.")
        else:
            _stats_bar_chart(usage_df, "설비", "평균 드릴사용량")
            st.dataframe(usage_df, width="stretch", hide_index=True)

    with tab_type:
        type_df = count_by_column(detail_df, "파손 형태", top_n)
        if type_df.empty:
            st.caption("파손 형태 데이터가 없습니다.")
        else:
            _stats_bar_chart(type_df, "파손 형태")
            st.dataframe(type_df, width="stretch", hide_index=True)

    with tab_matrix:
        matrix = product_equipment_matrix(detail_df, top_products=top_n, top_equipment=top_n)
        if matrix.empty:
            st.caption("교차 분석 데이터가 부족합니다.")
        else:
            st.dataframe(matrix, width="stretch")

    st.subheader("이벤트 로그 원본")
    st.dataframe(detail_df, width="stretch", hide_index=True)


tracker = _get_tracker()

grid_top, grid_bottom, grid_left, grid_right = 0.04, 0.97, 0.01, 0.88

with st.sidebar:
    st.header("데이터")
    if st.button("이벤트 로그 초기화", width="stretch"):
        tracker.reset_counts()
        tracker.save()
        st.rerun()
    if st.button("전체 초기화", width="stretch"):
        tracker.reset_all()
        tracker.save()
        st.rerun()

    st.divider()
    st.subheader("파손 형태 필터")
    type_options = breakage_type_options(_detail_df(tracker))
    st.radio(
        "표시 방식",
        ("전체 포함", "선택 제외", "선택만"),
        horizontal=True,
        key="breakage_type_mode",
        help="통계·맵·요약에 적용됩니다. 데이터 입력(업로드)은 전체 로그에 반영됩니다.",
    )
    if st.session_state.get("breakage_type_mode", "전체 포함") != "전체 포함":
        st.multiselect(
            "파손 형태",
            options=type_options if type_options else ["(미입력)"],
            key="breakage_type_selected",
        )
    elif not type_options:
        st.caption("이벤트 로그에 파손 형태(I열) 데이터가 없습니다.")

    st.divider()
    st.subheader("이미지 분석 영역")
    st.caption("스크린샷 해상도가 다르면 슬라이더로 그리드 위치를 맞춰 주세요.")
    grid_top = st.slider("상단", 0.0, 0.3, 0.04, 0.01)
    grid_bottom = st.slider("하단", 0.7, 1.0, 0.97, 0.01)
    grid_left = st.slider("좌측", 0.0, 0.2, 0.01, 0.01)
    grid_right = st.slider("우측", 0.7, 1.0, 0.88, 0.01)

    st.divider()
    if "confirm_shutdown" not in st.session_state:
        st.session_state.confirm_shutdown = False

    if not st.session_state.confirm_shutdown:
        if st.button("프로그램 종료", width="stretch", type="primary"):
            st.session_state.confirm_shutdown = True
            st.rerun()
    else:
        st.warning("프로그램을 종료할까요?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("예", key="shutdown_yes", width="stretch"):
                _shutdown_app(tracker)
        with c2:
            if st.button("아니오", key="shutdown_no", width="stretch"):
                st.session_state.confirm_shutdown = False
                st.rerun()
        st.caption("종료 전 데이터는 자동 저장됩니다.")

st.title("드릴 파손 카운트")
_render_global_period_selector(_filtered_by_breakage_type(tracker))
st.caption(
    "맵·통계·요약은 **이벤트 로그** 기준이며, **조회 기간**·사이드바 **파손 형태** 필터가 적용됩니다."
)
_show_breakage_type_filter_banner()
_show_period_filter_banner()

filtered_counts = _filtered_equipment_counts(tracker)
metrics = _breakage_metrics(tracker)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("총 파손 발생", f"{metrics['total_events']} 회")
with m2:
    st.metric("파손 대상 설비", f"{metrics['broken_equipment']} 대")
with m3:
    st.metric("미 파손 설비", f"{metrics['intact_equipment']} 대")
with m4:
    st.metric("총 파손 설비율", f"{metrics['breakage_rate_pct']:.1f} %")

tab_grid, tab_chart, tab_data, tab_upload, tab_manual, tab_log = st.tabs(
    ["설비 맵", "파손 통계", "데이터 입력", "스크린샷 분석", "상태 변경", "이벤트 로그"]
)

with tab_grid:
    st.subheader("DRA 설비 맵 (대시보드 배치)")
    st.markdown(_render_dashboard_map(filtered_counts), unsafe_allow_html=True)

    legend_cols = st.columns(3)
    for i, (status, label) in enumerate(MAP_LEGEND_ITEMS):
        with legend_cols[i]:
            st.markdown(
                f'<span style="color:{STATUS_COLORS[status]};">■</span> {label}',
                unsafe_allow_html=True,
            )
    st.caption("포인트 색상: 조회 기간·파손 형태 필터 적용 건수 기준 — 0회 미파손 · 1~2회 파손 · 3회 이상 파손")

with tab_chart:
    _render_event_statistics(tracker)

with tab_data:
    st.subheader("파손 이벤트 로그 입력")

    sub_single, sub_event, sub_bulk, sub_editor = st.tabs(
        ["단건 추가", "이벤트 로그 업로드", "횟수 일괄 변환", "설비별 건수 편집"]
    )

    with sub_single:
        st.caption("설비에 파손 이벤트를 **1건씩 추가**합니다. (이벤트 로그에 기록됨)")
        c1, c2, c3 = st.columns([2, 1.5, 1])
        with c1:
            count_eq = st.selectbox("설비", ALL_EQUIPMENT_IDS, key="count_eq")
        with c2:
            add_n = st.number_input("추가 건수", min_value=1, value=1, step=1, key="add_n")
        with c3:
            st.write("")
            st.write("")
            apply_count = st.button("추가", type="primary", key="apply_count")

        if apply_count:
            try:
                tracker.add_count(count_eq, int(add_n), note="manual_single")
                tracker.save()
                st.success(f"{count_eq} → 현재 {tracker.get_count(count_eq)}건")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

        if _display_filter_active():
            shown = filtered_counts.get(count_eq, 0)
            total = tracker.get_count(count_eq)
            st.caption(
                f"표시 기준 {count_eq} 파손 **{shown}건** (전체 로그 **{total}건**)"
            )
        else:
            st.caption(f"현재 {count_eq} 파손 **{tracker.get_count(count_eq)}건** (이벤트 로그 집계)")

    with sub_event:
            st.markdown(
                "파손 **1건 = 1행** 이벤트 로그입니다. 업로드한 행이 그대로 이벤트 로그에 쌓이고 "
                "맵·통계·요약에 반영됩니다.\n\n"
                "| 제품코드 | 설비 | 작업종료 | 등록자 | 툴설명 | 드릴 랏 | 브로큰 홀번호 | 드릴사용량 | 파손 형태 | 특이사항 |"
            )

            event_tpl = event_log_template_dataframe()
            st.download_button(
                "이벤트 로그 CSV 템플릿 다운로드",
                data=event_tpl.to_csv(index=False).encode("utf-8-sig"),
                file_name="드릴_파손_이벤트_템플릿.csv",
                mime="text/csv",
                key="dl_event_tpl",
            )
            st.dataframe(event_tpl, width="stretch", hide_index=True)

            event_file = st.file_uploader(
                "이벤트 로그 CSV / 엑셀",
                type=["csv", "xlsx", "xls"],
                key="event_file",
            )
            event_paste = st.text_area("또는 붙여넣기", height=120, key="event_paste")

            if st.button("이벤트 로그 반영", type="primary", key="apply_event"):
                try:
                    if event_file is not None:
                        raw_df = read_table_file(event_file.getvalue(), event_file.name)
                    elif event_paste.strip():
                        raw_df = pd.read_csv(StringIO(event_paste), sep=None, engine="python")
                    else:
                        st.warning("파일을 선택하거나 표를 붙여 넣어 주세요.")
                        raw_df = None

                    if raw_df is not None:
                        records, errors = parse_event_log_table(raw_df)
                        if errors:
                            for err in errors[:20]:
                                st.warning(err)
                            if len(errors) > 20:
                                st.warning(f"외 {len(errors) - 20}건 오류")
                        if not records:
                            st.error("반영할 이벤트가 없습니다.")
                        else:
                            n = tracker.import_detail_records(records)
                            tracker.save()
                            st.success(f"{n}건 이벤트 로그 반영 완료")
                            st.rerun()
                except Exception as e:
                    st.exception(e)

    with sub_bulk:
            st.caption(
                "설비별 **목표 건수**만 있는 CSV는 이벤트 로그 **빈 건**으로 변환해 반영합니다. "
                "상세 정보가 있으면 **이벤트 로그 업로드**를 사용하세요."
            )
            st.markdown(
                "| 설비 | 파손 횟수 |\n|------|----------|\n| DRA03 | 5 |\n| DRA84 | 2 |"
            )
            import_mode = st.radio(
                "반영 방식",
                ("덮어쓰기 (업로드한 설비만)", "더하기", "전체 교체 (미포함 설비는 0)"),
                horizontal=True,
            )
            mode_map = {
                "덮어쓰기 (업로드한 설비만)": "set",
                "더하기": "add",
                "전체 교체 (미포함 설비는 0)": "replace_all",
            }
            template_df = template_dataframe()[["설비", "파손 횟수"]]
            st.download_button(
                "횟수 변환용 CSV 템플릿",
                data=template_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="드릴_파손_횟수_변환_템플릿.csv",
                mime="text/csv",
                key="dl_count_tpl",
            )
            bulk_file = st.file_uploader("CSV / 엑셀", type=["csv", "xlsx", "xls"], key="bulk_file")
            paste_text = st.text_area("또는 붙여넣기", height=120, key="paste_area")
            if st.button("횟수 → 이벤트 로그 변환 반영", type="primary", key="apply_bulk"):
                try:
                    if bulk_file is not None:
                        raw_df = read_table_file(bulk_file.getvalue(), bulk_file.name)
                    elif paste_text.strip():
                        raw_df = pd.read_csv(StringIO(paste_text), sep=None, engine="python")
                    else:
                        st.warning("파일을 선택하거나 표를 붙여 넣어 주세요.")
                        raw_df = None
                    if raw_df is not None:
                        if is_event_log_format(raw_df):
                            st.warning("이벤트 로그 형식입니다. **이벤트 로그 업로드** 탭을 사용하세요.")
                        else:
                            counts, statuses, errors = parse_count_table(raw_df)
                            for err in errors[:20]:
                                st.warning(err)
                            if not counts:
                                st.error("반영할 데이터가 없습니다.")
                            else:
                                n = tracker.import_counts(counts, mode=mode_map[import_mode], statuses=None)
                                tracker.save()
                                st.success(f"{n}개 설비 · 총 {sum(counts.values())}건 이벤트 로그로 반영")
                                st.rerun()
                except Exception as e:
                    st.exception(e)

    with sub_editor:
            st.caption("설비별 **이벤트 로그 건수**를 맞춥니다. 줄이면 최근 이벤트부터 삭제됩니다.")
            editor_base = equipment_counts_dataframe(_detail_df(tracker))
            edited = st.data_editor(
                editor_base,
                width="stretch",
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "설비": st.column_config.TextColumn(disabled=True),
                    "파손 횟수": st.column_config.NumberColumn(min_value=0, step=1),
                },
                key="count_editor",
            )
            if st.button("건수 저장", type="primary", key="save_editor"):
                from data_import import normalize_equipment_id

                updated = 0
                for _, row in edited.iterrows():
                    eq_id = normalize_equipment_id(row["설비"])
                    if eq_id is None:
                        continue
                    try:
                        new_count = int(row["파손 횟수"])
                        if tracker.get_count(eq_id) != new_count:
                            tracker.set_count(eq_id, new_count, note="editor")
                            updated += 1
                    except (TypeError, ValueError):
                        pass
                tracker.save()
                st.success(f"{updated}개 설비 이벤트 로그 건수 조정됨")
                st.rerun()

with tab_upload:
        st.subheader("대시보드 스크린샷 업로드")
        uploaded = st.file_uploader("PNG/JPG 스크린샷", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded:
            st.image(uploaded, caption="업로드된 이미지", width="stretch")
            if st.button("이미지에서 상태 추출 및 반영", type="primary"):
                raw = uploaded.getvalue()
                detected = analyze_dashboard_image(
                    raw,
                    grid_top=grid_top,
                    grid_bottom=grid_bottom,
                    grid_left=grid_left,
                    grid_right=grid_right,
                )
                newly_broken = tracker.update_batch(detected)
                tracker.save()

                det_df = pd.DataFrame(
                    [
                        {
                            "설비": k,
                            "감지 상태": STATUS_LABELS[v],
                            "파손 횟수": tracker.get_count(k),
                        }
                        for k, v in sorted(detected.items())
                    ]
                )
                st.success(f"상태 반영 완료. 새 파손 이벤트: {len(newly_broken)}건")
                if newly_broken:
                    st.warning(f"파손 카운트 증가: {', '.join(newly_broken)}")
                st.dataframe(det_df, width="stretch", hide_index=True)
                st.rerun()

with tab_manual:
        st.subheader("설비 상태 변경 (스크린샷 없이)")
        col_a, col_b = st.columns(2)
        with col_a:
            eq_id = st.selectbox("설비 선택", ALL_EQUIPMENT_IDS)
        with col_b:
            status_options = list(EquipmentStatus)
            new_status = st.selectbox(
                "상태",
                status_options,
                format_func=lambda s: STATUS_LABELS[s],
            )

        if st.button("상태 적용", type="primary"):
            was_new = tracker.update_status(eq_id, new_status)
            tracker.save()
            if was_new:
                st.success(f"{eq_id} 파손 카운트 +1 (총 {tracker.get_count(eq_id)}회)")
            else:
                st.info(f"{eq_id} 상태를 {STATUS_LABELS[new_status]}(으)로 변경했습니다.")
            st.rerun()

        st.caption(
            "상태를 **파손**으로 바꾸면 카운트가 1 증가합니다. "
            "횟수를 직접 넣으려면 **데이터 입력** 탭을 사용하세요."
        )

with tab_log:
    st.subheader("이벤트 로그")
    detail_df = _display_detail_df(tracker)
    if detail_df.empty:
        _show_empty_filter_message(tracker)
    else:
        st.dataframe(detail_df.sort_values("작업종료", ascending=False, na_position="last"), width="stretch", hide_index=True)
        st.caption("파손 추이 (작업종료 기준)")
        _render_breakage_trends(detail_df)

st.caption(
    "맵·상단 지표·파손 통계는 **이벤트 로그**를 기준으로 하며, 상단 **조회 기간**·사이드바 **파손 형태** 필터가 적용됩니다."
)


def _launched_by_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" and not _launched_by_streamlit():
    import subprocess

    script = Path(__file__).resolve()
    app_dir = script.parent

    # 기존 8501 사용 프로세스 정리 (중복 실행 방지)
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in result.stdout.splitlines():
                if ":8501" in line and "LISTENING" in line:
                    pid = line.split()[-1]
                    if pid.isdigit():
                        subprocess.run(
                            ["taskkill", "/PID", pid, "/F"],
                            capture_output=True,
                            check=False,
                        )
        except OSError:
            pass

    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "false"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(script),
            "--server.headless",
            "false",
            "--server.port",
            "8501",
        ],
        check=False,
        env=env,
        cwd=str(app_dir),
    )
