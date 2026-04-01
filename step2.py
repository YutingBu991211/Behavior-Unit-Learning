# -*- coding: utf-8 -*-
"""
Step2-1 Part 1
=====================================

目标：
1) 在 Step1 终表基础上，构造 Step2 episode 切分前的统一输入表
2) 先不真正切 episode，只构造：
   - 连续竞价样本过滤
   - 相邻事件时间差 dt_ms
   - hard boundary 基础标记
   - soft boundary 所需基础变量
3) 输出 event_stream_step2_ready，供下一步顺序扫描切分使用

设计原则：
- 不修改 Step1 的核心逻辑
- 只在 Step1 结果上做“面向切分”的增量加工
- 保持 online 可得口径：只使用当前及过去可见信息
"""

from __future__ import annotations

from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Step2-1 配置
# =============================================================================

STEP2_DEFAULT_CONTINUOUS_SESSIONS = ["continuous_am", "continuous_pm"]


# =============================================================================
# Step2-1 工具函数
# =============================================================================

def _require_columns(df: pd.DataFrame, required_cols: List[str], func_name: str) -> None:
    """
    检查输入表是否具备必要字段。
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{func_name} missing required columns: {missing}")


def _safe_numeric(series: pd.Series) -> pd.Series:
    """
    安全转数值；非法值转为 NaN。
    """
    return pd.to_numeric(series, errors="coerce")


def _build_zscore_by_session(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "session_bucket_std",
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    对某个列按 session 做 z-score 标准化。
    目的：
    - 后续 soft boundary 可用统一阈值（如 |z| > 3）
    - 避免上午和下午波动水平不同导致阈值失真
    """
    if out_col is None:
        out_col = f"z_{value_col}"

    x = _safe_numeric(df[value_col])

    grp_mean = x.groupby(df[group_col]).transform("mean")
    grp_std = x.groupby(df[group_col]).transform("std")

    # 避免除以 0
    valid = grp_std.notna() & (grp_std > 0)
    z = pd.Series(np.nan, index=df.index, dtype="float64")
    z.loc[valid] = (x.loc[valid] - grp_mean.loc[valid]) / grp_std.loc[valid]

    df[out_col] = z
    return df


# =============================================================================
# Step2-1 Part 1：构造切分前输入表
# =============================================================================

def build_event_stream_step2_ready(
    symbol: str,
    keep_sessions: Optional[List[str]] = None,
    gap_hard_ms: int = 3000,
    lag_hard_ms: int = 5000,
    max_events_per_episode: int = 64,
    max_duration_ms: int = 10000,
    step1_artifact_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    构造 Step2 切分前的统一输入表 event_stream_step2_ready。

    输入假设：
    - 直接基于 Step1 最终输出函数 build_event_stream_step1_with_price_position(symbol)
    - 每行一条事件，且已经有稳定 event_idx
    - 字段名沿用你当前 Step1 的命名

    参数说明：
    - keep_sessions:
        保留哪些 session。默认只保留 continuous_am / continuous_pm
    - gap_hard_ms:
        相邻事件时间差超过该阈值时，视为 hard boundary 候选
    - lag_hard_ms:
        snapshot lag 超过该阈值时，视为 snapshot 过旧，触发 hard boundary 候选
    - max_events_per_episode:
        后续 episode 最长事件数阈值（这里只先写入配置列与基础计数，真正切分下一步做）
    - max_duration_ms:
        后续 episode 最长持续时间阈值（这里只先保留配置，真正切分下一步做）

    返回：
    - event_stream_step2_ready: pd.DataFrame
      在 Step1 基础上增加了：
        * dt_ms
        * is_continuous_session
        * is_valid_for_step2
        * hard boundary 基础标记
        * soft boundary 所需 z-score / rolling 基础变量
    """

    if keep_sessions is None:
        keep_sessions = STEP2_DEFAULT_CONTINUOUS_SESSIONS

    # -------------------------------------------------------------------------
    # 1) 读取 Step1 最终事件表
    # -------------------------------------------------------------------------
    from step1 import load_step1_final_artifact

    if step1_artifact_dir is None:
        raise ValueError(
            "build_event_stream_step2_ready requires step1_artifact_dir now. "
            "Please build Step1 final artifact first, then pass its directory into Step2."
        )

    es = load_step1_final_artifact(step1_artifact_dir)

    # -------------------------------------------------------------------------
    # 2) 检查必要字段
    # -------------------------------------------------------------------------
    required_cols = [
        "event_idx",
        "symbol",
        "event_time_ms",
        "event_time_str",
        "event_family",
        "event_type_raw_std",
        "event_qty",
        "event_side_raw_std",
        "session_bucket_std",
        "has_snapshot",
        "snapshot_is_stale",
        "snapshot_lag_ms",
        "snap_last",
        "snap_spread_abs",
        "snap_top_depth_total",
        "snap_top_depth_imbalance",
        "state_change_is_valid",
        "d_snap_last",
        "d_snap_spread_abs",
        "d_snap_top_depth_total",
        "d_snap_top_depth_imbalance",
        "event_price_position_std",
        "event_price_aggressive_flag",
    ]
    _require_columns(es, required_cols, "build_event_stream_step2_ready")

    # 关键：Step2 后续只需要这些列，先裁瘦再做过滤/rolling
    es = es.loc[:, required_cols].copy()

    # -------------------------------------------------------------------------
    # 3) 数值字段统一转 numeric，避免后面 rolling / 比较出问题
    # -------------------------------------------------------------------------
    numeric_cols = [
        "event_idx",
        "event_time_ms",
        "event_qty",
        "has_snapshot",
        "snapshot_is_stale",
        "snapshot_lag_ms",
        "snap_last",
        "snap_spread_abs",
        "snap_top_depth_total",
        "snap_top_depth_imbalance",
        "state_change_is_valid",
        "d_snap_last",
        "d_snap_spread_abs",
        "d_snap_top_depth_total",
        "d_snap_top_depth_imbalance",
    ]
    for col in numeric_cols:
        es[col] = _safe_numeric(es[col])

    # -------------------------------------------------------------------------
    # 4) 只保留连续竞价样本
    # -------------------------------------------------------------------------
    es["is_continuous_session"] = es["session_bucket_std"].isin(keep_sessions).astype("int8")

    mask = es["is_continuous_session"].eq(1)
    es = es.loc[mask].reset_index(drop=True)

    # 只按数值 event_idx 保证顺序；不要再让字符串 session 参与排序
    es["event_idx"] = pd.to_numeric(es["event_idx"], errors="coerce")
    es = es.sort_values("event_idx", ascending=True, kind="stable").reset_index(drop=True)
    # -------------------------------------------------------------------------
    # 5) 构造相邻事件时间差 dt_ms
    #    只在同一 session_bucket_std 内比较
    # -------------------------------------------------------------------------
    es["prev_event_idx"] = es.groupby("session_bucket_std")["event_idx"].shift(1)
    es["prev_event_time_ms"] = es.groupby("session_bucket_std")["event_time_ms"].shift(1)

    es["dt_ms"] = es["event_time_ms"] - es["prev_event_time_ms"]

    # 第一条事件没有前驱，不定义 dt_ms
    es.loc[es["prev_event_time_ms"].isna(), "dt_ms"] = pd.NA

    # 异常保护：如果出现负值，先标记，后续不要直接参与规则
    es["dt_ms_negative_flag"] = (
        es["dt_ms"].notna() &
        (_safe_numeric(es["dt_ms"]) < 0)
    ).astype(int)

    # -------------------------------------------------------------------------
    # 6) 构造 Step2 有效性标记
    #    注意：这里只是“是否适合作为 episode 切分输入”，不是 hard boundary 本身
    # -------------------------------------------------------------------------
    es["is_valid_for_step2"] = 1

    # 没 snapshot 的先保留到表里，但不视为高质量切分输入
    es.loc[es["has_snapshot"] != 1, "is_valid_for_step2"] = 0

    # snapshot stale 的样本也保留，但不视为高质量切分输入
    es.loc[es["snapshot_is_stale"] == 1, "is_valid_for_step2"] = 0

    # lag 过大
    es.loc[
        es["snapshot_lag_ms"].notna() &
        (es["snapshot_lag_ms"] > lag_hard_ms),
        "is_valid_for_step2"
    ] = 0

    # 时间差为负，视为异常
    es.loc[es["dt_ms_negative_flag"] == 1, "is_valid_for_step2"] = 0

    # -------------------------------------------------------------------------
    # 7) hard boundary 基础标记
    #    这里只是“候选切分原因”，真正切分下一步顺序扫描时统一处理
    # -------------------------------------------------------------------------
    # 7.1 session 开始处一定是新的候选片段起点
    es["hb_session_start"] = es["prev_event_idx"].isna().astype(int)

    # 7.2 时间静默过长
    es["hb_large_gap"] = (
        es["dt_ms"].notna() &
        (_safe_numeric(es["dt_ms"]) > gap_hard_ms)
    ).astype(int)

    # 7.3 没 snapshot
    es["hb_no_snapshot"] = (es["has_snapshot"] != 1).astype(int)

    # 7.4 snapshot stale
    es["hb_snapshot_stale"] = (es["snapshot_is_stale"] == 1).astype(int)

    # 7.5 snapshot lag 过大
    es["hb_snapshot_lag_too_large"] = (
        es["snapshot_lag_ms"].notna() &
        (_safe_numeric(es["snapshot_lag_ms"]) > lag_hard_ms)
    ).astype(int)

    # 7.6 state change 不可比
    es["hb_state_not_comparable"] = (es["state_change_is_valid"] != 1).astype(int)

    # 7.7 时间差为负
    es["hb_negative_dt"] = (es["dt_ms_negative_flag"] == 1).astype(int)

    # 汇总：这一行是否触发 hard boundary 候选
    hard_cols = [
        "hb_session_start",
        "hb_large_gap",
        "hb_no_snapshot",
        "hb_snapshot_stale",
        "hb_snapshot_lag_too_large",
        "hb_state_not_comparable",
        "hb_negative_dt",
    ]
    es["hard_boundary_candidate"] = (es[hard_cols].sum(axis=1) > 0).astype(int)

    # -------------------------------------------------------------------------
    # 8) soft boundary 需要的基础变量
    # -------------------------------------------------------------------------

    # 8.1 事件方向粗分类：后续用于方向翻转检测
    # 这里只做一个稳定、保守的 buy/sell/other 三分法
    side = es["event_side_raw_std"].astype("string").str.upper()

    es["event_side_simple"] = "other"
    es.loc[side.eq("B"), "event_side_simple"] = "buy"
    es.loc[side.eq("S"), "event_side_simple"] = "sell"

    es["is_buy_like"] = (es["event_side_simple"] == "buy").astype(int)
    es["is_sell_like"] = (es["event_side_simple"] == "sell").astype(int)

    # 8.2 aggressive 事件标记
    es["is_aggressive_event"] = (
        es["event_price_aggressive_flag"].astype("string").str.lower().eq("aggressive")
    ).astype(int)

    # 8.3 数量缩放：后续建模和 summary feature 都建议用 log1p
    qty_num = _safe_numeric(es["event_qty"]).clip(lower=0)
    es["event_qty_log1p"] = np.log1p(qty_num)

    # 8.4 价格位置 one-hot 基础计数（后续可聚合成 episode summary）
    pos = es["event_price_position_std"].astype("string")
    es["is_at_bid1"] = pos.eq("at_bid1").astype(int)
    es["is_at_ask1"] = pos.eq("at_ask1").astype(int)
    es["is_inside_spread"] = pos.eq("inside_spread").astype(int)
    es["is_lt_bid1"] = pos.eq("lt_bid1").astype(int)
    es["is_gt_ask1"] = pos.eq("gt_ask1").astype(int)

    # -------------------------------------------------------------------------
    # 9) soft boundary 用的 rolling 基础统计
    #    这里只构造，不真正用来切
    # -------------------------------------------------------------------------
    # 最近 8 个事件的 aggressive 占比
    es["rolling_aggr_ratio_8"] = (
        es.groupby("session_bucket_std")["is_aggressive_event"]
          .rolling(window=8, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )

    # 最近 8 个事件的 buy / sell 占比
    es["rolling_buy_ratio_8"] = (
        es.groupby("session_bucket_std")["is_buy_like"]
          .rolling(window=8, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )
    es["rolling_sell_ratio_8"] = (
        es.groupby("session_bucket_std")["is_sell_like"]
          .rolling(window=8, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )

    # 最近 8 个事件的中位 dt，用于判断“当前是否突然静默”
    es["rolling_median_dt_8"] = (
        es.groupby("session_bucket_std")["dt_ms"]
          .rolling(window=8, min_periods=1)
          .median()
          .reset_index(level=0, drop=True)
    )

    # -------------------------------------------------------------------------
    # 10) 对状态变化列做 session 内 z-score
    #     后续 soft boundary 可直接用统一阈值，比如 |z| > 3
    # -------------------------------------------------------------------------
    zscore_cols = [
        "d_snap_last",
        "d_snap_spread_abs",
        "d_snap_top_depth_total",
        "d_snap_top_depth_imbalance",
    ]
    for col in zscore_cols:
        es = _build_zscore_by_session(es, value_col=col, group_col="session_bucket_std", out_col=f"z_{col}")

    # -------------------------------------------------------------------------
    # 11) 构造 soft boundary 的“候选信号”，但暂不真正切分
    # -------------------------------------------------------------------------
    # 11.1 当前 dt 是否相对近期中位数突然增大
    cur_dt = _safe_numeric(es["dt_ms"])
    med_dt = _safe_numeric(es["rolling_median_dt_8"])

    es["sb_dt_jump_candidate"] = (
        cur_dt.notna() &
        med_dt.notna() &
        (
            (cur_dt > 1000) |
            (cur_dt > med_dt * 8)
        )
    ).astype(int)

    # 11.2 局部 aggressive burst 候选
    es["sb_aggr_burst_candidate"] = (
        _safe_numeric(es["rolling_aggr_ratio_8"]).notna() &
        (_safe_numeric(es["rolling_aggr_ratio_8"]) >= 0.75)
    ).astype(int)

    # 11.3 盘口状态突变候选
    es["sb_state_jump_candidate"] = (
        (
            _safe_numeric(es["z_d_snap_last"]).abs() > 3
        ) |
        (
            _safe_numeric(es["z_d_snap_top_depth_imbalance"]).abs() > 3
        ) |
        (
            _safe_numeric(es["z_d_snap_spread_abs"]).abs() > 3
        )
    ).astype(int)

    # 11.4 方向翻转候选
    es["prev_event_side_simple"] = es.groupby("session_bucket_std")["event_side_simple"].shift(1)

    es["sb_side_flip_candidate"] = (
        es["prev_event_side_simple"].notna() &
        es["event_side_simple"].isin(["buy", "sell"]) &
        es["prev_event_side_simple"].isin(["buy", "sell"]) &
        (es["event_side_simple"] != es["prev_event_side_simple"])
    ).astype(int)

    # 11.5 snapshot lag 偏大的 soft warning
    # 注意：这不是 hard boundary，只是提醒该事件的盘口上下文有点旧
    es["sb_snapshot_lag_warning"] = (
        es["snapshot_lag_ms"].notna() &
        (_safe_numeric(es["snapshot_lag_ms"]) > 3000)
    ).astype(int)

    # -------------------------------------------------------------------------
    # 12) 保留一些切分参数，便于后续函数直接读取
    # -------------------------------------------------------------------------
    es["cfg_gap_hard_ms"] = gap_hard_ms
    es["cfg_lag_hard_ms"] = lag_hard_ms
    es["cfg_max_events_per_episode"] = max_events_per_episode
    es["cfg_max_duration_ms"] = max_duration_ms

    # -------------------------------------------------------------------------
    # 13) 列顺序整理
    # -------------------------------------------------------------------------
    front_cols = [
        "event_idx",
        "symbol",
        "event_time_str",
        "event_time_ms",
        "session_bucket_std",
        "event_family",
        "event_type_raw_std",
        "event_side_raw_std",
        "event_side_simple",
        "event_price_position_std",
        "event_price_aggressive_flag",
        "event_qty",
        "event_qty_log1p",
        "dt_ms",
        "has_snapshot",
        "snapshot_is_stale",
        "snapshot_lag_ms",
        "snap_last",
        "snap_spread_abs",
        "snap_top_depth_total",
        "snap_top_depth_imbalance",
        "d_snap_last",
        "d_snap_spread_abs",
        "d_snap_top_depth_total",
        "d_snap_top_depth_imbalance",
        "z_d_snap_last",
        "z_d_snap_spread_abs",
        "z_d_snap_top_depth_total",
        "z_d_snap_top_depth_imbalance",
        "is_valid_for_step2",
        "hard_boundary_candidate",
        "hb_session_start",
        "hb_large_gap",
        "hb_no_snapshot",
        "hb_snapshot_stale",
        "hb_snapshot_lag_too_large",
        "hb_state_not_comparable",
        "hb_negative_dt",
        "sb_dt_jump_candidate",
        "sb_aggr_burst_candidate",
        "sb_state_jump_candidate",
        "sb_side_flip_candidate",
        "rolling_aggr_ratio_8",
        "rolling_buy_ratio_8",
        "rolling_sell_ratio_8",
        "rolling_median_dt_8",
    ]
    keep_cols = [c for c in front_cols if c in es.columns] + [c for c in es.columns if c not in front_cols]
    es = es[keep_cols].copy()

    return es


# =============================================================================
# Step2-1 Part 1 审计函数
# =============================================================================

def inspect_event_stream_step2_ready_df(es: pd.DataFrame, n: int = 20) -> None:
    """
    对已经构造好的 event_stream_step2_ready 做检查。
    注意：
    - 不再重新调用 build_event_stream_step2_ready
    - 避免 main 里重复跑整条 Step1/Step2 链
    """
    print("\n" + "=" * 120)
    print("STEP2-1 PART1 | event_stream_step2_ready")
    print("=" * 120)

    print("\n[shape]")
    print(es.shape)

    print("\n[columns]")
    print(list(es.columns))

    show_cols = [
        "event_idx",
        "event_time_str",
        "event_time_ms",
        "session_bucket_std",
        "event_family",
        "event_type_raw_std",
        "event_side_raw_std",
        "event_side_simple",
        "event_price_position_std",
        "event_price_aggressive_flag",
        "event_qty",
        "event_qty_log1p",
        "dt_ms",
        "has_snapshot",
        "snapshot_is_stale",
        "snapshot_lag_ms",
        "hard_boundary_candidate",
        "hb_session_start",
        "hb_large_gap",
        "hb_no_snapshot",
        "hb_snapshot_stale",
        "hb_snapshot_lag_too_large",
        "hb_state_not_comparable",
        "sb_dt_jump_candidate",
        "sb_aggr_burst_candidate",
        "sb_state_jump_candidate",
        "sb_side_flip_candidate",
    ]
    show_cols = [c for c in show_cols if c in es.columns]

    print("\n[head]")
    print(es[show_cols].head(n))

    print("\n[session counts]")
    print(es["session_bucket_std"].value_counts(dropna=False))

    print("\n[hard boundary candidate counts]")
    print(es["hard_boundary_candidate"].value_counts(dropna=False))

    print("\n[hard boundary by reason]")
    hard_cols = [
        "hb_session_start",
        "hb_large_gap",
        "hb_no_snapshot",
        "hb_snapshot_stale",
        "hb_snapshot_lag_too_large",
        "hb_state_not_comparable",
        "hb_negative_dt",
    ]
    hard_cols = [c for c in hard_cols if c in es.columns]
    print(es[hard_cols].sum().sort_values(ascending=False))

    print("\n[soft candidate by reason]")
    soft_cols = [
        "sb_dt_jump_candidate",
        "sb_aggr_burst_candidate",
        "sb_state_jump_candidate",
        "sb_side_flip_candidate",
        "sb_snapshot_lag_warning",
    ]
    soft_cols = [c for c in soft_cols if c in es.columns]
    print(es[soft_cols].sum().sort_values(ascending=False))

    print("\n[dt_ms summary]")
    print(pd.to_numeric(es["dt_ms"], errors="coerce").describe())

    print("\n[snapshot_lag_ms summary]")
    print(pd.to_numeric(es["snapshot_lag_ms"], errors="coerce").describe())

    print("\n[sample hard boundary rows]")
    sample_hard = es.loc[es["hard_boundary_candidate"] == 1, show_cols].head(20)
    print(sample_hard)

    print("\n[sample soft candidate rows]")
    soft_mask = (
        (es["sb_dt_jump_candidate"] == 1) |
        (es["sb_aggr_burst_candidate"] == 1) |
        (es["sb_state_jump_candidate"] == 1) |
        (es["sb_side_flip_candidate"] == 1) |
        (es["sb_snapshot_lag_warning"] == 1)
    )
    sample_soft = es.loc[soft_mask, show_cols].head(20)
    print(sample_soft)


# =============================================================================
# Step2-1 Part 2：顺序扫描切分 episode
# =============================================================================

def build_episode_candidates(
    symbol: str,
    keep_sessions: Optional[List[str]] = None,
    gap_hard_ms: int = 3000,
    lag_hard_ms: int = 5000,
    max_events_per_episode: int = 64,
    max_duration_ms: int = 10000,
    min_events_soft_cut: int = 5,
    min_events_episode_flag: int = 4,
    step1_artifact_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    基于 Step2-ready 事件流，顺序扫描生成 episode。

    返回：
    - es2_with_episode: 在 event_stream_step2_ready 基础上增加 episode_id / cut 标记
    - episode_table: 每个 episode 一行
    - episode_summary: 每个 episode 的统计摘要
    """

    es = build_event_stream_step2_ready(
        symbol=symbol,
        keep_sessions=keep_sessions,
        gap_hard_ms=gap_hard_ms,
        lag_hard_ms=lag_hard_ms,
        max_events_per_episode=max_events_per_episode,
        max_duration_ms=max_duration_ms,
        step1_artifact_dir=step1_artifact_dir,
    ).copy()

    # 避免后面 groupby / agg / 行扫描被 pyarrow dtype 拖慢
    for col in [
        "event_time_ms",
        "dt_ms",
        "snap_spread_abs",
        "snap_top_depth_total",
        "snap_top_depth_imbalance",
        "d_snap_last",
        "d_snap_top_depth_total",
        "d_snap_top_depth_imbalance",
        "event_qty_log1p",
    ]:
        if col in es.columns:
            es[col] = pd.to_numeric(es[col], errors="coerce")

    # -------------------------------------------------------------------------
    # 1) 顺序扫描分配 episode_id（numpy 加速版）
    # -------------------------------------------------------------------------
    n = len(es)

    # 先把扫描要用的列转成 numpy，避免逐行 iloc 非常慢
    event_time_arr = pd.to_numeric(es["event_time_ms"], errors="coerce").to_numpy()
    session_arr = es["session_bucket_std"].astype("string").to_numpy()

    hard_arr = pd.to_numeric(es["hard_boundary_candidate"], errors="coerce").fillna(0).astype(int).to_numpy()
    dt_jump_arr = pd.to_numeric(es["sb_dt_jump_candidate"], errors="coerce").fillna(0).astype(int).to_numpy()
    side_flip_arr = pd.to_numeric(es["sb_side_flip_candidate"], errors="coerce").fillna(0).astype(int).to_numpy()
    state_jump_arr = pd.to_numeric(es["sb_state_jump_candidate"], errors="coerce").fillna(0).astype(int).to_numpy()

    episode_ids = np.empty(n, dtype=np.int64)
    is_episode_start = np.zeros(n, dtype=np.int8)
    episode_cut_reason = np.empty(n, dtype=object)

    current_episode_id = 1
    current_start_idx = 0
    current_start_time_ms = event_time_arr[0]
    current_session = session_arr[0]

    episode_ids[0] = current_episode_id
    is_episode_start[0] = 1
    episode_cut_reason[0] = "start_of_stream"

    for i in range(1, n):
        row_time = event_time_arr[i]
        row_session = session_arr[i]

        current_length = i - current_start_idx
        current_duration_ms = (
            row_time - current_start_time_ms
            if pd.notna(row_time) and pd.notna(current_start_time_ms)
            else np.nan
        )

        # -------------------------
        # hard cut
        # -------------------------
        hard_cut = False
        hard_reason = None

        if hard_arr[i] == 1:
            hard_cut = True
            hard_reason = "hard_boundary_candidate"

        elif current_length >= max_events_per_episode:
            hard_cut = True
            hard_reason = "max_events_per_episode"

        elif pd.notna(current_duration_ms) and current_duration_ms >= max_duration_ms:
            hard_cut = True
            hard_reason = "max_duration_ms"

        elif row_session != current_session:
            hard_cut = True
            hard_reason = "session_change"

        # -------------------------
        # soft cut
        # -------------------------
        soft_cut = False
        soft_reason = None

        if not hard_cut and current_length >= min_events_soft_cut:
            if side_flip_arr[i] == 1 and dt_jump_arr[i] == 1:
                soft_cut = True
                soft_reason = "soft_side_flip_with_dt_jump"
            elif dt_jump_arr[i] == 1:
                soft_cut = True
                soft_reason = "soft_dt_jump"
            elif state_jump_arr[i] == 1:
                soft_cut = True
                soft_reason = "soft_state_jump"

        # -------------------------
        # 是否开启新 episode
        # -------------------------
        if hard_cut or soft_cut:
            current_episode_id += 1
            current_start_idx = i
            current_start_time_ms = row_time
            current_session = row_session

            episode_ids[i] = current_episode_id
            is_episode_start[i] = 1
            episode_cut_reason[i] = hard_reason if hard_cut else soft_reason
        else:
            episode_ids[i] = current_episode_id
            episode_cut_reason[i] = None

    es["episode_id"] = episode_ids
    es["is_episode_start"] = is_episode_start
    es["episode_cut_reason"] = episode_cut_reason

    # -------------------------------------------------------------------------
    # 2) episode_table：边界级信息
    # -------------------------------------------------------------------------
    g = es.groupby("episode_id", sort=True)

    episode_table = g.agg(
        symbol=("symbol", "first"),
        session_bucket_std=("session_bucket_std", "first"),
        start_event_idx=("event_idx", "first"),
        end_event_idx=("event_idx", "last"),
        start_time_ms=("event_time_ms", "first"),
        end_time_ms=("event_time_ms", "last"),
        start_time_str=("event_time_str", "first"),
        end_time_str=("event_time_str", "last"),
        n_events=("event_idx", "size"),
    ).reset_index()

    episode_table["duration_ms"] = (
        pd.to_numeric(episode_table["end_time_ms"], errors="coerce") -
        pd.to_numeric(episode_table["start_time_ms"], errors="coerce")
    )

    # 每个 episode 的起始 cut reason
    start_reason_map = (
        es.loc[es["is_episode_start"] == 1, ["episode_id", "episode_cut_reason"]]
          .drop_duplicates(subset=["episode_id"])
          .rename(columns={"episode_cut_reason": "episode_start_reason"})
    )
    episode_table = episode_table.merge(start_reason_map, on="episode_id", how="left")

    # -------------------------------------------------------------------------
    # 3) episode_summary：统计特征
    # -------------------------------------------------------------------------
    ef = es["event_family"].astype("string").str.lower()

    es["is_trade_event"] = ef.eq("trade").astype(int)
    es["is_add_event"] = ef.eq("order").astype(int)
    es["is_cancel_event"] = ef.eq("cancel").astype(int)

    summary = g.agg(
        symbol=("symbol", "first"),
        session_bucket_std=("session_bucket_std", "first"),
        n_events=("event_idx", "size"),
        start_event_idx=("event_idx", "first"),
        end_event_idx=("event_idx", "last"),
        start_time_ms=("event_time_ms", "first"),
        end_time_ms=("event_time_ms", "last"),
        n_trade=("is_trade_event", "sum"),
        n_add=("is_add_event", "sum"),
        n_cancel=("is_cancel_event", "sum"),
        buy_like_ratio=("is_buy_like", "mean"),
        sell_like_ratio=("is_sell_like", "mean"),
        aggressive_ratio=("is_aggressive_event", "mean"),
        at_bid1_ratio=("is_at_bid1", "mean"),
        at_ask1_ratio=("is_at_ask1", "mean"),
        inside_spread_ratio=("is_inside_spread", "mean"),
        mean_dt_ms=("dt_ms", "mean"),
        median_dt_ms=("dt_ms", "median"),
        max_dt_ms=("dt_ms", "max"),
        mean_qty_log1p=("event_qty_log1p", "mean"),
        mean_spread_abs=("snap_spread_abs", "mean"),
        mean_top_depth_total=("snap_top_depth_total", "mean"),
        mean_top_depth_imbalance=("snap_top_depth_imbalance", "mean"),
        delta_snap_last=("d_snap_last", "sum"),
        delta_depth_total=("d_snap_top_depth_total", "sum"),
        delta_depth_imbalance=("d_snap_top_depth_imbalance", "sum"),
    ).reset_index()

    summary["duration_ms"] = (
        pd.to_numeric(summary["end_time_ms"], errors="coerce") -
        pd.to_numeric(summary["start_time_ms"], errors="coerce")
    )

    summary["is_too_short_episode"] = (summary["n_events"] < min_events_episode_flag).astype(int)

    summary = summary.merge(
        episode_table[["episode_id", "episode_start_reason"]],
        on="episode_id",
        how="left"
    )

    return es, episode_table, summary

def inspect_episode_candidates(
    out_dir: str,
    n_episode_rows: int = 20,
) -> None:
    """
    审计 episode 切分结果。
    """
    es, episode_table, summary = load_step2_artifacts(out_dir)

    print("\n" + "=" * 120)
    print("STEP2-1 PART2 | episode candidates")
    print("=" * 120)

    print("\n[event rows]")
    print(es.shape)

    print("\n[number of episodes]")
    print(episode_table.shape[0])

    print("\n[episode_table head]")
    print(episode_table.head(n_episode_rows))

    print("\n[episode_start_reason counts]")
    print(episode_table["episode_start_reason"].value_counts(dropna=False))

    print("\n[n_events summary]")
    print(pd.to_numeric(episode_table["n_events"], errors="coerce").describe())

    print("\n[duration_ms summary]")
    print(pd.to_numeric(episode_table["duration_ms"], errors="coerce").describe())

    print("\n[too short episode ratio]")
    print(summary["is_too_short_episode"].value_counts(dropna=False))

    print("\n[summary head]")
    print(summary.head(n_episode_rows))


# =============================================================================
# Step2-2 Part 1：构造 episode feature table（baseline 输入）
# =============================================================================

def build_episode_feature_table(
    out_dir: str,
    clip_quantile_low: float = 0.01,
    clip_quantile_high: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    基于 episode_summary 构造 baseline / clustering 用的 episode 特征表。

    返回：
    - feature_raw:
        原始特征表（每行一个 episode）
    - feature_model:
        轻度清洗后的建模特征表（每行一个 episode）
    - feature_cols_model:
        最终建议用于聚类/降维的数值列
    """

    # -------------------------------------------------------------------------
    # 1) 先拿 episode 级结果
    # -------------------------------------------------------------------------
    es, episode_table, summary = load_step2_artifacts(out_dir)
    feature_raw = summary.copy()

    # -------------------------------------------------------------------------
    # 2) 第一版基础特征
    # -------------------------------------------------------------------------
    base_feature_cols = [
        "n_events",
        "duration_ms",
        "n_trade",
        "n_add",
        "n_cancel",
        "buy_like_ratio",
        "sell_like_ratio",
        "aggressive_ratio",
        "at_bid1_ratio",
        "at_ask1_ratio",
        "inside_spread_ratio",
        "mean_dt_ms",
        "median_dt_ms",
        "max_dt_ms",
        "mean_qty_log1p",
        "mean_spread_abs",
        "mean_top_depth_total",
        "mean_top_depth_imbalance",
        "delta_snap_last",
        "delta_depth_total",
        "delta_depth_imbalance",
    ]
    feature_cols = [c for c in base_feature_cols if c in feature_raw.columns]

    # -------------------------------------------------------------------------
    # 3) 派生特征
    # -------------------------------------------------------------------------
    # 事件组成占比
    if {"n_trade", "n_events"}.issubset(feature_raw.columns):
        feature_raw["trade_ratio"] = (
            pd.to_numeric(feature_raw["n_trade"], errors="coerce") /
            pd.to_numeric(feature_raw["n_events"], errors="coerce")
        )
    if {"n_add", "n_events"}.issubset(feature_raw.columns):
        feature_raw["add_ratio"] = (
            pd.to_numeric(feature_raw["n_add"], errors="coerce") /
            pd.to_numeric(feature_raw["n_events"], errors="coerce")
        )
    if {"n_cancel", "n_events"}.issubset(feature_raw.columns):
        feature_raw["cancel_ratio"] = (
            pd.to_numeric(feature_raw["n_cancel"], errors="coerce") /
            pd.to_numeric(feature_raw["n_events"], errors="coerce")
        )

    # 方向不平衡：越接近 1/-1 越单边
    if {"buy_like_ratio", "sell_like_ratio"}.issubset(feature_raw.columns):
        feature_raw["side_imbalance"] = (
            pd.to_numeric(feature_raw["buy_like_ratio"], errors="coerce") -
            pd.to_numeric(feature_raw["sell_like_ratio"], errors="coerce")
        )

    # 顶档位置不平衡：偏 bid1 / ask1
    if {"at_bid1_ratio", "at_ask1_ratio"}.issubset(feature_raw.columns):
        feature_raw["top1_pos_imbalance"] = (
            pd.to_numeric(feature_raw["at_bid1_ratio"], errors="coerce") -
            pd.to_numeric(feature_raw["at_ask1_ratio"], errors="coerce")
        )

    derived_cols = [
        "trade_ratio",
        "add_ratio",
        "cancel_ratio",
        "side_imbalance",
        "top1_pos_imbalance",
    ]
    derived_cols = [c for c in derived_cols if c in feature_raw.columns]

    feature_cols = feature_cols + derived_cols

    # 去重保序
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    # -------------------------------------------------------------------------
    # 4) 构造建模表
    # -------------------------------------------------------------------------
    keep_meta_cols = [
        "episode_id",
        "symbol",
        "session_bucket_std",
        "episode_start_reason",
        "is_too_short_episode",
        "start_event_idx",
        "end_event_idx",
        "start_time_ms",
        "end_time_ms",
    ]
    keep_meta_cols = [c for c in keep_meta_cols if c in feature_raw.columns]

    feature_model = feature_raw[keep_meta_cols + feature_cols].copy()

    # -------------------------------------------------------------------------
    # 5) 全部转 numeric
    # -------------------------------------------------------------------------
    for col in feature_cols:
        feature_model[col] = pd.to_numeric(feature_model[col], errors="coerce")

    # -------------------------------------------------------------------------
    # 6) 缺失值处理：用中位数填
    # -------------------------------------------------------------------------
    for col in feature_cols:
        med = feature_model[col].median(skipna=True)
        if pd.isna(med):
            med = 0.0
        feature_model[col] = feature_model[col].fillna(med)

    # -------------------------------------------------------------------------
    # 7) 对长尾“规模类特征”做 log1p
    # -------------------------------------------------------------------------
    log_cols = [
        "n_events",
        "duration_ms",
        "n_trade",
        "n_add",
        "n_cancel",
        "mean_dt_ms",
        "median_dt_ms",
        "max_dt_ms",
        "mean_spread_abs",
        "mean_top_depth_total",
    ]
    log_cols = [c for c in log_cols if c in feature_model.columns]

    for col in log_cols:
        x = pd.to_numeric(feature_model[col], errors="coerce").clip(lower=0)
        feature_model[f"{col}_log1p"] = np.log1p(x)

    # 建模时优先使用 log 后列
    feature_cols_model = []
    for col in feature_cols:
        if col in log_cols:
            feature_cols_model.append(f"{col}_log1p")
        else:
            feature_cols_model.append(col)

    # -------------------------------------------------------------------------
    # 8) 温和 clip，防止极端 episode 主导后续聚类
    # -------------------------------------------------------------------------
    for col in feature_cols_model:
        x = pd.to_numeric(feature_model[col], errors="coerce")
        ql = x.quantile(clip_quantile_low)
        qh = x.quantile(clip_quantile_high)
        if pd.notna(ql) and pd.notna(qh):
            feature_model[col] = x.clip(lower=ql, upper=qh)

    return feature_raw, feature_model, feature_cols_model


def inspect_episode_feature_table(
    out_dir: str,
    clip_quantile_low: float = 0.01,
    clip_quantile_high: float = 0.99,
    n_rows: int = 10,
) -> None:
    """
    审计 episode feature table。
    """
    feature_raw, feature_model, feature_cols_model = build_episode_feature_table(
        out_dir=out_dir,
        clip_quantile_low=clip_quantile_low,
        clip_quantile_high=clip_quantile_high,
    )

    print("\n" + "=" * 120)
    print("STEP2-2 PART1 | episode feature table")
    print("=" * 120)

    print("\n[feature_raw shape]")
    print(feature_raw.shape)

    print("\n[feature_model shape]")
    print(feature_model.shape)

    print("\n[number of model features]")
    print(len(feature_cols_model))

    print("\n[model feature columns]")
    print(feature_cols_model)

    show_cols = [c for c in ["episode_id", "symbol", "session_bucket_std", "episode_start_reason"] + feature_cols_model if c in feature_model.columns]
    print("\n[feature_model head]")
    print(feature_model[show_cols].head(n_rows))

    print("\n[current model NA counts]")
    na_counts = feature_model[feature_cols_model].isna().sum().sort_values(ascending=False)
    print(na_counts)

    print("\n[feature summary statistics]")
    print(feature_model[feature_cols_model].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]])


# =============================================================================
# Step2-2 Part 2：baseline clustering（先做最稳的 KMeans）
# =============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def run_episode_baseline_kmeans(
    out_dir: str,
    clip_quantile_low: float = 0.01,
    clip_quantile_high: float = 0.99,
    pca_n_components: int = 8,
    kmeans_k: int = 8,
    random_state: int = 42,
    kmeans_n_init: int = 1,
    kmeans_max_iter: int = 30,
    silhouette_sample_size: int = 2000,
) -> dict:
    """
    在 episode feature table 基础上运行最稳的 baseline clustering：
    feature table -> remove constant cols -> standardize -> PCA -> KMeans
    """

    feature_raw, feature_model, feature_cols_model = build_episode_feature_table(
        out_dir=out_dir,
        clip_quantile_low=clip_quantile_low,
        clip_quantile_high=clip_quantile_high,
    )

    # -------------------------------------------------------------------------
    # 1) 去掉恒定列 / 近似恒定列
    # -------------------------------------------------------------------------
    usable_feature_cols = []
    dropped_constant_cols = []

    for col in feature_cols_model:
        x = pd.to_numeric(feature_model[col], errors="coerce")
        nunique = x.nunique(dropna=False)
        std = x.std(skipna=True)
        if nunique <= 1 or pd.isna(std) or std == 0:
            dropped_constant_cols.append(col)
        else:
            usable_feature_cols.append(col)

    X_df = feature_model[usable_feature_cols].copy()

    # 再次兜底
    for col in usable_feature_cols:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)

    # -------------------------------------------------------------------------
    # 2) 标准化
    # -------------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    # -------------------------------------------------------------------------
    # 3) PCA
    # -------------------------------------------------------------------------
    pca_n = min(pca_n_components, X_scaled.shape[1], X_scaled.shape[0])
    pca_model = PCA(n_components=pca_n, random_state=random_state)
    X_pca = pca_model.fit_transform(X_scaled)
    X_pca = np.asarray(X_pca, dtype=np.float32)

    # -------------------------------------------------------------------------
    # 4) KMeans
    # -------------------------------------------------------------------------
    kmeans_model = KMeans(
        n_clusters=kmeans_k,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
        random_state=random_state,
        algorithm="elkan",
    )
    kmeans_labels = kmeans_model.fit_predict(X_pca)

    feature_model = feature_model.copy()
    feature_model["cluster_kmeans"] = kmeans_labels

    # -------------------------------------------------------------------------
    # 5) metrics
    # -------------------------------------------------------------------------
    n_unique = len(np.unique(kmeans_labels))

    if n_unique > 1:
        if silhouette_sample_size is not None and len(X_pca) > silhouette_sample_size:
            rng = np.random.default_rng(random_state)
            sample_idx = rng.choice(len(X_pca), size=silhouette_sample_size, replace=False)
            X_sil = X_pca[sample_idx]
            y_sil = kmeans_labels[sample_idx]
            silhouette_val = float(silhouette_score(X_sil, y_sil))
            silhouette_n_used = int(silhouette_sample_size)
        else:
            silhouette_val = float(silhouette_score(X_pca, kmeans_labels))
            silhouette_n_used = int(len(X_pca))

        dbi_val = float(davies_bouldin_score(X_pca, kmeans_labels))
    else:
        silhouette_val = np.nan
        dbi_val = np.nan
        silhouette_n_used = 0

    metrics = {
        "silhouette_score": silhouette_val,
        "davies_bouldin_score": dbi_val,
        "silhouette_n_used": silhouette_n_used,
        "n_clusters": int(n_unique),
        "n_input_features": int(len(usable_feature_cols)),
        "dropped_constant_cols": dropped_constant_cols,
    }

    # -------------------------------------------------------------------------
    # 6) cluster counts
    # -------------------------------------------------------------------------
    cluster_counts = (
        feature_model["cluster_kmeans"]
        .value_counts(dropna=False)
        .sort_index()
        .rename("cluster_size")
        .reset_index()
        .rename(columns={"index": "cluster_kmeans"})
    )

    # -------------------------------------------------------------------------
    # 7) cluster profile
    #    用更易解释的列做画像
    # -------------------------------------------------------------------------
    profile_cols = [
        "n_events",
        "duration_ms",
        "trade_ratio",
        "cancel_ratio",
        "buy_like_ratio",
        "sell_like_ratio",
        "aggressive_ratio",
        "at_bid1_ratio",
        "at_ask1_ratio",
        "inside_spread_ratio",
        "mean_dt_ms",
        "max_dt_ms",
        "mean_qty_log1p",
        "mean_spread_abs",
        "mean_top_depth_total",
        "mean_top_depth_imbalance",
        "delta_snap_last",
        "delta_depth_total",
        "delta_depth_imbalance",
        "side_imbalance",
        "top1_pos_imbalance",
    ]
    profile_cols = [c for c in profile_cols if c in feature_model.columns]

    cluster_profile = (
        feature_model.groupby("cluster_kmeans")[profile_cols]
        .mean()
        .reset_index()
        .sort_values("cluster_kmeans")
    )
    cluster_profile = cluster_profile.merge(cluster_counts, on="cluster_kmeans", how="left")

    return {
        "feature_raw": feature_raw,
        "feature_model": feature_model,
        "feature_cols_model": feature_cols_model,
        "usable_feature_cols": usable_feature_cols,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "scaler": scaler,
        "pca_model": pca_model,
        "kmeans_model": kmeans_model,
        "kmeans_labels": kmeans_labels,
        "metrics": metrics,
        "cluster_counts": cluster_counts,
        "cluster_profile": cluster_profile,
    }


def inspect_episode_baseline_kmeans(
    out_dir: str,
    clip_quantile_low: float = 0.01,
    clip_quantile_high: float = 0.99,
    pca_n_components: int = 8,
    kmeans_k: int = 8,
    random_state: int = 42,
    kmeans_n_init: int = 1,
    kmeans_max_iter: int = 30,
    silhouette_sample_size: int = 2000,
    n_rows: int = 10,
) -> None:
    """
    审计 baseline KMeans clustering 结果。
    """
    res = run_episode_baseline_kmeans(
        out_dir=out_dir,
        clip_quantile_low=clip_quantile_low,
        clip_quantile_high=clip_quantile_high,
        pca_n_components=pca_n_components,
        kmeans_k=kmeans_k,
        random_state=random_state,
        kmeans_n_init=kmeans_n_init,
        kmeans_max_iter=kmeans_max_iter,
        silhouette_sample_size=silhouette_sample_size,
    )

    feature_model = res["feature_model"]
    pca_model = res["pca_model"]

    print("\n" + "=" * 120)
    print("STEP2-2 PART2 | baseline KMeans clustering")
    print("=" * 120)

    print("\n[metrics]")
    print(res["metrics"])

    print("\n[PCA explained variance ratio]")
    print(pca_model.explained_variance_ratio_)
    print("total_explained_variance =", float(np.sum(pca_model.explained_variance_ratio_)))

    print("\n[cluster counts]")
    print(res["cluster_counts"])

    print("\n[cluster profile head]")
    profile_head = res["cluster_profile"].head(n_rows).copy()

    # 只保留前若干列，避免终端渲染超宽表卡住
    max_cols_to_print = 12
    if profile_head.shape[1] > max_cols_to_print:
        profile_head = profile_head.iloc[:, :max_cols_to_print]

    print(profile_head.to_string(index=False))
    print(f"\n[cluster profile shape] {res['cluster_profile'].shape}")
    print(f"[cluster profile columns] {res['cluster_profile'].columns.tolist()[:20]}")
    show_cols = [
        "episode_id",
        "symbol",
        "session_bucket_std",
        "episode_start_reason",
        "cluster_kmeans",
        "n_events",
        "duration_ms",
        "trade_ratio",
        "cancel_ratio",
        "aggressive_ratio",
        "mean_dt_ms",
        "mean_spread_abs",
        "side_imbalance",
        "top1_pos_imbalance",
    ]
    show_cols = [c for c in show_cols if c in feature_model.columns]

    print("\n[sample labeled episodes]")
    print(feature_model[show_cols].head(n_rows))


# =============================================================================
# Step2-3 Part 1：构造 small Transformer 用的 episode sequence dataset
# =============================================================================

def _normalize_str_for_token(x) -> str:
    """
    将 token 组成字段统一成稳定字符串，避免 NaN / None / 空字符串导致 token 爆炸。
    """
    if pd.isna(x):
        return "UNK"
    s = str(x).strip()
    if s == "":
        return "UNK"
    return s

def _normalize_series_for_token(s: pd.Series) -> pd.Series:
    """
    对整列做更快的字符串标准化，尽量避免 pandas StringDtype 的慢路径。
    """
    s = s.astype("object")
    s = s.where(pd.notna(s), "UNK")
    s = s.map(lambda x: str(x).strip() if x is not None else "UNK")
    s = s.mask(s.eq(""), "UNK")
    return s

def build_event_token_vocab_from_es(es: pd.DataFrame, min_freq: int = 1) -> tuple[dict, dict, pd.DataFrame, list[str]]:
    """
    从事件级表构造 token vocabulary。

    token 定义：
    event_family | event_side_simple | event_price_position_std | event_price_aggressive_flag

    返回：
    - token_to_id
    - id_to_token
    - token_freq_table
    """

    required_cols = [
        "event_family",
        "event_side_simple",
        "event_price_position_std",
        "event_price_aggressive_flag",
    ]
    _require_columns(es, required_cols, "build_event_token_vocab_from_es")

    event_family_s = _normalize_series_for_token(es["event_family"]).astype(str).to_numpy()
    event_side_s = _normalize_series_for_token(es["event_side_simple"]).astype(str).to_numpy()
    event_pos_s = _normalize_series_for_token(es["event_price_position_std"]).astype(str).to_numpy()
    event_aggr_s = _normalize_series_for_token(es["event_price_aggressive_flag"]).astype(str).to_numpy()

    from collections import Counter

    token_list = [
        f"{a}|{b}|{c}|{d}"
        for a, b, c, d in zip(event_family_s, event_side_s, event_pos_s, event_aggr_s)
    ]

    print("[Step2-3] token_list built, length =", len(token_list))

    counter = Counter(token_list)
    freq = pd.DataFrame({
        "token": list(counter.keys()),
        "freq": list(counter.values()),
    }).sort_values(["freq", "token"], ascending=[False, True]).reset_index(drop=True)

    print("[Step2-3] token frequency table built, vocab candidates =", len(freq))
    freq = freq.sort_values(["freq", "token"], ascending=[False, True]).reset_index(drop=True)

    # 预留特殊 token
    token_to_id = {
        "[PAD]": 0,
        "[UNK]": 1,
    }

    for _, row in freq.iterrows():
        tok = row["token"]
        f = int(row["freq"])
        if f >= min_freq:
            if tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)

    id_to_token = {v: k for k, v in token_to_id.items()}

    return token_to_id, id_to_token, freq, token_list

def build_episode_sequence_dataset(
    out_dir: str,
    token_min_freq: int = 1,
    max_seq_len: int = 64,
) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    """
    构造 small Transformer 用的 episode sequence dataset。

    返回：
    - seq_df:
        每行一个 episode，包含序列字段（list 形式）
    - token_to_id:
        token -> id
    - id_to_token:
        id -> token
    - token_freq_table:
        token 频数表
    """

    # -------------------------------------------------------------------------
    # 1) 先拿 episode 级切分结果
    # -------------------------------------------------------------------------
    es, episode_table, summary = load_step2_artifacts(out_dir)
    es = es.copy()

    # 避免后面字符串处理和 groupby 被 pyarrow dtype 拖慢
    for col in [
        "event_family",
        "event_side_simple",
        "event_price_position_std",
        "event_price_aggressive_flag",
        "session_bucket_std",
        "episode_cut_reason",
    ]:
        if col in es.columns:
            es[col] = es[col].astype("string")

    # -------------------------------------------------------------------------
    # 2) 先构建 token vocabulary
    # -------------------------------------------------------------------------
    token_to_id, id_to_token, token_freq_table, token_list = build_event_token_vocab_from_es(
        es=es,
        min_freq=token_min_freq,
    )

    unk_id = token_to_id["[UNK]"]
    es["event_token_id"] = [token_to_id.get(x, unk_id) for x in token_list]

    # -------------------------------------------------------------------------
    # 4) 构建 small Transformer 用的连续特征
    # -------------------------------------------------------------------------
    # dt_log1p
    dt_num = pd.to_numeric(es["dt_ms"], errors="coerce").fillna(0).clip(lower=0)
    es["dt_log1p"] = np.log1p(dt_num)

    # qty 已经有 event_qty_log1p
    if "event_qty_log1p" not in es.columns:
        qty_num = pd.to_numeric(es["event_qty"], errors="coerce").fillna(0).clip(lower=0)
        es["event_qty_log1p"] = np.log1p(qty_num)

    # 连续上下文字段统一 numeric
    cont_cols = [
        "dt_log1p",
        "event_qty_log1p",
        "snap_spread_abs",
        "snap_top_depth_imbalance",
    ]
    for col in cont_cols:
        es[col] = pd.to_numeric(es[col], errors="coerce")

    # 缺失值用列中位数填
    for col in cont_cols:
        med = es[col].median(skipna=True)
        if pd.isna(med):
            med = 0.0
        es[col] = es[col].fillna(med)

   # -------------------------------------------------------------------------
    # 5) 按 episode 聚合成序列样本（numpy slicing 版，替代 pandas groupby）
    # -------------------------------------------------------------------------
    seq_rows = []

    group_cols = [
        "episode_id",
        "symbol",
        "session_bucket_std",
        "episode_cut_reason",
        "event_idx",
        "event_time_ms",
        "event_token_id",
        "dt_ms",
        "dt_log1p",
        "event_qty_log1p",
        "snap_spread_abs",
        "snap_top_depth_imbalance",
    ]
    group_cols = [c for c in group_cols if c in es.columns]

    es_seq = es[group_cols].copy()
    es_seq = es_seq.sort_values(["episode_id", "event_idx"], kind="stable").reset_index(drop=True)

    print("[Step2-3] before grouping episodes into sequence rows")

    # 先整体转成 numpy / list，避免每个 group 上反复做 pandas 访问
    episode_id_arr = pd.to_numeric(es_seq["episode_id"], errors="coerce").to_numpy()
    symbol_arr = es_seq["symbol"].astype(object).to_numpy() if "symbol" in es_seq.columns else None
    session_arr = es_seq["session_bucket_std"].astype(object).to_numpy() if "session_bucket_std" in es_seq.columns else None
    cut_reason_arr = es_seq["episode_cut_reason"].astype(object).to_numpy() if "episode_cut_reason" in es_seq.columns else None

    token_id_arr = pd.to_numeric(es_seq["event_token_id"], errors="coerce").fillna(0).astype(int).to_numpy()
    dt_ms_arr = pd.to_numeric(es_seq["dt_ms"], errors="coerce").fillna(0.0).to_numpy()
    dt_log1p_arr = pd.to_numeric(es_seq["dt_log1p"], errors="coerce").fillna(0.0).to_numpy()
    qty_log1p_arr = pd.to_numeric(es_seq["event_qty_log1p"], errors="coerce").fillna(0.0).to_numpy()
    spread_arr = pd.to_numeric(es_seq["snap_spread_abs"], errors="coerce").fillna(0.0).to_numpy()
    imbalance_arr = pd.to_numeric(es_seq["snap_top_depth_imbalance"], errors="coerce").fillna(0.0).to_numpy()

    # 找每个 episode 的边界
    change_idx = np.flatnonzero(np.r_[True, episode_id_arr[1:] != episode_id_arr[:-1]])
    start_idx = change_idx
    end_idx = np.r_[change_idx[1:], len(episode_id_arr)]

    for s, e in zip(start_idx, end_idx):
        episode_id = int(episode_id_arr[s])
        seq_len = int(e - s)

        if seq_len <= 0:
            continue

        # 截断：保留最后 max_seq_len 个事件
        if seq_len > max_seq_len:
            s = e - max_seq_len
            seq_len = max_seq_len

        seq_rows.append({
            "episode_id": episode_id,
            "symbol": symbol_arr[s] if symbol_arr is not None else None,
            "session_bucket_std": session_arr[s] if session_arr is not None else None,
            "episode_cut_reason": cut_reason_arr[s] if cut_reason_arr is not None else None,
            "seq_len": seq_len,

            "event_token_id_seq": token_id_arr[s:e].tolist(),
            "dt_ms_seq": dt_ms_arr[s:e].tolist(),
            "dt_log1p_seq": dt_log1p_arr[s:e].tolist(),
            "event_qty_log1p_seq": qty_log1p_arr[s:e].tolist(),
            "snap_spread_abs_seq": spread_arr[s:e].tolist(),
            "snap_top_depth_imbalance_seq": imbalance_arr[s:e].tolist(),
        })

    print("[Step2-3] finished grouping episodes")
    print("[Step2-3] number of seq rows =", len(seq_rows))

    seq_df = pd.DataFrame(seq_rows)
    return seq_df, token_to_id, id_to_token, token_freq_table


# =============================================================================
# Step2-3 sequence dataset cache
# =============================================================================

from pathlib import Path
import pickle


def _make_sequence_cache_tag(token_min_freq: int, max_seq_len: int) -> str:
    return f"tmf{token_min_freq}_msl{max_seq_len}"


def build_and_save_episode_sequence_artifact(
    out_dir: str,
    token_min_freq: int = 1,
    max_seq_len: int = 64,
) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    """
    第一次从 Step2 artifact 构建 sequence dataset，并落盘缓存。
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tag = _make_sequence_cache_tag(token_min_freq=token_min_freq, max_seq_len=max_seq_len)

    seq_df_file = out_path / f"episode_sequence_dataset_{tag}.pkl"
    vocab_file = out_path / f"episode_sequence_vocab_{tag}.pkl"
    token_freq_file = out_path / f"episode_sequence_token_freq_{tag}.pkl"
    meta_file = out_path / f"episode_sequence_meta_{tag}.txt"

    print("\n" + "=" * 120)
    print("STEP2-3 SEQUENCE CACHE BUILD")
    print("=" * 120)
    print(f"[cache build] out_dir = {out_path}")
    print(f"[cache build] tag = {tag}")

    seq_df, token_to_id, id_to_token, token_freq_table = build_episode_sequence_dataset(
        out_dir=out_dir,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
    )

    seq_df.to_pickle(seq_df_file)
    token_freq_table.to_pickle(token_freq_file)

    with open(vocab_file, "wb") as f:
        pickle.dump(
            {
                "token_to_id": token_to_id,
                "id_to_token": id_to_token,
            },
            f,
        )

    meta_text = (
        f"token_min_freq={token_min_freq}\n"
        f"max_seq_len={max_seq_len}\n"
        f"seq_df_shape={seq_df.shape}\n"
        f"vocab_size={len(token_to_id)}\n"
    )
    meta_file.write_text(meta_text, encoding="utf-8")

    print("[cache build] saved successfully")
    print(f"[cache build] seq_df_file = {seq_df_file}")
    print(f"[cache build] vocab_file = {vocab_file}")
    print(f"[cache build] token_freq_file = {token_freq_file}")
    print(f"[cache build] seq_df shape = {seq_df.shape}")
    print(f"[cache build] vocab size = {len(token_to_id)}")

    return seq_df, token_to_id, id_to_token, token_freq_table


def load_or_build_episode_sequence_artifact(
    out_dir: str,
    token_min_freq: int = 1,
    max_seq_len: int = 64,
    force_rebuild: bool = False,
) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    """
    优先读取 sequence dataset cache；若不存在，则现场构建并保存。
    """
    out_path = Path(out_dir)
    tag = _make_sequence_cache_tag(token_min_freq=token_min_freq, max_seq_len=max_seq_len)

    seq_df_file = out_path / f"episode_sequence_dataset_{tag}.pkl"
    vocab_file = out_path / f"episode_sequence_vocab_{tag}.pkl"
    token_freq_file = out_path / f"episode_sequence_token_freq_{tag}.pkl"

    cache_exists = seq_df_file.exists() and vocab_file.exists() and token_freq_file.exists()

    if (not force_rebuild) and cache_exists:
        print("\n" + "=" * 120)
        print("STEP2-3 SEQUENCE CACHE LOAD")
        print("=" * 120)
        print(f"[cache load] out_dir = {out_path}")
        print(f"[cache load] tag = {tag}")

        seq_df = pd.read_pickle(seq_df_file)
        token_freq_table = pd.read_pickle(token_freq_file)

        with open(vocab_file, "rb") as f:
            vocab_obj = pickle.load(f)

        token_to_id = vocab_obj["token_to_id"]
        id_to_token = vocab_obj["id_to_token"]

        print(f"[cache load] seq_df shape = {seq_df.shape}")
        print(f"[cache load] vocab size = {len(token_to_id)}")

        return seq_df, token_to_id, id_to_token, token_freq_table

    return build_and_save_episode_sequence_artifact(
        out_dir=out_dir,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
    )


def inspect_episode_sequence_dataset(
    out_dir: str,
    token_min_freq: int = 1,
    max_seq_len: int = 64,
    n_rows: int = 5,
) -> None:
    """
    审计 episode sequence dataset。
    """
    seq_df, token_to_id, id_to_token, token_freq_table = build_episode_sequence_dataset(
        out_dir=out_dir,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
    )

    print("\n" + "=" * 120)
    print("STEP2-3 PART1 | episode sequence dataset")
    print("=" * 120)

    print("\n[seq_df shape]")
    print(seq_df.shape)

    print("\n[vocab size]")
    print(len(token_to_id))

    print("\n[top token frequencies]")
    print(token_freq_table.head(20))

    print("\n[seq_len summary]")
    print(pd.to_numeric(seq_df["seq_len"], errors="coerce").describe())

    print("\n[episode_cut_reason counts]")
    if "episode_cut_reason" in seq_df.columns:
        print(seq_df["episode_cut_reason"].value_counts(dropna=False))
    else:
        print("episode_cut_reason not found")

    print("\n[sample rows]")
    show_cols = [
        "episode_id",
        "symbol",
        "session_bucket_std",
        "episode_cut_reason",
        "seq_len",
    ]
    show_cols = [c for c in show_cols if c in seq_df.columns]
    print(seq_df[show_cols].head(n_rows))

    if len(seq_df) > 0:
        print("\n[first sample token id seq]")
        print(seq_df["event_token_id_seq"].iloc[0][:20])

        print("\n[first sample dt_log1p_seq]")
        print(seq_df["dt_log1p_seq"].iloc[0][:20])

        print("\n[first sample event_qty_log1p_seq]")
        print(seq_df["event_qty_log1p_seq"].iloc[0][:20])

        print("\n[first sample snap_spread_abs_seq]")
        print(seq_df["snap_spread_abs_seq"].iloc[0][:20])

        print("\n[first sample snap_top_depth_imbalance_seq]")
        print(seq_df["snap_top_depth_imbalance_seq"].iloc[0][:20])


# =============================================================================
# Step2 Artifact Layer：落盘与读取
# =============================================================================

def build_and_save_step2_artifacts(
    symbol: str,
    out_dir: str,
    keep_sessions: Optional[List[str]] = None,
    gap_hard_ms: int = 3000,
    lag_hard_ms: int = 5000,
    max_events_per_episode: int = 64,
    max_duration_ms: int = 10000,
    min_events_soft_cut: int = 8,
    min_events_episode_flag: int = 4,
    step1_artifact_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    构建 Step2 的核心中间产物，并保存到 parquet。

    保存文件：
    - es_with_episode.parquet
    - episode_table.parquet
    - episode_summary.parquet
    - meta.json（简单参数记录，可选）

    返回：
    - es
    - episode_table
    - summary
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 120)
    print("STEP2 ARTIFACT BUILD")
    print("=" * 120)
    print(f"[artifact] output dir = {out_path}")

    es, episode_table, summary = build_episode_candidates(
        symbol=symbol,
        keep_sessions=keep_sessions,
        gap_hard_ms=gap_hard_ms,
        lag_hard_ms=lag_hard_ms,
        max_events_per_episode=max_events_per_episode,
        max_duration_ms=max_duration_ms,
        min_events_soft_cut=min_events_soft_cut,
        min_events_episode_flag=min_events_episode_flag,
        step1_artifact_dir=step1_artifact_dir,
    )

    es_file = out_path / "es_minimal_for_sequence.pkl"
    episode_table_file = out_path / "episode_table.pkl"
    summary_file = out_path / "episode_summary.pkl"
    meta_file = out_path / "meta.txt"

    # 只保留后面 sequence dataset 和回看真正要用到的事件级列
    minimal_es_cols = [
        "episode_id",
        "symbol",
        "session_bucket_std",
        "episode_cut_reason",
        "event_idx",
        "event_time_ms",
        "event_family",
        "event_side_simple",
        "event_price_position_std",
        "event_price_aggressive_flag",
        "dt_ms",
        "event_qty",
        "event_qty_log1p",
        "snap_spread_abs",
        "snap_top_depth_imbalance",
    ]
    minimal_es_cols = [c for c in minimal_es_cols if c in es.columns]

    es_minimal = es[minimal_es_cols].copy()

    # 明确把数值列转 numeric，减少 parquet 写入负担
    for col in [
        "episode_id",
        "event_idx",
        "event_time_ms",
        "dt_ms",
        "event_qty",
        "event_qty_log1p",
        "snap_spread_abs",
        "snap_top_depth_imbalance",
    ]:
        if col in es_minimal.columns:
            es_minimal[col] = pd.to_numeric(es_minimal[col], errors="coerce")

    # 字符串列统一成普通 string/object，避免奇怪 dtype
    for col in [
        "symbol",
        "session_bucket_std",
        "episode_cut_reason",
        "event_family",
        "event_side_simple",
        "event_price_position_std",
        "event_price_aggressive_flag",
    ]:
        if col in es_minimal.columns:
            es_minimal[col] = es_minimal[col].astype("string")

    print(f"[artifact] saving minimal es -> {es_file}")
    print(f"[artifact] minimal es shape = {es_minimal.shape}")
    es_minimal.to_pickle(es_file)

    print(f"[artifact] saving episode_table -> {episode_table_file}")
    episode_table.to_pickle(episode_table_file)

    print(f"[artifact] saving summary -> {summary_file}")
    summary.to_pickle(summary_file)

    meta_text = f"""symbol={symbol}
    keep_sessions={keep_sessions}
    gap_hard_ms={gap_hard_ms}
    lag_hard_ms={lag_hard_ms}
    max_events_per_episode={max_events_per_episode}
    max_duration_ms={max_duration_ms}
    min_events_soft_cut={min_events_soft_cut}
    min_events_episode_flag={min_events_episode_flag}
    es_minimal_shape={es_minimal.shape}
    episode_table_shape={episode_table.shape}
    summary_shape={summary.shape}
    """
    meta_file.write_text(meta_text, encoding="utf-8")

    print("[artifact] saved successfully")
    print(f"[artifact] es shape = {es.shape}")
    print(f"[artifact] episode_table shape = {episode_table.shape}")
    print(f"[artifact] summary shape = {summary.shape}")

    return es, episode_table, summary


def load_step2_artifacts(out_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    从 parquet 读取 Step2 中间产物。
    """
    out_path = Path(out_dir)

    es_file = out_path / "es_minimal_for_sequence.pkl"
    episode_table_file = out_path / "episode_table.pkl"
    summary_file = out_path / "episode_summary.pkl"

    if not es_file.exists():
        raise FileNotFoundError(f"missing artifact file: {es_file}")
    if not episode_table_file.exists():
        raise FileNotFoundError(f"missing artifact file: {episode_table_file}")
    if not summary_file.exists():
        raise FileNotFoundError(f"missing artifact file: {summary_file}")

    print("\n" + "=" * 120)
    print("STEP2 ARTIFACT LOAD")
    print("=" * 120)
    print(f"[artifact] loading from = {out_path}")

    es = pd.read_pickle(es_file)
    episode_table = pd.read_pickle(episode_table_file)
    summary = pd.read_pickle(summary_file)

    print(f"[artifact] es shape = {es.shape}")
    print(f"[artifact] episode_table shape = {episode_table.shape}")
    print(f"[artifact] summary shape = {summary.shape}")

    return es, episode_table, summary


def inspect_step2_artifacts(out_dir: str, n: int = 10) -> None:
    """
    快速检查 pickle 是否保存正确。
    """
    es, episode_table, summary = load_step2_artifacts(out_dir)

    print("\n[es head]")
    print(es.head(n))

    print("\n[episode_table head]")
    print(episode_table.head(n))

    print("\n[summary head]")
    print(summary.head(n))


# =============================================================================
# Step2-3 Part 2：PyTorch Dataset + collate_fn
# =============================================================================

import torch
from torch.utils.data import Dataset, DataLoader


class EpisodeSequenceTorchDataset(Dataset):
    """
    small Transformer 用的 episode sequence dataset。

    每个样本返回：
    - episode_id
    - seq_len
    - token_ids: List[int]
    - cont_feats: List[List[float]]，每个事件一个连续特征向量
    """

    def __init__(self, seq_df: pd.DataFrame):
        self.seq_df = seq_df.reset_index(drop=True).copy()

        required_cols = [
            "episode_id",
            "seq_len",
            "event_token_id_seq",
            "dt_log1p_seq",
            "event_qty_log1p_seq",
            "snap_spread_abs_seq",
            "snap_top_depth_imbalance_seq",
        ]
        missing = [c for c in required_cols if c not in self.seq_df.columns]
        if missing:
            raise ValueError(f"EpisodeSequenceTorchDataset missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.seq_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.seq_df.iloc[idx]

        token_ids = row["event_token_id_seq"]
        dt_log1p_seq = row["dt_log1p_seq"]
        qty_log1p_seq = row["event_qty_log1p_seq"]
        spread_seq = row["snap_spread_abs_seq"]
        imbalance_seq = row["snap_top_depth_imbalance_seq"]

        seq_len = int(row["seq_len"])

        # 防御式检查：长度要一致
        lengths = [
            len(token_ids),
            len(dt_log1p_seq),
            len(qty_log1p_seq),
            len(spread_seq),
            len(imbalance_seq),
        ]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Length mismatch at idx={idx}, episode_id={row['episode_id']}: {lengths}"
            )

        cont_feats = list(zip(
            dt_log1p_seq,
            qty_log1p_seq,
            spread_seq,
            imbalance_seq,
        ))

        return {
            "episode_id": int(row["episode_id"]),
            "seq_len": seq_len,
            "token_ids": token_ids,
            "cont_feats": cont_feats,
        }


def make_episode_sequence_collate_fn(pad_token_id: int = 0):
    """
    返回一个 collate_fn，用于将变长序列 padding 成 batch 张量。
    """

    def collate_fn(batch: list[dict]) -> dict:
        if len(batch) == 0:
            raise ValueError("Empty batch in collate_fn")

        batch_size = len(batch)
        seq_lens = [int(x["seq_len"]) for x in batch]
        max_len = max(seq_lens)

        cont_dim = len(batch[0]["cont_feats"][0]) if batch[0]["seq_len"] > 0 else 4

        token_tensor = torch.full(
            (batch_size, max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )

        cont_tensor = torch.zeros(
            (batch_size, max_len, cont_dim),
            dtype=torch.float32,
        )

        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool,
        )

        episode_ids = []
        seq_len_tensor = torch.tensor(seq_lens, dtype=torch.long)

        for i, sample in enumerate(batch):
            cur_len = sample["seq_len"]
            episode_ids.append(sample["episode_id"])

            if cur_len == 0:
                continue

            token_ids = torch.tensor(sample["token_ids"], dtype=torch.long)
            cont_feats = torch.tensor(sample["cont_feats"], dtype=torch.float32)

            token_tensor[i, :cur_len] = token_ids
            cont_tensor[i, :cur_len, :] = cont_feats
            attention_mask[i, :cur_len] = True

        return {
            "episode_id": torch.tensor(episode_ids, dtype=torch.long),
            "seq_len": seq_len_tensor,
            "token_ids": token_tensor,
            "cont_feats": cont_tensor,
            "attention_mask": attention_mask,
        }

    return collate_fn


def build_torch_episode_dataset_from_artifacts(
    out_dir: str,
    token_min_freq: int = 1,
    max_seq_len: int = 64,
    force_rebuild_sequence_cache: bool = False,
    token_to_id_override: Optional[dict] = None,
) -> tuple[EpisodeSequenceTorchDataset, dict, dict, pd.DataFrame]:
    seq_df, token_to_id_cache, id_to_token_cache, token_freq_table = load_or_build_episode_sequence_artifact(
        out_dir=out_dir,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
        force_rebuild=force_rebuild_sequence_cache,
    )

    if token_to_id_override is not None:
        seq_df = remap_seq_df_token_ids_to_external_vocab(
            seq_df=seq_df,
            cached_id_to_token=id_to_token_cache,
            external_token_to_id=token_to_id_override,
        )
        dataset = EpisodeSequenceTorchDataset(seq_df)
        id_to_token_override = {v: k for k, v in token_to_id_override.items()}
        return dataset, token_to_id_override, id_to_token_override, seq_df

    dataset = EpisodeSequenceTorchDataset(seq_df)
    return dataset, token_to_id_cache, id_to_token_cache, seq_df


def inspect_torch_episode_dataloader(
    out_dir: str,
    token_min_freq: int = 1,
    max_seq_len: int = 64,
    batch_size: int = 8,
    n_preview_rows: int = 3,
) -> None:
    """
    检查 PyTorch Dataset + DataLoader + collate_fn 是否正确。
    """
    dataset, token_to_id, id_to_token, seq_df = build_torch_episode_dataset_from_artifacts(
        out_dir=out_dir,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
    )

    pad_token_id = token_to_id["[PAD]"]
    collate_fn = make_episode_sequence_collate_fn(pad_token_id=pad_token_id)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    batch = next(iter(loader))

    print("\n" + "=" * 120)
    print("STEP2-3 PART2 | torch dataset + dataloader")
    print("=" * 120)

    print("\n[dataset size]")
    print(len(dataset))

    print("\n[vocab size]")
    print(len(token_to_id))

    print("\n[seq_df shape]")
    print(seq_df.shape)

    print("\n[batch keys]")
    print(batch.keys())

    print("\n[batch tensor shapes]")
    print("episode_id     :", tuple(batch["episode_id"].shape))
    print("seq_len        :", tuple(batch["seq_len"].shape))
    print("token_ids      :", tuple(batch["token_ids"].shape))
    print("cont_feats     :", tuple(batch["cont_feats"].shape))
    print("attention_mask :", tuple(batch["attention_mask"].shape))

    print("\n[batch seq_len]")
    print(batch["seq_len"])

    print("\n[first token_ids rows]")
    print(batch["token_ids"][:n_preview_rows])

    print("\n[first attention_mask rows]")
    print(batch["attention_mask"][:n_preview_rows])

    print("\n[first cont_feats rows, first 5 steps]")
    print(batch["cont_feats"][:n_preview_rows, :5, :])

    # decode first sample first few token ids
    first_ids = batch["token_ids"][0].tolist()
    first_mask = batch["attention_mask"][0].tolist()

    decoded = []
    for tid, m in zip(first_ids, first_mask):
        if not m:
            break
        decoded.append(id_to_token.get(int(tid), "[UNK]"))

    print("\n[first sample decoded tokens]")
    print(decoded[:20])


# =============================================================================
# Step2-4 Part 1：small Transformer encoder + mean pooling（forward 检查）
# =============================================================================

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    标准 sinusoidal positional encoding
    输入输出 shape: [B, T, D]
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, T, D]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class SmallEpisodeTransformer(nn.Module):
    """
    small Transformer episode encoder

    输入:
    - token_ids: [B, T]
    - cont_feats: [B, T, C]
    - attention_mask: [B, T]，True 表示有效位置

    输出:
    - hidden: [B, T, D]
    - pooled: [B, D]
    """
    def __init__(
        self,
        vocab_size: int,
        cont_dim: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.cont_dim = cont_dim
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_token_id,
        )

        self.cont_projection = nn.Sequential(
            nn.Linear(cont_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.input_dropout = nn.Dropout(dropout)

    def masked_mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden: [B, T, D]
        attention_mask: [B, T]，True 表示有效位置
        return: [B, D]
        """
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)   # [B, T, 1]
        hidden_sum = (hidden * mask).sum(dim=1)                # [B, D]
        denom = mask.sum(dim=1).clamp(min=1.0)                 # [B, 1]
        pooled = hidden_sum / denom
        return pooled

    def forward(
        self,
        token_ids: torch.Tensor,
        cont_feats: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        token_ids: [B, T]
        cont_feats: [B, T, C]
        attention_mask: [B, T]，True=有效，False=padding
        """
        token_embed = self.token_embedding(token_ids)          # [B, T, D]
        cont_embed = self.cont_projection(cont_feats)          # [B, T, D]

        x = token_embed + cont_embed
        x = self.positional_encoding(x)
        x = self.input_dropout(x)

        # Transformer 的 src_key_padding_mask: True 表示 padding，需要屏蔽
        src_key_padding_mask = ~attention_mask                 # [B, T]

        hidden = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )                                                     # [B, T, D]

        pooled = self.masked_mean_pool(hidden, attention_mask)

        return {
            "hidden": hidden,
            "pooled": pooled,
        }


def inspect_small_transformer_forward(
    out_dir: str,
    token_min_freq: int = 1,
    max_seq_len: int = 64,
    batch_size: int = 8,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    max_len: int = 128,
) -> None:
    """
    只做 small Transformer forward 检查，不训练。
    """
    dataset, token_to_id, id_to_token, seq_df = build_torch_episode_dataset_from_artifacts(
        out_dir=out_dir,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
    )

    pad_token_id = token_to_id["[PAD]"]
    collate_fn = make_episode_sequence_collate_fn(pad_token_id=pad_token_id)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    batch = next(iter(loader))

    model = SmallEpisodeTransformer(
        vocab_size=len(token_to_id),
        cont_dim=batch["cont_feats"].shape[-1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
        pad_token_id=pad_token_id,
    )

    model.eval()
    with torch.no_grad():
        out = model(
            token_ids=batch["token_ids"],
            cont_feats=batch["cont_feats"],
            attention_mask=batch["attention_mask"],
        )

    hidden = out["hidden"]
    pooled = out["pooled"]

    print("\n" + "=" * 120)
    print("STEP2-4 PART1 | small transformer forward")
    print("=" * 120)

    print("\n[vocab size]")
    print(len(token_to_id))

    print("\n[input batch shapes]")
    print("token_ids      :", tuple(batch["token_ids"].shape))
    print("cont_feats     :", tuple(batch["cont_feats"].shape))
    print("attention_mask :", tuple(batch["attention_mask"].shape))

    print("\n[model config]")
    print({
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "max_len": max_len,
        "pad_token_id": pad_token_id,
    })

    print("\n[output shapes]")
    print("hidden :", tuple(hidden.shape))
    print("pooled :", tuple(pooled.shape))

    print("\n[first 3 pooled vectors, first 8 dims]")
    print(pooled[:3, :8])

    print("\n[attention valid counts]")
    print(batch["attention_mask"].sum(dim=1))

    print("\n[check finite]")
    print("hidden finite =", torch.isfinite(hidden).all().item())
    print("pooled finite =", torch.isfinite(pooled).all().item())


# =============================================================================
# Step2-4 Part 2：small Transformer + MLM training
# =============================================================================

import random
from dataclasses import dataclass


@dataclass
class MLMBatch:
    token_ids_input: torch.Tensor      # [B, T]
    token_labels: torch.Tensor         # [B, T], 非 mask 位 = -100
    cont_feats: torch.Tensor           # [B, T, C]
    attention_mask: torch.Tensor       # [B, T]
    episode_id: torch.Tensor           # [B]
    seq_len: torch.Tensor              # [B]


def mask_tokens_for_mlm(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int,
    mask_token_id: int,
    mask_prob: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    标准 BERT 风格 MLM:
    - 只在有效 token 上 mask
    - labels 里未 mask 的位置设为 -100
    - 80% -> [MASK]
    - 10% -> 随机 token
    - 10% -> 保持原 token
    """
    device = token_ids.device

    token_ids_input = token_ids.clone()
    labels = torch.full_like(token_ids, fill_value=-100)

    valid_mask = attention_mask & (token_ids != pad_token_id)

    # 采样哪些位置要被 mask
    rand = torch.rand(token_ids.shape, device=device)
    mask_positions = (rand < mask_prob) & valid_mask

    labels[mask_positions] = token_ids[mask_positions]

    # BERT 风格替换
    replace_rand = torch.rand(token_ids.shape, device=device)

    # 80% -> [MASK]
    mask_mask = mask_positions & (replace_rand < 0.8)
    token_ids_input[mask_mask] = mask_token_id

    # 10% -> random token
    random_mask = mask_positions & (replace_rand >= 0.8) & (replace_rand < 0.9)
    if random_mask.any():
        vocab_upper = int(token_ids.max().item()) + 1
        random_tokens = torch.randint(
            low=0,
            high=max(vocab_upper, mask_token_id + 1),
            size=token_ids.shape,
            device=device,
            dtype=torch.long,
        )
        token_ids_input[random_mask] = random_tokens[random_mask]

    # 10% -> 保持原 token，不用额外处理

    return token_ids_input, labels


class SmallEpisodeTransformerMLM(nn.Module):
    """
    在 SmallEpisodeTransformer 上增加 token prediction head
    """
    def __init__(
        self,
        vocab_size: int,
        cont_dim: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.encoder_model = SmallEpisodeTransformer(
            vocab_size=vocab_size,
            cont_dim=cont_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            pad_token_id=pad_token_id,
        )

        self.token_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        cont_feats: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        out = self.encoder_model(
            token_ids=token_ids,
            cont_feats=cont_feats,
            attention_mask=attention_mask,
        )
        hidden = out["hidden"]
        pooled = out["pooled"]

        logits = self.token_head(hidden)   # [B, T, V]

        return {
            "hidden": hidden,
            "pooled": pooled,
            "logits": logits,
        }


def build_mlm_dataloader_from_artifacts(
    out_dir: str,
    token_min_freq: int = 1,
    max_seq_len: int = 64,
    batch_size: int = 64,
    shuffle: bool = True,
    force_rebuild_sequence_cache: bool = False,
    token_to_id_override: Optional[dict] = None,
):
    """
    构造 MLM 训练用 DataLoader
    """
    dataset, token_to_id, id_to_token, seq_df = build_torch_episode_dataset_from_artifacts(
        out_dir=out_dir,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
        force_rebuild_sequence_cache=force_rebuild_sequence_cache,
        token_to_id_override=token_to_id_override,
    )

    pad_token_id = token_to_id["[PAD]"]
    collate_fn = make_episode_sequence_collate_fn(pad_token_id=pad_token_id)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
    )

    return loader, token_to_id, id_to_token, seq_df


def train_small_transformer_mlm(
    out_dir: str,
    token_min_freq: int = 1,
    max_seq_len: int = 64,
    batch_size: int = 32,
    num_epochs: int = 3,
    max_batches_per_epoch: Optional[int] = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    max_len: int = 128,
    mask_prob: float = 0.15,
    device: Optional[str] = None,
) -> tuple[nn.Module, dict, dict]:
    """
    第一版 MLM 训练。
    返回：
    - model
    - token_to_id
    - id_to_token
    """
    loader, token_to_id, id_to_token, seq_df = build_mlm_dataloader_from_artifacts(
        out_dir=out_dir,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=True,
    )

    pad_token_id = token_to_id["[PAD]"]

    # 新增一个 [MASK] token
    if "[MASK]" not in token_to_id:
        token_to_id = token_to_id.copy()
        token_to_id["[MASK]"] = len(token_to_id)
        id_to_token = {v: k for k, v in token_to_id.items()}

    mask_token_id = token_to_id["[MASK]"]

    if device is None:
        device = "cpu"

    model = SmallEpisodeTransformerMLM(
        vocab_size=len(token_to_id),
        cont_dim=4,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
        pad_token_id=pad_token_id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print("\n" + "=" * 120)
    print("STEP2-4 PART2 | small transformer MLM training")
    print("=" * 120)
    print(f"[train] device = {device}")
    print(f"[train] dataset size = {len(seq_df)}")
    print(f"[train] vocab size = {len(token_to_id)}")
    print(f"[train] batch_size = {batch_size}")

    for epoch in range(1, num_epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_masked = 0

        for batch_idx, batch in enumerate(loader):
            token_ids = batch["token_ids"].to(device)
            cont_feats = batch["cont_feats"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            token_ids_input, token_labels = mask_tokens_for_mlm(
                token_ids=token_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                mask_token_id=mask_token_id,
                mask_prob=mask_prob,
            )

            out = model(
                token_ids=token_ids_input,
                cont_feats=cont_feats,
                attention_mask=attention_mask,
            )

            logits = out["logits"]  # [B, T, V]
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                token_labels.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_masked += int((token_labels != -100).sum().item())

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"[epoch {epoch}/{num_epochs}] "
                    f"batch={batch_idx+1}/{len(loader)} "
                    f"loss={loss.item():.4f}"
                )
            
            if max_batches_per_epoch is not None and (batch_idx + 1) >= max_batches_per_epoch:
                break

            n_batches_ran = batch_idx + 1 if 'batch_idx' in locals() else 0
            avg_loss = epoch_loss / max(n_batches_ran, 1)
            print(
                f"[epoch {epoch}/{num_epochs}] "
                f"avg_loss={avg_loss:.4f} "
                f"masked_tokens={epoch_masked} "
                f"batches_ran={n_batches_ran}"
            )

    return model, token_to_id, id_to_token


def extract_episode_embeddings(
    model: nn.Module,
    out_dir: str,
    token_to_id: dict,
    token_min_freq: int = 1,
    max_seq_len: int = 64,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    用训练后的模型提取每个 episode 的 pooled embedding。
    返回：
    - embedding_df: [episode_id, emb_0, emb_1, ...]
    """
    loader, _, _, seq_df = build_mlm_dataloader_from_artifacts(
        out_dir=out_dir,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=False,
        token_to_id_override=token_to_id,
    )

    if device is None:
        device = "cpu"

    model.eval()
    model.to(device)

    rows = []

    with torch.no_grad():
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            cont_feats = batch["cont_feats"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            episode_ids = batch["episode_id"].cpu().tolist()

            out = model.encoder_model(
                token_ids=token_ids,
                cont_feats=cont_feats,
                attention_mask=attention_mask,
            )
            pooled = out["pooled"].cpu().numpy()

            for ep_id, vec in zip(episode_ids, pooled):
                row = {"episode_id": int(ep_id)}
                for j, val in enumerate(vec):
                    row[f"emb_{j}"] = float(val)
                rows.append(row)

    embedding_df = pd.DataFrame(rows)
    return embedding_df


def inspect_small_transformer_mlm_train(
    out_dir: str,
    token_min_freq: int = 1,
    max_seq_len: int = 64,
    batch_size: int = 32,
    num_epochs: int = 2,
    lr: float = 1e-3,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    max_len: int = 128,
    mask_prob: float = 0.15,
    extract_batch_size: int = 64,
    device: Optional[str] = None,
    max_batches_per_epoch: Optional[int] = None,
) -> None:
    """
    训练一个最小可用的 MLM 版本，并导出 embedding 预览。
    """
    model, token_to_id, id_to_token = train_small_transformer_mlm(
        out_dir=out_dir,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        num_epochs=num_epochs,
        max_batches_per_epoch=max_batches_per_epoch,
        lr=lr,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
        mask_prob=mask_prob,
        device=device,
    )

    embedding_df = extract_episode_embeddings(
        model=model,
        out_dir=out_dir,
        token_to_id=token_to_id,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
        batch_size=extract_batch_size,
        device=device,
    )

    print("\n" + "=" * 120)
    print("STEP2-4 PART2 | embedding extraction preview")
    print("=" * 120)

    print("\n[embedding_df shape]")
    print(embedding_df.shape)

    print("\n[embedding_df head]")
    print(embedding_df.head(5))


# =============================================================================
# Step2 baseline cluster audit
# =============================================================================

from pathlib import Path


def export_step2_baseline_cluster_audit(
    out_dir: str,
    clip_quantile_low: float = 0.01,
    clip_quantile_high: float = 0.99,
    pca_n_components: int = 8,
    kmeans_k: int = 8,
    random_state: int = 42,
    kmeans_n_init: int = 1,
    kmeans_max_iter: int = 30,
    silhouette_sample_size: int = 2000,
    sample_per_cluster: int = 20,
) -> dict:
    """
    基于现有 Step2 artifacts，导出 baseline clustering 的审计结果：
    1) cluster_summary.csv
    2) cluster_topdiff_features.csv
    3) cluster_episode_samples.csv
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1) baseline KMeans（基于现有 artifacts）
    # ---------------------------------------------------------------------
    res = run_episode_baseline_kmeans(
        out_dir=out_dir,
        clip_quantile_low=clip_quantile_low,
        clip_quantile_high=clip_quantile_high,
        pca_n_components=pca_n_components,
        kmeans_k=kmeans_k,
        random_state=random_state,
        kmeans_n_init=kmeans_n_init,
        kmeans_max_iter=kmeans_max_iter,
        silhouette_sample_size=silhouette_sample_size,
    )

    # ---------------------------------------------------------------------
    # 2) 读取 summary + feature table
    # ---------------------------------------------------------------------
    _, _, summary = load_step2_artifacts(out_dir)
    summary = summary.copy()

    feature_raw, feature_model, feature_cols_model = build_episode_feature_table(
        out_dir=out_dir,
        clip_quantile_low=clip_quantile_low,
        clip_quantile_high=clip_quantile_high,
    )

    # ---------------------------------------------------------------------
    # 3) 取聚类标签
    # ---------------------------------------------------------------------
    labels_df = res["feature_model"].copy()
    if "episode_id" not in labels_df.columns:
        raise ValueError("feature_model must contain episode_id")
    if "cluster_kmeans" not in labels_df.columns:
        raise ValueError("feature_model must contain cluster_kmeans")

    summary = summary.merge(
        labels_df[["episode_id", "cluster_kmeans"]],
        on="episode_id",
        how="left",
        validate="one_to_one",
    )

    feature_model = feature_model.merge(
        labels_df[["episode_id", "cluster_kmeans"]],
        on="episode_id",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------------
    # 4) 合并成 audit_base
    # ---------------------------------------------------------------------
    audit_base = summary.merge(
        feature_model,
        on="episode_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_fm"),
    )

    # ---------------------------------------------------------------------
    # 5) cluster summary
    # ---------------------------------------------------------------------
    agg_spec = {
        "episode_id": "count",
        "n_events": ["mean", "median"],
        "duration_ms": ["mean", "median"],
        "n_trade": "mean",
        "n_add": "mean",
        "n_cancel": "mean",
        "trade_ratio": "mean",
        "add_ratio": "mean",
        "cancel_ratio": "mean",
        "aggressive_ratio": "mean",
        "at_bid1_ratio": "mean",
        "at_ask1_ratio": "mean",
        "inside_spread_ratio": "mean",
    }
    available_agg_spec = {k: v for k, v in agg_spec.items() if k in audit_base.columns}

    cluster_summary = (
        audit_base
        .groupby("cluster_kmeans", dropna=False)
        .agg(available_agg_spec)
    )

    flat_cols = []
    for col in cluster_summary.columns:
        if isinstance(col, tuple):
            base, stat = col
            if base == "episode_id" and stat == "count":
                flat_cols.append("n_episodes")
            elif stat == "mean":
                flat_cols.append(f"mean_{base}")
            elif stat == "median":
                flat_cols.append(f"median_{base}")
            else:
                flat_cols.append(f"{base}_{stat}")
        else:
            flat_cols.append(str(col))

    cluster_summary.columns = flat_cols
    cluster_summary = (
        cluster_summary
        .reset_index()
        .sort_values("cluster_kmeans")
        .reset_index(drop=True)
    )

    cluster_summary_file = out_path / "cluster_summary.csv"
    cluster_summary.to_csv(cluster_summary_file, index=False)

    # ---------------------------------------------------------------------
    # 6) top-diff features
    # ---------------------------------------------------------------------
    model_feature_cols = res["usable_feature_cols"]
    global_mean = feature_model[model_feature_cols].mean(axis=0)

    topdiff_rows = []
    for cl, sub in feature_model.groupby("cluster_kmeans", dropna=False):
        cl_mean = sub[model_feature_cols].mean(axis=0)
        diff = (cl_mean - global_mean).sort_values(key=lambda s: s.abs(), ascending=False)

        top_k = min(8, len(diff))
        for rank, feat in enumerate(diff.index[:top_k], start=1):
            topdiff_rows.append({
                "cluster_kmeans": cl,
                "rank": rank,
                "feature": feat,
                "cluster_mean": float(cl_mean[feat]),
                "global_mean": float(global_mean[feat]),
                "diff_vs_global": float(diff[feat]),
                "abs_diff_vs_global": float(abs(diff[feat])),
            })

    cluster_topdiff = (
        pd.DataFrame(topdiff_rows)
        .sort_values(["cluster_kmeans", "rank"], ascending=[True, True])
        .reset_index(drop=True)
    )

    cluster_topdiff_file = out_path / "cluster_topdiff_features.csv"
    cluster_topdiff.to_csv(cluster_topdiff_file, index=False)

    # ---------------------------------------------------------------------
    # 7) per-cluster sampled episodes
    # ---------------------------------------------------------------------
    audit_cols = [
        "episode_id",
        "cluster_kmeans",
        "start_event_idx",
        "end_event_idx",
        "start_time_ms",
        "end_time_ms",
        "episode_start_reason",
        "n_events",
        "duration_ms",
        "n_trade",
        "n_add",
        "n_cancel",
        "trade_ratio",
        "add_ratio",
        "cancel_ratio",
        "aggressive_ratio",
        "at_bid1_ratio",
        "at_ask1_ratio",
        "inside_spread_ratio",
    ]
    audit_cols = [c for c in audit_cols if c in audit_base.columns]

    sample_rows = []
    rng = np.random.default_rng(random_state)

    for cl, sub in audit_base.groupby("cluster_kmeans", dropna=False):
        n_take = min(sample_per_cluster, len(sub))
        picked_idx = rng.choice(sub.index.to_numpy(), size=n_take, replace=False)
        sub_sample = sub.loc[picked_idx, audit_cols].copy()
        sub_sample = sub_sample.sort_values(
            ["duration_ms", "n_events"], ascending=[False, False]
        ).reset_index(drop=True)
        sample_rows.append(sub_sample)

    cluster_samples = pd.concat(sample_rows, axis=0, ignore_index=True)

    cluster_samples_file = out_path / "cluster_episode_samples.csv"
    cluster_samples.to_csv(cluster_samples_file, index=False)

    # ---------------------------------------------------------------------
    # 8) report
    # ---------------------------------------------------------------------
    metrics = res["metrics"]
    pca_ratio = res["pca_model"].explained_variance_ratio_
    cluster_counts = res["cluster_counts"]

    report_lines = [
        "STEP2 BASELINE CLUSTER AUDIT",
        "=" * 80,
        f"out_dir = {out_path.resolve()}",
        "",
        "[metrics]",
        str(metrics),
        "",
        "[pca_explained_variance_ratio]",
        str(pca_ratio),
        "",
        "[cluster_counts]",
        str(cluster_counts),
        "",
        f"cluster_summary_file = {cluster_summary_file}",
        f"cluster_topdiff_file = {cluster_topdiff_file}",
        f"cluster_samples_file = {cluster_samples_file}",
    ]

    report_file = out_path / "cluster_audit_report.txt"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "metrics": metrics,
        "pca_explained_variance_ratio": pca_ratio,
        "cluster_counts": cluster_counts,
        "cluster_summary": cluster_summary,
        "cluster_topdiff": cluster_topdiff,
        "cluster_samples": cluster_samples,
        "files": {
            "cluster_summary_file": str(cluster_summary_file),
            "cluster_topdiff_file": str(cluster_topdiff_file),
            "cluster_samples_file": str(cluster_samples_file),
            "report_file": str(report_file),
        },
    }


def run_episode_baseline_kmeans_custom(
    out_dir: str,
    clip_quantile_low: float = 0.01,
    clip_quantile_high: float = 0.99,
    pca_n_components: int = 8,
    kmeans_k: int = 8,
    random_state: int = 42,
    kmeans_n_init: int = 1,
    kmeans_max_iter: int = 30,
    silhouette_sample_size: int = 2000,
    drop_feature_cols: Optional[list[str]] = None,
) -> dict:
    """
    在现有 baseline KMeans 基础上，支持人为剔除部分特征做稳健性分析。
    不修改原始 artifact，只在 feature_model 上删列后重新聚类。
    """

    feature_raw, feature_model, feature_cols_model = build_episode_feature_table(
        out_dir=out_dir,
        clip_quantile_low=clip_quantile_low,
        clip_quantile_high=clip_quantile_high,
    )

    drop_feature_cols = drop_feature_cols or []
    drop_feature_cols = [c for c in drop_feature_cols if c in feature_cols_model]

    usable_feature_cols = [c for c in feature_cols_model if c not in drop_feature_cols]
    if len(usable_feature_cols) == 0:
        raise ValueError("No usable feature columns left after dropping requested features.")

    X = feature_model[usable_feature_cols].copy()

    # 去常数列
    nunique = X.nunique(dropna=False)
    dropped_constant_cols = nunique[nunique <= 1].index.tolist()
    usable_feature_cols = [c for c in usable_feature_cols if c not in dropped_constant_cols]

    X = feature_model[usable_feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_model = PCA(n_components=min(pca_n_components, X_scaled.shape[1]), random_state=random_state)
    X_pca = pca_model.fit_transform(X_scaled)
    X_pca = np.asarray(X_pca, dtype=np.float32)

    kmeans_model = KMeans(
        n_clusters=kmeans_k,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
        random_state=random_state,
        algorithm="elkan",
    )
    kmeans_labels = kmeans_model.fit_predict(X_pca)

    n_unique = len(np.unique(kmeans_labels))
    if n_unique > 1:
        if silhouette_sample_size is not None and len(X_pca) > silhouette_sample_size:
            rng = np.random.default_rng(random_state)
            sample_idx = rng.choice(len(X_pca), size=silhouette_sample_size, replace=False)
            X_sil = X_pca[sample_idx]
            y_sil = kmeans_labels[sample_idx]
            silhouette_val = float(silhouette_score(X_sil, y_sil))
            silhouette_n_used = int(silhouette_sample_size)
        else:
            silhouette_val = float(silhouette_score(X_pca, kmeans_labels))
            silhouette_n_used = int(len(X_pca))

        dbi_val = float(davies_bouldin_score(X_pca, kmeans_labels))
    else:
        silhouette_val = np.nan
        dbi_val = np.nan
        silhouette_n_used = 0

    metrics = {
        "silhouette_score": silhouette_val,
        "davies_bouldin_score": dbi_val,
        "silhouette_n_used": silhouette_n_used,
        "n_clusters": int(n_unique),
        "n_input_features": int(len(usable_feature_cols)),
        "dropped_constant_cols": dropped_constant_cols,
        "dropped_by_request_cols": drop_feature_cols,
    }

    feature_model_out = feature_model.copy()
    feature_model_out["cluster_kmeans"] = kmeans_labels

    cluster_counts = (
        feature_model_out.groupby("cluster_kmeans", dropna=False)
        .size()
        .reset_index(name="cluster_size")
        .sort_values("cluster_kmeans")
        .reset_index(drop=True)
    )

    cluster_profile = (
        feature_model_out.groupby("cluster_kmeans", dropna=False)[usable_feature_cols]
        .mean()
        .reset_index()
        .sort_values("cluster_kmeans")
        .reset_index(drop=True)
    )

    return {
        "feature_raw": feature_raw,
        "feature_model": feature_model_out,
        "usable_feature_cols": usable_feature_cols,
        "scaler": scaler,
        "pca_model": pca_model,
        "kmeans_model": kmeans_model,
        "kmeans_labels": kmeans_labels,
        "cluster_counts": cluster_counts,
        "cluster_profile": cluster_profile,
        "metrics": metrics,
    }

def export_step2_baseline_sensitivity(
    out_dir: str,
    run_name: str,
    drop_feature_cols: Optional[list[str]] = None,
    clip_quantile_low: float = 0.01,
    clip_quantile_high: float = 0.99,
    pca_n_components: int = 8,
    kmeans_k: int = 8,
    random_state: int = 42,
    kmeans_n_init: int = 1,
    kmeans_max_iter: int = 30,
    silhouette_sample_size: int = 2000,
) -> dict:
    """
    针对指定删列方案，导出一个 sensitivity run 的结果。
    """

    res = run_episode_baseline_kmeans_custom(
        out_dir=out_dir,
        clip_quantile_low=clip_quantile_low,
        clip_quantile_high=clip_quantile_high,
        pca_n_components=pca_n_components,
        kmeans_k=kmeans_k,
        random_state=random_state,
        kmeans_n_init=kmeans_n_init,
        kmeans_max_iter=kmeans_max_iter,
        silhouette_sample_size=silhouette_sample_size,
        drop_feature_cols=drop_feature_cols,
    )

    run_dir = Path(out_dir) / f"sensitivity_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cluster_counts_file = run_dir / "cluster_counts.csv"
    cluster_profile_file = run_dir / "cluster_profile.csv"
    report_file = run_dir / "report.txt"

    res["cluster_counts"].to_csv(cluster_counts_file, index=False)
    res["cluster_profile"].to_csv(cluster_profile_file, index=False)

    report_lines = [
        f"STEP2 BASELINE SENSITIVITY | {run_name}",
        "=" * 80,
        f"out_dir = {out_dir}",
        f"run_dir = {run_dir}",
        "",
        "[metrics]",
        str(res["metrics"]),
        "",
        "[pca_explained_variance_ratio]",
        str(res["pca_model"].explained_variance_ratio_),
        "",
        "[usable_feature_cols]",
        str(res["usable_feature_cols"]),
        "",
        "[cluster_counts]",
        str(res["cluster_counts"]),
    ]
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "run_name": run_name,
        "metrics": res["metrics"],
        "pca_explained_variance_ratio": res["pca_model"].explained_variance_ratio_,
        "cluster_counts": res["cluster_counts"],
        "cluster_profile": res["cluster_profile"],
        "usable_feature_cols": res["usable_feature_cols"],
        "files": {
            "cluster_counts_file": str(cluster_counts_file),
            "cluster_profile_file": str(cluster_profile_file),
            "report_file": str(report_file),
        },
    }


def export_step2_cross_run_cluster_matching(
    base_run_dir: str,
    target_run_dir: str,
    base_run_name: str,
    target_run_name: str,
    out_dir: str,
) -> dict:
    """
    对两个 sensitivity run 的 cluster_profile 做 cluster matching。
    使用公共特征列，基于 Pearson correlation 做 Hungarian matching。
    """

    from pathlib import Path
    from scipy.optimize import linear_sum_assignment
    from scipy.stats import pearsonr

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base_profile_file = Path(base_run_dir) / "cluster_profile.csv"
    target_profile_file = Path(target_run_dir) / "cluster_profile.csv"

    if not base_profile_file.exists():
        raise FileNotFoundError(f"missing base profile file: {base_profile_file}")
    if not target_profile_file.exists():
        raise FileNotFoundError(f"missing target profile file: {target_profile_file}")

    base_df = pd.read_csv(base_profile_file)
    target_df = pd.read_csv(target_profile_file)

    if "cluster_kmeans" not in base_df.columns:
        raise ValueError(f"{base_profile_file} missing cluster_kmeans")
    if "cluster_kmeans" not in target_df.columns:
        raise ValueError(f"{target_profile_file} missing cluster_kmeans")

    # 只保留公共特征列
    exclude_cols = {"cluster_kmeans"}
    base_features = [c for c in base_df.columns if c not in exclude_cols]
    target_features = [c for c in target_df.columns if c not in exclude_cols]
    common_features = sorted(list(set(base_features).intersection(set(target_features))))

    if len(common_features) == 0:
        raise ValueError("No common feature columns found between two cluster profiles.")

    base_mat = base_df[common_features].to_numpy(dtype=float)
    target_mat = target_df[common_features].to_numpy(dtype=float)

    n_base = base_mat.shape[0]
    n_target = target_mat.shape[0]

    corr_mat = np.zeros((n_base, n_target), dtype=float)
    dist_mat = np.zeros((n_base, n_target), dtype=float)

    for i in range(n_base):
        for j in range(n_target):
            x = base_mat[i]
            y = target_mat[j]

            # Pearson corr
            if np.std(x) == 0 or np.std(y) == 0:
                corr_val = np.nan
            else:
                corr_val = pearsonr(x, y)[0]

            if np.isnan(corr_val):
                corr_val = -1.0

            corr_mat[i, j] = corr_val
            dist_mat[i, j] = float(np.linalg.norm(x - y))

    cost_mat = 1.0 - corr_mat
    row_ind, col_ind = linear_sum_assignment(cost_mat)

    match_rows = []
    for r, c in zip(row_ind, col_ind):
        match_rows.append({
            "base_run": base_run_name,
            "target_run": target_run_name,
            "base_cluster": int(base_df.loc[r, "cluster_kmeans"]),
            "target_cluster": int(target_df.loc[c, "cluster_kmeans"]),
            "corr": float(corr_mat[r, c]),
            "distance": float(dist_mat[r, c]),
            "n_common_features": int(len(common_features)),
        })

    match_df = pd.DataFrame(match_rows).sort_values(
        ["base_cluster"], ascending=[True]
    ).reset_index(drop=True)

    corr_df = pd.DataFrame(
        corr_mat,
        index=[f"base_{int(x)}" for x in base_df["cluster_kmeans"].tolist()],
        columns=[f"target_{int(x)}" for x in target_df["cluster_kmeans"].tolist()],
    )

    dist_df = pd.DataFrame(
        dist_mat,
        index=[f"base_{int(x)}" for x in base_df["cluster_kmeans"].tolist()],
        columns=[f"target_{int(x)}" for x in target_df["cluster_kmeans"].tolist()],
    )

    match_file = out_path / f"match_{base_run_name}_vs_{target_run_name}.csv"
    corr_file = out_path / f"corr_matrix_{base_run_name}_vs_{target_run_name}.csv"
    dist_file = out_path / f"dist_matrix_{base_run_name}_vs_{target_run_name}.csv"
    report_file = out_path / f"report_{base_run_name}_vs_{target_run_name}.txt"

    match_df.to_csv(match_file, index=False)
    corr_df.to_csv(corr_file)
    dist_df.to_csv(dist_file)

    report_lines = [
        f"STEP2 CROSS-RUN CLUSTER MATCHING | {base_run_name} vs {target_run_name}",
        "=" * 80,
        f"base_run_dir = {base_run_dir}",
        f"target_run_dir = {target_run_dir}",
        "",
        f"n_common_features = {len(common_features)}",
        "",
        "[matched pairs]",
        str(match_df),
        "",
        f"mean_corr = {match_df['corr'].mean():.6f}",
        f"min_corr = {match_df['corr'].min():.6f}",
        f"max_corr = {match_df['corr'].max():.6f}",
        "",
        f"match_file = {match_file}",
        f"corr_file = {corr_file}",
        f"dist_file = {dist_file}",
    ]
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "match_df": match_df,
        "corr_df": corr_df,
        "dist_df": dist_df,
        "common_features": common_features,
        "files": {
            "match_file": str(match_file),
            "corr_file": str(corr_file),
            "dist_file": str(dist_file),
            "report_file": str(report_file),
        },
    }

def export_step2_final_baseline_package(
    out_dir: str,
    final_run_name: str = "final_baseline_no_delta_depth_total",
    drop_feature_cols: Optional[list[str]] = None,
    clip_quantile_low: float = 0.01,
    clip_quantile_high: float = 0.99,
    pca_n_components: int = 8,
    kmeans_k: int = 8,
    random_state: int = 42,
    kmeans_n_init: int = 1,
    kmeans_max_iter: int = 30,
    silhouette_sample_size: int = 2000,
) -> dict:
    """
    固化最终 baseline clustering 版本，供后续 Transformer 对比使用。
    当前建议用于 no_delta_depth_total 版本。
    """

    run_dir = Path(out_dir) / final_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    res = run_episode_baseline_kmeans_custom(
        out_dir=out_dir,
        clip_quantile_low=clip_quantile_low,
        clip_quantile_high=clip_quantile_high,
        pca_n_components=pca_n_components,
        kmeans_k=kmeans_k,
        random_state=random_state,
        kmeans_n_init=kmeans_n_init,
        kmeans_max_iter=kmeans_max_iter,
        silhouette_sample_size=silhouette_sample_size,
        drop_feature_cols=drop_feature_cols,
    )

    _, _, summary = load_step2_artifacts(out_dir)
    summary = summary.copy()

    feature_model = res["feature_model"].copy()

    audit_base = summary.merge(
        feature_model,
        on="episode_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_fm"),
    )

    # 1) final cluster counts
    cluster_counts = res["cluster_counts"].copy()
    cluster_counts_file = run_dir / "final_cluster_counts.csv"
    cluster_counts.to_csv(cluster_counts_file, index=False)

    # 2) final cluster profile
    cluster_profile = res["cluster_profile"].copy()
    cluster_profile_file = run_dir / "final_cluster_profile.csv"
    cluster_profile.to_csv(cluster_profile_file, index=False)

    # 3) final cluster summary
    agg_spec = {
        "episode_id": "count",
        "n_events": ["mean", "median"],
        "duration_ms": ["mean", "median"],
        "n_trade": "mean",
        "n_add": "mean",
        "n_cancel": "mean",
        "trade_ratio": "mean",
        "add_ratio": "mean",
        "cancel_ratio": "mean",
        "aggressive_ratio": "mean",
        "at_bid1_ratio": "mean",
        "at_ask1_ratio": "mean",
        "inside_spread_ratio": "mean",
    }
    available_agg_spec = {k: v for k, v in agg_spec.items() if k in audit_base.columns}

    cluster_summary = (
        audit_base
        .groupby("cluster_kmeans", dropna=False)
        .agg(available_agg_spec)
    )

    flat_cols = []
    for col in cluster_summary.columns:
        if isinstance(col, tuple):
            base, stat = col
            if base == "episode_id" and stat == "count":
                flat_cols.append("n_episodes")
            elif stat == "mean":
                flat_cols.append(f"mean_{base}")
            elif stat == "median":
                flat_cols.append(f"median_{base}")
            else:
                flat_cols.append(f"{base}_{stat}")
        else:
            flat_cols.append(str(col))

    cluster_summary.columns = flat_cols
    cluster_summary = (
        cluster_summary
        .reset_index()
        .sort_values("cluster_kmeans")
        .reset_index(drop=True)
    )

    cluster_summary_file = run_dir / "final_cluster_summary.csv"
    cluster_summary.to_csv(cluster_summary_file, index=False)

    # 4) final feature assignment
    feature_assignment_file = run_dir / "final_feature_model_with_cluster.csv"
    feature_model.to_csv(feature_assignment_file, index=False)

    # 5) final episode assignment
    episode_assignment_cols = [c for c in [
        "episode_id",
        "cluster_kmeans",
        "episode_start_reason",
        "n_events",
        "duration_ms",
        "n_trade",
        "n_add",
        "n_cancel",
        "trade_ratio",
        "add_ratio",
        "cancel_ratio",
        "aggressive_ratio",
        "at_bid1_ratio",
        "at_ask1_ratio",
        "inside_spread_ratio",
    ] if c in audit_base.columns]

    episode_assignment = audit_base[episode_assignment_cols].copy()
    episode_assignment_file = run_dir / "final_episode_assignment.csv"
    episode_assignment.to_csv(episode_assignment_file, index=False)

    # 6) protocol note
    report_lines = [
        "STEP2 FINAL BASELINE PACKAGE",
        "=" * 80,
        f"out_dir = {out_dir}",
        f"final_run_name = {final_run_name}",
        "",
        "[baseline protocol]",
        f"drop_feature_cols = {drop_feature_cols}",
        f"pca_n_components = {pca_n_components}",
        f"kmeans_k = {kmeans_k}",
        f"random_state = {random_state}",
        f"kmeans_n_init = {kmeans_n_init}",
        f"kmeans_max_iter = {kmeans_max_iter}",
        f"silhouette_sample_size = {silhouette_sample_size}",
        "",
        "[metrics]",
        str(res["metrics"]),
        "",
        "[pca_explained_variance_ratio]",
        str(res["pca_model"].explained_variance_ratio_),
        "",
        "[usable_feature_cols]",
        str(res["usable_feature_cols"]),
        "",
        "[cluster_counts]",
        str(cluster_counts),
        "",
        f"cluster_counts_file = {cluster_counts_file}",
        f"cluster_profile_file = {cluster_profile_file}",
        f"cluster_summary_file = {cluster_summary_file}",
        f"feature_assignment_file = {feature_assignment_file}",
        f"episode_assignment_file = {episode_assignment_file}",
    ]
    report_file = run_dir / "final_baseline_report.txt"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "metrics": res["metrics"],
        "pca_explained_variance_ratio": res["pca_model"].explained_variance_ratio_,
        "cluster_counts": cluster_counts,
        "cluster_profile": cluster_profile,
        "cluster_summary": cluster_summary,
        "usable_feature_cols": res["usable_feature_cols"],
        "files": {
            "cluster_counts_file": str(cluster_counts_file),
            "cluster_profile_file": str(cluster_profile_file),
            "cluster_summary_file": str(cluster_summary_file),
            "feature_assignment_file": str(feature_assignment_file),
            "episode_assignment_file": str(episode_assignment_file),
            "report_file": str(report_file),
        },
    }


def export_step2_tempo_sensitive_cluster_audit(
    out_dir: str,
    matching_dir: str,
    base_run_name: str = "baseline",
    target_run_name: str = "no_tempo",
    sample_per_cluster: int = 30,
    suspicious_corr_threshold: float = 0.0,
) -> dict:
    """
    检查 baseline vs no_tempo 中 corr 很低或为负的异常 cluster matching，
    导出 baseline cluster 和 target cluster 的样本及 profile，便于人工审查。
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    match_file = Path(matching_dir) / f"match_{base_run_name}_vs_{target_run_name}.csv"
    if not match_file.exists():
        raise FileNotFoundError(f"missing match file: {match_file}")

    match_df = pd.read_csv(match_file)

    suspicious_df = match_df.loc[match_df["corr"] <= suspicious_corr_threshold].copy()
    suspicious_df = suspicious_df.sort_values(["corr", "base_cluster"], ascending=[True, True]).reset_index(drop=True)

    if suspicious_df.empty:
        report_file = out_path / "tempo_sensitive_audit_report.txt"
        report_file.write_text(
            "No suspicious matched clusters found under current threshold.",
            encoding="utf-8",
        )
        return {
            "suspicious_matches": suspicious_df,
            "files": {
                "report_file": str(report_file),
            },
        }

    base_dir = Path(out_dir).parent / "sensitivity_baseline"
    target_dir = Path(out_dir).parent / "sensitivity_no_tempo"

    base_assign = pd.read_csv(base_dir / "cluster_profile.csv")
    target_assign = pd.read_csv(target_dir / "cluster_profile.csv")

    # 读取 final assignments
    base_feature_res = run_episode_baseline_kmeans_custom(
        out_dir=str(Path(out_dir).parent),
        drop_feature_cols=[],
        clip_quantile_low=0.01,
        clip_quantile_high=0.99,
        pca_n_components=8,
        kmeans_k=8,
        random_state=42,
        kmeans_n_init=1,
        kmeans_max_iter=30,
        silhouette_sample_size=2000,
    )
    target_feature_res = run_episode_baseline_kmeans_custom(
        out_dir=str(Path(out_dir).parent),
        drop_feature_cols=[
            "duration_ms_log1p",
            "mean_dt_ms_log1p",
            "median_dt_ms_log1p",
            "max_dt_ms_log1p",
            "min_dt_ms_log1p",
        ],
        clip_quantile_low=0.01,
        clip_quantile_high=0.99,
        pca_n_components=8,
        kmeans_k=8,
        random_state=42,
        kmeans_n_init=1,
        kmeans_max_iter=30,
        silhouette_sample_size=2000,
    )

    _, _, summary = load_step2_artifacts(str(Path(out_dir).parent))
    summary = summary.copy()

    base_assign_df = summary.merge(
        base_feature_res["feature_model"][["episode_id", "cluster_kmeans"]],
        on="episode_id",
        how="left",
        validate="one_to_one",
    )
    target_assign_df = summary.merge(
        target_feature_res["feature_model"][["episode_id", "cluster_kmeans"]],
        on="episode_id",
        how="left",
        validate="one_to_one",
    )

    sample_cols = [c for c in [
        "episode_id",
        "cluster_kmeans",
        "episode_start_reason",
        "n_events",
        "duration_ms",
        "n_trade",
        "n_add",
        "n_cancel",
    ] if c in base_assign_df.columns]

    rng = np.random.default_rng(42)
    collected_samples = []

    for _, row in suspicious_df.iterrows():
        base_cluster = int(row["base_cluster"])
        target_cluster = int(row["target_cluster"])

        base_sub = base_assign_df.loc[base_assign_df["cluster_kmeans"] == base_cluster].copy()
        target_sub = target_assign_df.loc[target_assign_df["cluster_kmeans"] == target_cluster].copy()

        n_base_take = min(sample_per_cluster, len(base_sub))
        n_target_take = min(sample_per_cluster, len(target_sub))

        if n_base_take > 0:
            base_idx = rng.choice(base_sub.index.to_numpy(), size=n_base_take, replace=False)
            base_sample = base_sub.loc[base_idx, sample_cols].copy()
            base_sample["source_run"] = "baseline"
            base_sample["matched_base_cluster"] = base_cluster
            base_sample["matched_target_cluster"] = target_cluster
            base_sample["match_corr"] = row["corr"]
            collected_samples.append(base_sample)

        if n_target_take > 0:
            target_idx = rng.choice(target_sub.index.to_numpy(), size=n_target_take, replace=False)
            target_sample = target_sub.loc[target_idx, sample_cols].copy()
            target_sample["source_run"] = "no_tempo"
            target_sample["matched_base_cluster"] = base_cluster
            target_sample["matched_target_cluster"] = target_cluster
            target_sample["match_corr"] = row["corr"]
            collected_samples.append(target_sample)

    suspicious_samples = pd.concat(collected_samples, axis=0, ignore_index=True)

    suspicious_match_file = out_path / "suspicious_matches.csv"
    suspicious_samples_file = out_path / "suspicious_match_samples.csv"
    suspicious_profile_file = out_path / "suspicious_matches_summary.csv"
    report_file = out_path / "tempo_sensitive_audit_report.txt"

    suspicious_df.to_csv(suspicious_match_file, index=False)
    suspicious_samples.to_csv(suspicious_samples_file, index=False)

    suspicious_profile_rows = []
    for _, row in suspicious_df.iterrows():
        suspicious_profile_rows.append({
            "base_cluster": int(row["base_cluster"]),
            "target_cluster": int(row["target_cluster"]),
            "corr": float(row["corr"]),
            "distance": float(row["distance"]),
            "comment": "Needs manual audit: possible tempo-sensitive cluster or bad one-to-one match.",
        })
    suspicious_profile_df = pd.DataFrame(suspicious_profile_rows)
    suspicious_profile_df.to_csv(suspicious_profile_file, index=False)

    report_lines = [
        "STEP2 TEMPO-SENSITIVE CLUSTER AUDIT",
        "=" * 80,
        f"matching_dir = {matching_dir}",
        f"suspicious_corr_threshold = {suspicious_corr_threshold}",
        "",
        "[suspicious_matches]",
        str(suspicious_df),
        "",
        f"suspicious_match_file = {suspicious_match_file}",
        f"suspicious_samples_file = {suspicious_samples_file}",
        f"suspicious_profile_file = {suspicious_profile_file}",
    ]
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "suspicious_matches": suspicious_df,
        "suspicious_samples": suspicious_samples,
        "files": {
            "suspicious_match_file": str(suspicious_match_file),
            "suspicious_samples_file": str(suspicious_samples_file),
            "suspicious_profile_file": str(suspicious_profile_file),
            "report_file": str(report_file),
        },
    }


# =============================================================================
# Step2-4 Part 0：Transformer input audit
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd


def audit_transformer_episode_input_vs_final_baseline(
    out_dir: str,
    final_baseline_dir: str,
    token_min_freq: int = 1,
    max_seq_len: int = 64,
    show_n_mismatch: int = 20,
) -> dict:
    """
    目标：
    1) 检查 Transformer sequence dataset 和 final baseline 是否使用同一批 episode
    2) 给出 sequence 长度 / cont feature / vocab 的基础统计
    3) 只做审计，不训练，不改 baseline

    参数：
    - out_dir:
        Step2 artifact 主目录（sequence dataset 从这里读）
    - final_baseline_dir:
        final baseline 结果目录，例如：
        /home/bu-yuting/新建文件夹/step2_outputs_validate_baseline_v2/final_baseline_no_delta_depth_total
    """

    print("\n" + "=" * 120)
    print("STEP2-4 PART0 | TRANSFORMER INPUT AUDIT")
    print("=" * 120)
    print(f"[out_dir] = {out_dir}")
    print(f"[final_baseline_dir] = {final_baseline_dir}")

    # ---------------------------------------------------------------------
    # 1) sequence side
    # ---------------------------------------------------------------------
    seq_df, token_to_id, id_to_token, token_freq_table = build_episode_sequence_dataset(
        out_dir=out_dir,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
    )

    if seq_df.empty:
        raise ValueError("sequence dataset is empty")

    seq_episode_ids = set(pd.to_numeric(seq_df["episode_id"], errors="coerce").dropna().astype(int).tolist())
    seq_len_num = pd.to_numeric(seq_df["seq_len"], errors="coerce")

    # 用第一行判断当前 cont feature 维度
    cont_seq_cols = [
        "dt_log1p_seq",
        "event_qty_log1p_seq",
        "snap_spread_abs_seq",
        "snap_top_depth_imbalance_seq",
    ]
    existing_cont_seq_cols = [c for c in cont_seq_cols if c in seq_df.columns]
    cont_dim = len(existing_cont_seq_cols)

    # token 相关统计
    vocab_size = len(token_to_id)
    token_freq_top = token_freq_table.head(10).copy()

    # ---------------------------------------------------------------------
    # 2) baseline side
    #    优先读 final_feature_model_with_cluster.csv
    #    如果没有，再 fallback 到 final_episode_assignment.csv
    # ---------------------------------------------------------------------
    baseline_dir = Path(final_baseline_dir)
    feature_file = baseline_dir / "final_feature_model_with_cluster.csv"
    episode_assignment_file = baseline_dir / "final_episode_assignment.csv"

    baseline_df = None
    baseline_source = None

    if feature_file.exists():
        baseline_df = pd.read_csv(feature_file)
        baseline_source = str(feature_file)
    elif episode_assignment_file.exists():
        baseline_df = pd.read_csv(episode_assignment_file)
        baseline_source = str(episode_assignment_file)
    else:
        raise FileNotFoundError(
            "Neither final_feature_model_with_cluster.csv nor final_episode_assignment.csv "
            f"exists under {final_baseline_dir}"
        )

    if "episode_id" not in baseline_df.columns:
        raise ValueError(f"baseline file missing episode_id column: {baseline_source}")

    baseline_episode_ids = set(
        pd.to_numeric(baseline_df["episode_id"], errors="coerce").dropna().astype(int).tolist()
    )

    # ---------------------------------------------------------------------
    # 3) set comparison
    # ---------------------------------------------------------------------
    common_episode_ids = seq_episode_ids & baseline_episode_ids
    only_in_sequence = sorted(seq_episode_ids - baseline_episode_ids)
    only_in_baseline = sorted(baseline_episode_ids - seq_episode_ids)

    # ---------------------------------------------------------------------
    # 4) 打印摘要
    # ---------------------------------------------------------------------
    print("\n[sequence dataset]")
    print("seq_df shape              :", seq_df.shape)
    print("sequence episode count    :", len(seq_episode_ids))
    print("vocab size                :", vocab_size)
    print("cont_dim                  :", cont_dim)

    print("\n[baseline dataset]")
    print("baseline source           :", baseline_source)
    print("baseline df shape         :", baseline_df.shape)
    print("baseline episode count    :", len(baseline_episode_ids))

    print("\n[episode universe comparison]")
    print("common episode count      :", len(common_episode_ids))
    print("only_in_sequence count    :", len(only_in_sequence))
    print("only_in_baseline count    :", len(only_in_baseline))
    print("exact_match               :", len(only_in_sequence) == 0 and len(only_in_baseline) == 0)

    print("\n[sequence length summary]")
    print(seq_len_num.describe(percentiles=[0.5, 0.9, 0.99]))

    print("\n[token frequency top 10]")
    print(token_freq_top)

    print("\n[first mismatches | only_in_sequence]")
    print(only_in_sequence[:show_n_mismatch])

    print("\n[first mismatches | only_in_baseline]")
    print(only_in_baseline[:show_n_mismatch])

    # ---------------------------------------------------------------------
    # 5) 再用 torch dataset + collate 检查一次 batch 维度
    # ---------------------------------------------------------------------
    dataset, token_to_id_2, id_to_token_2, seq_df_2 = build_torch_episode_dataset_from_artifacts(
        out_dir=out_dir,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
    )

    pad_token_id = token_to_id_2["[PAD]"]
    collate_fn = make_episode_sequence_collate_fn(pad_token_id=pad_token_id)

    sample_n = min(8, len(dataset))
    sample_batch = [dataset[i] for i in range(sample_n)]
    batch = collate_fn(sample_batch)

    print("\n[torch batch shapes]")
    print("token_ids      :", tuple(batch["token_ids"].shape))
    print("cont_feats     :", tuple(batch["cont_feats"].shape))
    print("attention_mask :", tuple(batch["attention_mask"].shape))
    print("seq_len        :", tuple(batch["seq_len"].shape))
    print("episode_id     :", tuple(batch["episode_id"].shape))

    print("\n[torch batch finite check]")
    print("token_ids finite      =", torch.isfinite(batch["token_ids"]).all().item())
    print("cont_feats finite     =", torch.isfinite(batch["cont_feats"]).all().item())
    print("attention_mask dtype  =", batch["attention_mask"].dtype)

    result = {
        "sequence_episode_count": len(seq_episode_ids),
        "baseline_episode_count": len(baseline_episode_ids),
        "common_episode_count": len(common_episode_ids),
        "only_in_sequence_count": len(only_in_sequence),
        "only_in_baseline_count": len(only_in_baseline),
        "only_in_sequence_head": only_in_sequence[:show_n_mismatch],
        "only_in_baseline_head": only_in_baseline[:show_n_mismatch],
        "seq_len_summary": seq_len_num.describe(percentiles=[0.5, 0.9, 0.99]).to_dict(),
        "vocab_size": vocab_size,
        "cont_dim": cont_dim,
        "baseline_source": baseline_source,
        "exact_match": len(only_in_sequence) == 0 and len(only_in_baseline) == 0,
    }

    return result


# =============================================================================
# Step2-4 Part 2：learned embedding clustering（与 final baseline 公平对齐）
# =============================================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def run_learned_embedding_kmeans(
    embedding_csv: str,
    baseline_feature_csv: str,
    pca_n_components: int = 8,
    kmeans_k: int = 8,
    random_state: int = 42,
    kmeans_n_init: int = 1,
    kmeans_max_iter: int = 30,
    silhouette_sample_size: int = 2000,
) -> dict:
    """
    对 learned episode embedding 做 clustering。
    protocol 尽量与 final baseline 一致：
    embedding -> standardize -> PCA -> KMeans

    同时用 final baseline feature table 里的解释列给 learned cluster 做 profile。
    """

    print("\n" + "=" * 120)
    print("STEP2-4 PART2 | LEARNED EMBEDDING KMEANS")
    print("=" * 120)
    print(f"[embedding_csv] = {embedding_csv}")
    print(f"[baseline_feature_csv] = {baseline_feature_csv}")

    emb_df = pd.read_csv(embedding_csv)
    base_df = pd.read_csv(baseline_feature_csv)

    if "episode_id" not in emb_df.columns:
        raise ValueError("embedding_csv missing episode_id")
    if "episode_id" not in base_df.columns:
        raise ValueError("baseline_feature_csv missing episode_id")

    emb_df["episode_id"] = pd.to_numeric(emb_df["episode_id"], errors="coerce").astype("Int64")
    base_df["episode_id"] = pd.to_numeric(base_df["episode_id"], errors="coerce").astype("Int64")

    emb_df = emb_df.dropna(subset=["episode_id"]).copy()
    base_df = base_df.dropna(subset=["episode_id"]).copy()

    emb_df["episode_id"] = emb_df["episode_id"].astype(int)
    base_df["episode_id"] = base_df["episode_id"].astype(int)

    # ------------------------------------------------------------------
    # 1) 对齐 episode universe
    # ------------------------------------------------------------------
    common_ids = sorted(set(emb_df["episode_id"]) & set(base_df["episode_id"]))
    if len(common_ids) == 0:
        raise ValueError("No common episode_id between embedding and baseline feature table")

    emb_df = emb_df[emb_df["episode_id"].isin(common_ids)].copy()
    base_df = base_df[base_df["episode_id"].isin(common_ids)].copy()

    emb_df = emb_df.sort_values("episode_id").reset_index(drop=True)
    base_df = base_df.sort_values("episode_id").reset_index(drop=True)

    if not np.array_equal(emb_df["episode_id"].to_numpy(), base_df["episode_id"].to_numpy()):
        raise ValueError("episode_id order mismatch after alignment")

    # ------------------------------------------------------------------
    # 2) 取 embedding 列
    # ------------------------------------------------------------------
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    if len(emb_cols) == 0:
        raise ValueError("No emb_* columns found in embedding_csv")

    X_df = emb_df[emb_cols].copy()
    for col in emb_cols:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0.0)

    # ------------------------------------------------------------------
    # 3) 标准化 -> PCA -> KMeans
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    pca_n = min(pca_n_components, X_scaled.shape[1], X_scaled.shape[0])
    pca_model = PCA(n_components=pca_n, random_state=random_state)
    X_pca = pca_model.fit_transform(X_scaled)
    X_pca = np.asarray(X_pca, dtype=np.float32)

    kmeans_model = KMeans(
        n_clusters=kmeans_k,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
        random_state=random_state,
        algorithm="elkan",
    )
    labels = kmeans_model.fit_predict(X_pca)

    learned_assignment = emb_df[["episode_id"]].copy()
    learned_assignment["cluster_kmeans"] = labels

    # ------------------------------------------------------------------
    # 4) metrics
    # ------------------------------------------------------------------
    n_unique = len(np.unique(labels))

    if n_unique > 1:
        if silhouette_sample_size is not None and len(X_pca) > silhouette_sample_size:
            rng = np.random.default_rng(random_state)
            sample_idx = rng.choice(len(X_pca), size=silhouette_sample_size, replace=False)
            X_sil = X_pca[sample_idx]
            y_sil = labels[sample_idx]
            silhouette_val = float(silhouette_score(X_sil, y_sil))
            silhouette_n_used = int(silhouette_sample_size)
        else:
            silhouette_val = float(silhouette_score(X_pca, labels))
            silhouette_n_used = int(len(X_pca))

        dbi_val = float(davies_bouldin_score(X_pca, labels))
    else:
        silhouette_val = np.nan
        dbi_val = np.nan
        silhouette_n_used = 0

    metrics = {
        "silhouette_score": silhouette_val,
        "davies_bouldin_score": dbi_val,
        "silhouette_n_used": silhouette_n_used,
        "n_clusters": int(n_unique),
        "n_embedding_dims": int(len(emb_cols)),
        "n_common_episodes": int(len(common_ids)),
    }

    # ------------------------------------------------------------------
    # 5) cluster counts
    # ------------------------------------------------------------------
    cluster_counts = (
        learned_assignment["cluster_kmeans"]
        .value_counts(dropna=False)
        .sort_index()
        .rename("cluster_size")
        .reset_index()
        .rename(columns={"index": "cluster_kmeans"})
    )

    # ------------------------------------------------------------------
    # 6) cluster profile
    #    画像列沿用 final baseline 那套解释列，方便公平解释
    # ------------------------------------------------------------------
    profile_cols = [
        "n_events",
        "duration_ms",
        "trade_ratio",
        "cancel_ratio",
        "buy_like_ratio",
        "sell_like_ratio",
        "aggressive_ratio",
        "at_bid1_ratio",
        "at_ask1_ratio",
        "inside_spread_ratio",
        "mean_dt_ms",
        "max_dt_ms",
        "mean_qty_log1p",
        "mean_spread_abs",
        "mean_top_depth_total",
        "mean_top_depth_imbalance",
        "delta_snap_last",
        "delta_depth_total",
        "delta_depth_imbalance",
        "side_imbalance",
        "top1_pos_imbalance",
    ]
    profile_cols = [c for c in profile_cols if c in base_df.columns]

    learned_profile_df = base_df[["episode_id"] + profile_cols].copy()
    learned_profile_df = learned_profile_df.merge(
        learned_assignment,
        on="episode_id",
        how="inner",
    )

    cluster_profile = (
        learned_profile_df.groupby("cluster_kmeans")[profile_cols]
        .mean()
        .reset_index()
        .sort_values("cluster_kmeans")
    )
    cluster_profile = cluster_profile.merge(cluster_counts, on="cluster_kmeans", how="left")

    # ------------------------------------------------------------------
    # 7) 带 cluster 的 embedding 表
    # ------------------------------------------------------------------
    embedding_with_cluster = emb_df.merge(
        learned_assignment,
        on="episode_id",
        how="left",
    )

    print("\n[metrics]")
    print(metrics)

    print("\n[cluster counts]")
    print(cluster_counts)

    print("\n[cluster profile head]")
    print(cluster_profile.head())

    return {
        "embedding_df": emb_df,
        "embedding_with_cluster": embedding_with_cluster,
        "assignment_df": learned_assignment,
        "cluster_counts": cluster_counts,
        "cluster_profile": cluster_profile,
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "scaler": scaler,
        "pca_model": pca_model,
        "kmeans_model": kmeans_model,
        "labels": labels,
        "metrics": metrics,
        "profile_cols": profile_cols,
        "embedding_cols": emb_cols,
    }


def export_step2_learned_embedding_cluster_audit(
    embedding_csv: str,
    baseline_feature_csv: str,
    out_dir: str,
    run_name: str = "learned_small_transformer_v1_cluster",
    pca_n_components: int = 8,
    kmeans_k: int = 8,
    random_state: int = 42,
    kmeans_n_init: int = 1,
    kmeans_max_iter: int = 30,
    silhouette_sample_size: int = 2000,
) -> dict:
    """
    导出 learned embedding clustering 的完整结果包。
    """
    res = run_learned_embedding_kmeans(
        embedding_csv=embedding_csv,
        baseline_feature_csv=baseline_feature_csv,
        pca_n_components=pca_n_components,
        kmeans_k=kmeans_k,
        random_state=random_state,
        kmeans_n_init=kmeans_n_init,
        kmeans_max_iter=kmeans_max_iter,
        silhouette_sample_size=silhouette_sample_size,
    )

    save_dir = Path(out_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    assignment_file = save_dir / "learned_episode_assignment.csv"
    counts_file = save_dir / "learned_cluster_counts.csv"
    profile_file = save_dir / "learned_cluster_profile.csv"
    embedding_with_cluster_file = save_dir / "learned_embedding_with_cluster.csv"
    metrics_file = save_dir / "learned_cluster_metrics.json"
    audit_report_file = save_dir / "learned_cluster_report.txt"

    res["assignment_df"].to_csv(assignment_file, index=False)
    res["cluster_counts"].to_csv(counts_file, index=False)
    res["cluster_profile"].to_csv(profile_file, index=False)
    res["embedding_with_cluster"].to_csv(embedding_with_cluster_file, index=False)

    metrics_file.write_text(
        json.dumps(res["metrics"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("STEP2-4 LEARNED EMBEDDING CLUSTER AUDIT")
    lines.append("=" * 80)
    lines.append(f"embedding_csv = {embedding_csv}")
    lines.append(f"baseline_feature_csv = {baseline_feature_csv}")
    lines.append(f"run_name = {run_name}")
    lines.append("")
    lines.append("[metrics]")
    for k, v in res["metrics"].items():
        lines.append(f"{k} = {v}")
    lines.append("")
    lines.append("[embedding cols]")
    lines.append(", ".join(res["embedding_cols"]))
    lines.append("")
    lines.append("[profile cols]")
    lines.append(", ".join(res["profile_cols"]))
    lines.append("")
    lines.append("[cluster counts]")
    lines.append(res["cluster_counts"].to_string(index=False))
    lines.append("")
    lines.append("[cluster profile]")
    lines.append(res["cluster_profile"].to_string(index=False))

    audit_report_file.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 120)
    print("STEP2-4 PART2 | LEARNED EMBEDDING CLUSTER EXPORT")
    print("=" * 120)
    print("[saved files]")
    print("assignment_file            =", assignment_file)
    print("counts_file                =", counts_file)
    print("profile_file               =", profile_file)
    print("embedding_with_cluster_file=", embedding_with_cluster_file)
    print("metrics_file               =", metrics_file)
    print("audit_report_file          =", audit_report_file)

    return {
        **res,
        "save_dir": str(save_dir),
        "assignment_file": str(assignment_file),
        "counts_file": str(counts_file),
        "profile_file": str(profile_file),
        "embedding_with_cluster_file": str(embedding_with_cluster_file),
        "metrics_file": str(metrics_file),
        "audit_report_file": str(audit_report_file),
    }


# =============================================================================
# Step2-4 Part 3：learned vs final baseline profile matching
# =============================================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd


def _safe_zscore_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce")
        mu = x.mean()
        sd = x.std(ddof=0)
        if pd.isna(sd) or sd < 1e-12:
            out[c] = 0.0
        else:
            out[c] = (x - mu) / sd
    return out


def _corr_pairwise(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("a and b must be 1d")
    if len(a) != len(b):
        raise ValueError("a and b length mismatch")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.allclose(a.std(), 0.0) or np.allclose(b.std(), 0.0):
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def compare_learned_vs_baseline_cluster_profiles(
    learned_profile_csv: str,
    baseline_profile_csv: str,
    show_top_n: int = 3,
) -> dict:
    """
    用 cluster profile 做 learned vs baseline matching。
    核心输出：
    - cluster-to-cluster correlation matrix
    - 每个 learned cluster 的 best baseline match
    - 每个 baseline cluster 的 best learned match
    """

    print("\n" + "=" * 120)
    print("STEP2-4 PART3 | LEARNED VS BASELINE PROFILE MATCHING")
    print("=" * 120)
    print(f"[learned_profile_csv]  = {learned_profile_csv}")
    print(f"[baseline_profile_csv] = {baseline_profile_csv}")

    learned_df = pd.read_csv(learned_profile_csv)
    baseline_df = pd.read_csv(baseline_profile_csv)

    if "cluster_kmeans" not in learned_df.columns:
        raise ValueError("learned profile missing cluster_kmeans")
    if "cluster_kmeans" not in baseline_df.columns:
        raise ValueError("baseline profile missing cluster_kmeans")

    # 仅保留双方共有的 profile 列
    exclude_cols = {"cluster_kmeans", "cluster_size"}
    learned_cols = [c for c in learned_df.columns if c not in exclude_cols]
    baseline_cols = [c for c in baseline_df.columns if c not in exclude_cols]
    common_profile_cols = [c for c in learned_cols if c in baseline_cols]

    if len(common_profile_cols) == 0:
        raise ValueError("No common profile columns between learned and baseline")

    # 转 numeric
    for c in common_profile_cols:
        learned_df[c] = pd.to_numeric(learned_df[c], errors="coerce")
        baseline_df[c] = pd.to_numeric(baseline_df[c], errors="coerce")

    learned_df = learned_df.sort_values("cluster_kmeans").reset_index(drop=True)
    baseline_df = baseline_df.sort_values("cluster_kmeans").reset_index(drop=True)

    # 在各自 cluster profile 表内做 z-score，避免绝对尺度影响匹配
    learned_z = _safe_zscore_df(learned_df[["cluster_kmeans"] + common_profile_cols], common_profile_cols)
    baseline_z = _safe_zscore_df(baseline_df[["cluster_kmeans"] + common_profile_cols], common_profile_cols)

    learned_ids = learned_z["cluster_kmeans"].tolist()
    baseline_ids = baseline_z["cluster_kmeans"].tolist()

    corr_mat = np.zeros((len(learned_ids), len(baseline_ids)), dtype=float)

    for i, lid in enumerate(learned_ids):
        lv = learned_z.loc[learned_z["cluster_kmeans"] == lid, common_profile_cols].iloc[0].to_numpy(dtype=float)
        for j, bid in enumerate(baseline_ids):
            bv = baseline_z.loc[baseline_z["cluster_kmeans"] == bid, common_profile_cols].iloc[0].to_numpy(dtype=float)
            corr_mat[i, j] = _corr_pairwise(lv, bv)

    corr_df = pd.DataFrame(
        corr_mat,
        index=[f"learned_{x}" for x in learned_ids],
        columns=[f"baseline_{x}" for x in baseline_ids],
    )

    # learned -> baseline best match
    learned_best_rows = []
    for i, lid in enumerate(learned_ids):
        row = corr_mat[i, :]
        order = np.argsort(-row)
        best_j = int(order[0])
        learned_best_rows.append({
            "learned_cluster": int(lid),
            "best_baseline_cluster": int(baseline_ids[best_j]),
            "best_corr": float(row[best_j]),
            "top_matches": "; ".join(
                [f"baseline_{int(baseline_ids[j])}:{row[j]:.4f}" for j in order[:show_top_n]]
            ),
        })
    learned_best_df = pd.DataFrame(learned_best_rows).sort_values("learned_cluster").reset_index(drop=True)

    # baseline -> learned best match
    baseline_best_rows = []
    for j, bid in enumerate(baseline_ids):
        col = corr_mat[:, j]
        order = np.argsort(-col)
        best_i = int(order[0])
        baseline_best_rows.append({
            "baseline_cluster": int(bid),
            "best_learned_cluster": int(learned_ids[best_i]),
            "best_corr": float(col[best_i]),
            "top_matches": "; ".join(
                [f"learned_{int(learned_ids[i])}:{col[i]:.4f}" for i in order[:show_top_n]]
            ),
        })
    baseline_best_df = pd.DataFrame(baseline_best_rows).sort_values("baseline_cluster").reset_index(drop=True)

    # 贪心一对一 matching（按相关系数从高到低选不冲突对）
    pairs = []
    all_pairs = []
    for i, lid in enumerate(learned_ids):
        for j, bid in enumerate(baseline_ids):
            all_pairs.append((int(lid), int(bid), float(corr_mat[i, j])))
    all_pairs = sorted(all_pairs, key=lambda x: x[2], reverse=True)

    used_l = set()
    used_b = set()
    for lid, bid, corr in all_pairs:
        if lid in used_l or bid in used_b:
            continue
        pairs.append({
            "learned_cluster": lid,
            "baseline_cluster": bid,
            "corr": corr,
        })
        used_l.add(lid)
        used_b.add(bid)

    greedy_match_df = pd.DataFrame(pairs).sort_values("learned_cluster").reset_index(drop=True)

    # 摘要统计
    best_corrs = learned_best_df["best_corr"].to_numpy(dtype=float)
    summary = {
        "n_common_profile_cols": int(len(common_profile_cols)),
        "mean_best_corr_learned_to_baseline": float(np.nanmean(best_corrs)),
        "min_best_corr_learned_to_baseline": float(np.nanmin(best_corrs)),
        "max_best_corr_learned_to_baseline": float(np.nanmax(best_corrs)),
        "n_learned_clusters": int(len(learned_ids)),
        "n_baseline_clusters": int(len(baseline_ids)),
    }

    print("\n[summary]")
    print(summary)

    print("\n[learned -> baseline best match]")
    print(learned_best_df)

    print("\n[baseline -> learned best match]")
    print(baseline_best_df)

    print("\n[greedy one-to-one match]")
    print(greedy_match_df)

    return {
        "learned_profile_df": learned_df,
        "baseline_profile_df": baseline_df,
        "common_profile_cols": common_profile_cols,
        "corr_df": corr_df,
        "learned_best_df": learned_best_df,
        "baseline_best_df": baseline_best_df,
        "greedy_match_df": greedy_match_df,
        "summary": summary,
    }


def export_learned_vs_baseline_profile_matching(
    learned_profile_csv: str,
    baseline_profile_csv: str,
    out_dir: str,
    run_name: str = "learned_vs_baseline_profile_matching",
) -> dict:
    res = compare_learned_vs_baseline_cluster_profiles(
        learned_profile_csv=learned_profile_csv,
        baseline_profile_csv=baseline_profile_csv,
        show_top_n=3,
    )

    save_dir = Path(out_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    corr_file = save_dir / "profile_corr_matrix.csv"
    learned_best_file = save_dir / "learned_to_baseline_best_match.csv"
    baseline_best_file = save_dir / "baseline_to_learned_best_match.csv"
    greedy_match_file = save_dir / "greedy_one_to_one_match.csv"
    summary_file = save_dir / "matching_summary.json"
    report_file = save_dir / "profile_matching_report.txt"

    res["corr_df"].to_csv(corr_file, index=True)
    res["learned_best_df"].to_csv(learned_best_file, index=False)
    res["baseline_best_df"].to_csv(baseline_best_file, index=False)
    res["greedy_match_df"].to_csv(greedy_match_file, index=False)

    summary_file.write_text(
        json.dumps(res["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = []
    lines.append("STEP2-4 LEARNED VS BASELINE PROFILE MATCHING")
    lines.append("=" * 80)
    lines.append(f"learned_profile_csv = {learned_profile_csv}")
    lines.append(f"baseline_profile_csv = {baseline_profile_csv}")
    lines.append("")
    lines.append("[summary]")
    for k, v in res["summary"].items():
        lines.append(f"{k} = {v}")
    lines.append("")
    lines.append("[common_profile_cols]")
    lines.append(", ".join(res["common_profile_cols"]))
    lines.append("")
    lines.append("[learned -> baseline best match]")
    lines.append(res["learned_best_df"].to_string(index=False))
    lines.append("")
    lines.append("[baseline -> learned best match]")
    lines.append(res["baseline_best_df"].to_string(index=False))
    lines.append("")
    lines.append("[greedy one-to-one match]")
    lines.append(res["greedy_match_df"].to_string(index=False))
    lines.append("")
    lines.append("[profile corr matrix]")
    lines.append(res["corr_df"].to_string())

    report_file.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 120)
    print("STEP2-4 PART3 | PROFILE MATCH EXPORT")
    print("=" * 120)
    print("[saved files]")
    print("corr_file          =", corr_file)
    print("learned_best_file  =", learned_best_file)
    print("baseline_best_file =", baseline_best_file)
    print("greedy_match_file  =", greedy_match_file)
    print("summary_file       =", summary_file)
    print("report_file        =", report_file)

    return {
        **res,
        "save_dir": str(save_dir),
        "corr_file": str(corr_file),
        "learned_best_file": str(learned_best_file),
        "baseline_best_file": str(baseline_best_file),
        "greedy_match_file": str(greedy_match_file),
        "summary_file": str(summary_file),
        "report_file": str(report_file),
    }


# =============================================================================
# Step2-4 Part 4：diagnose one learned cluster (default: cluster 6)
# =============================================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd


def diagnose_one_learned_cluster(
    learned_assignment_csv: str,
    learned_profile_csv: str,
    baseline_assignment_csv: str,
    baseline_profile_csv: str,
    baseline_feature_csv: str,
    target_cluster: int = 6,
) -> dict:
    """
    对某一个 learned cluster 做重点诊断。
    核心输出：
    1) 该 cluster 的 size / 占比
    2) 相对全样本均值的偏离（z-score 风格）
    3) 与 baseline cluster 的交叉表
    4) target cluster 内部 episode 的长度 / duration / n_events 摘要
    """

    print("\n" + "=" * 120)
    print("STEP2-4 PART4 | DIAGNOSE ONE LEARNED CLUSTER")
    print("=" * 120)
    print(f"[target_cluster] = {target_cluster}")
    print(f"[learned_assignment_csv]  = {learned_assignment_csv}")
    print(f"[baseline_assignment_csv] = {baseline_assignment_csv}")
    print(f"[baseline_feature_csv]    = {baseline_feature_csv}")

    learned_assign = pd.read_csv(learned_assignment_csv)
    learned_profile = pd.read_csv(learned_profile_csv)
    baseline_assign = pd.read_csv(baseline_assignment_csv)
    baseline_profile = pd.read_csv(baseline_profile_csv)
    baseline_feat = pd.read_csv(baseline_feature_csv)

    for df_name, df in [
        ("learned_assign", learned_assign),
        ("baseline_assign", baseline_assign),
        ("baseline_feat", baseline_feat),
    ]:
        if "episode_id" not in df.columns:
            raise ValueError(f"{df_name} missing episode_id")

    if "cluster_kmeans" not in learned_assign.columns:
        raise ValueError("learned_assignment_csv missing cluster_kmeans")
    if "cluster_kmeans" not in baseline_assign.columns:
        raise ValueError("baseline_assignment_csv missing cluster_kmeans")

    # 统一字段类型
    for df in [learned_assign, baseline_assign, baseline_feat]:
        df["episode_id"] = pd.to_numeric(df["episode_id"], errors="coerce")
        df.dropna(subset=["episode_id"], inplace=True)
        df["episode_id"] = df["episode_id"].astype(int)

    learned_assign["cluster_kmeans"] = pd.to_numeric(
        learned_assign["cluster_kmeans"], errors="coerce"
    ).astype(int)
    baseline_assign["cluster_kmeans"] = pd.to_numeric(
        baseline_assign["cluster_kmeans"], errors="coerce"
    ).astype(int)

    # merge 到 episode level
    ep_df = baseline_feat.merge(
        learned_assign[["episode_id", "cluster_kmeans"]].rename(columns={"cluster_kmeans": "learned_cluster"}),
        on="episode_id",
        how="inner",
    ).merge(
        baseline_assign[["episode_id", "cluster_kmeans"]].rename(columns={"cluster_kmeans": "baseline_cluster"}),
        on="episode_id",
        how="inner",
    )

    n_total = len(ep_df)
    target_df = ep_df[ep_df["learned_cluster"] == target_cluster].copy()
    n_target = len(target_df)
    target_ratio = n_target / max(n_total, 1)

    if n_target == 0:
        raise ValueError(f"No episode found for learned cluster {target_cluster}")

    # ------------------------------------------------------------------
    # 1) target cluster 对应的 profile 行
    # ------------------------------------------------------------------
    if "cluster_kmeans" not in learned_profile.columns:
        raise ValueError("learned_profile_csv missing cluster_kmeans")
    if "cluster_kmeans" not in baseline_profile.columns:
        raise ValueError("baseline_profile_csv missing cluster_kmeans")

    learned_profile["cluster_kmeans"] = pd.to_numeric(
        learned_profile["cluster_kmeans"], errors="coerce"
    ).astype(int)
    baseline_profile["cluster_kmeans"] = pd.to_numeric(
        baseline_profile["cluster_kmeans"], errors="coerce"
    ).astype(int)

    if target_cluster not in set(learned_profile["cluster_kmeans"]):
        raise ValueError(f"target cluster {target_cluster} not found in learned profile")

    exclude_cols = {"cluster_kmeans", "cluster_size"}
    learned_profile_cols = [c for c in learned_profile.columns if c not in exclude_cols]

    target_profile_row = learned_profile.loc[
        learned_profile["cluster_kmeans"] == target_cluster, learned_profile_cols
    ].iloc[0]

    all_profile_matrix = learned_profile[learned_profile_cols].copy()
    for c in learned_profile_cols:
        all_profile_matrix[c] = pd.to_numeric(all_profile_matrix[c], errors="coerce")

    profile_mean = all_profile_matrix.mean(axis=0, numeric_only=True)
    profile_std = all_profile_matrix.std(axis=0, ddof=0, numeric_only=True).replace(0, np.nan)

    target_z = ((pd.to_numeric(target_profile_row, errors="coerce") - profile_mean) / profile_std).replace([np.inf, -np.inf], np.nan)

    target_z_df = pd.DataFrame({
        "feature": target_z.index,
        "target_value": pd.to_numeric(target_profile_row, errors="coerce").values,
        "cluster_mean_across_clusters": profile_mean.values,
        "cluster_std_across_clusters": profile_std.values,
        "zscore_vs_learned_cluster_set": target_z.values,
    }).sort_values("zscore_vs_learned_cluster_set", ascending=False)

    # 绝对值最大的偏离
    target_abs_z_df = target_z_df.copy()
    target_abs_z_df["abs_z"] = target_abs_z_df["zscore_vs_learned_cluster_set"].abs()
    target_abs_z_df = target_abs_z_df.sort_values("abs_z", ascending=False)

    # ------------------------------------------------------------------
    # 2) target cluster 内部 episode-level 摘要
    # ------------------------------------------------------------------
    episode_diag_cols = [
        "n_events",
        "duration_ms",
        "trade_ratio",
        "cancel_ratio",
        "buy_like_ratio",
        "sell_like_ratio",
        "aggressive_ratio",
        "at_bid1_ratio",
        "at_ask1_ratio",
        "inside_spread_ratio",
        "mean_dt_ms",
        "max_dt_ms",
        "mean_qty_log1p",
        "mean_spread_abs",
        "mean_top_depth_total",
        "mean_top_depth_imbalance",
        "delta_snap_last",
        "delta_depth_total",
        "delta_depth_imbalance",
        "side_imbalance",
        "top1_pos_imbalance",
    ]
    episode_diag_cols = [c for c in episode_diag_cols if c in target_df.columns]

    target_episode_summary = target_df[episode_diag_cols].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    ).T.reset_index().rename(columns={"index": "feature"})

    overall_episode_summary = ep_df[episode_diag_cols].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    ).T.reset_index().rename(columns={"index": "feature"})

    target_vs_overall_mean = (
        target_df[episode_diag_cols].mean(numeric_only=True).rename("target_mean").reset_index()
        .rename(columns={"index": "feature"})
        .merge(
            ep_df[episode_diag_cols].mean(numeric_only=True).rename("overall_mean").reset_index()
            .rename(columns={"index": "feature"}),
            on="feature",
            how="left",
        )
    )
    target_vs_overall_mean["diff"] = target_vs_overall_mean["target_mean"] - target_vs_overall_mean["overall_mean"]
    target_vs_overall_mean["ratio"] = target_vs_overall_mean["target_mean"] / target_vs_overall_mean["overall_mean"].replace(0, np.nan)
    target_vs_overall_mean["abs_diff"] = target_vs_overall_mean["diff"].abs()
    target_vs_overall_mean = target_vs_overall_mean.sort_values("abs_diff", ascending=False)

    # ------------------------------------------------------------------
    # 3) 与 baseline cluster 的交叉表
    # ------------------------------------------------------------------
    cross_counts = (
        pd.crosstab(target_df["baseline_cluster"], target_df["learned_cluster"])
        .reset_index()
        .rename_axis(None, axis=1)
    )

    baseline_mix = (
        target_df["baseline_cluster"]
        .value_counts(normalize=False)
        .sort_index()
        .rename("count")
        .reset_index()
        .rename(columns={"index": "baseline_cluster"})
    )
    baseline_mix["ratio_in_target_cluster"] = baseline_mix["count"] / baseline_mix["count"].sum()

    # target 在各 baseline cluster 中的渗透率
    baseline_total = (
        ep_df["baseline_cluster"].value_counts()
        .sort_index()
        .rename("baseline_total")
        .reset_index()
        .rename(columns={"index": "baseline_cluster"})
    )
    baseline_mix = baseline_mix.merge(baseline_total, on="baseline_cluster", how="left")
    baseline_mix["penetration_within_baseline_cluster"] = baseline_mix["count"] / baseline_mix["baseline_total"]

    # ------------------------------------------------------------------
    # 4) 选一些极端样本，后续人工检查
    # ------------------------------------------------------------------
    sample_cols = ["episode_id"] + [c for c in [
        "learned_cluster", "baseline_cluster",
        "n_events", "duration_ms",
        "trade_ratio", "cancel_ratio",
        "aggressive_ratio",
        "at_bid1_ratio", "at_ask1_ratio",
        "mean_dt_ms", "max_dt_ms",
        "mean_top_depth_imbalance",
        "side_imbalance", "top1_pos_imbalance"
    ] if c in target_df.columns or c in ["episode_id", "learned_cluster", "baseline_cluster"]]

    target_sorted_short = target_df.sort_values(["duration_ms", "n_events"], ascending=[True, True])[sample_cols].head(15)
    target_sorted_long = target_df.sort_values(["duration_ms", "n_events"], ascending=[False, False])[sample_cols].head(15)
    target_sorted_aggr = target_df.sort_values(
        ["aggressive_ratio", "trade_ratio"], ascending=[False, False]
    )[sample_cols].head(15)

    # ------------------------------------------------------------------
    # 5) summary
    # ------------------------------------------------------------------
    summary = {
        "target_cluster": int(target_cluster),
        "n_total_episodes": int(n_total),
        "n_target_episodes": int(n_target),
        "target_ratio": float(target_ratio),
        "top_baseline_cluster_in_target": int(baseline_mix.sort_values("count", ascending=False).iloc[0]["baseline_cluster"]),
        "top_baseline_cluster_count_in_target": int(baseline_mix.sort_values("count", ascending=False).iloc[0]["count"]),
        "top_baseline_cluster_ratio_in_target": float(baseline_mix.sort_values("count", ascending=False).iloc[0]["ratio_in_target_cluster"]),
    }

    print("\n[summary]")
    print(summary)

    print("\n[top abs z-score features]")
    print(target_abs_z_df.head(12))

    print("\n[baseline mix inside target learned cluster]")
    print(baseline_mix)

    print("\n[target vs overall mean | top abs diff]")
    print(target_vs_overall_mean.head(12))

    return {
        "ep_df": ep_df,
        "target_df": target_df,
        "summary": summary,
        "target_z_df": target_z_df,
        "target_abs_z_df": target_abs_z_df,
        "target_episode_summary": target_episode_summary,
        "overall_episode_summary": overall_episode_summary,
        "target_vs_overall_mean": target_vs_overall_mean,
        "cross_counts": cross_counts,
        "baseline_mix": baseline_mix,
        "target_sorted_short": target_sorted_short,
        "target_sorted_long": target_sorted_long,
        "target_sorted_aggr": target_sorted_aggr,
    }


def export_one_learned_cluster_diagnosis(
    learned_assignment_csv: str,
    learned_profile_csv: str,
    baseline_assignment_csv: str,
    baseline_profile_csv: str,
    baseline_feature_csv: str,
    out_dir: str,
    target_cluster: int = 6,
    run_name: str = "learned_cluster_6_diagnosis",
) -> dict:
    res = diagnose_one_learned_cluster(
        learned_assignment_csv=learned_assignment_csv,
        learned_profile_csv=learned_profile_csv,
        baseline_assignment_csv=baseline_assignment_csv,
        baseline_profile_csv=baseline_profile_csv,
        baseline_feature_csv=baseline_feature_csv,
        target_cluster=target_cluster,
    )

    save_dir = Path(out_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_file = save_dir / "diagnosis_summary.json"
    target_abs_z_file = save_dir / "target_abs_z_features.csv"
    baseline_mix_file = save_dir / "baseline_mix_within_target.csv"
    target_vs_overall_file = save_dir / "target_vs_overall_mean.csv"
    target_episode_summary_file = save_dir / "target_episode_summary.csv"
    target_short_file = save_dir / "target_short_examples.csv"
    target_long_file = save_dir / "target_long_examples.csv"
    target_aggr_file = save_dir / "target_aggressive_examples.csv"
    report_file = save_dir / "diagnosis_report.txt"

    summary_file.write_text(
        json.dumps(res["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    res["target_abs_z_df"].to_csv(target_abs_z_file, index=False)
    res["baseline_mix"].to_csv(baseline_mix_file, index=False)
    res["target_vs_overall_mean"].to_csv(target_vs_overall_file, index=False)
    res["target_episode_summary"].to_csv(target_episode_summary_file, index=False)
    res["target_sorted_short"].to_csv(target_short_file, index=False)
    res["target_sorted_long"].to_csv(target_long_file, index=False)
    res["target_sorted_aggr"].to_csv(target_aggr_file, index=False)

    lines = []
    lines.append("STEP2-4 ONE LEARNED CLUSTER DIAGNOSIS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("[summary]")
    for k, v in res["summary"].items():
        lines.append(f"{k} = {v}")
    lines.append("")
    lines.append("[top abs z-score features]")
    lines.append(res["target_abs_z_df"].head(15).to_string(index=False))
    lines.append("")
    lines.append("[baseline mix inside target learned cluster]")
    lines.append(res["baseline_mix"].to_string(index=False))
    lines.append("")
    lines.append("[target vs overall mean | top abs diff]")
    lines.append(res["target_vs_overall_mean"].head(15).to_string(index=False))
    lines.append("")
    lines.append("[short examples]")
    lines.append(res["target_sorted_short"].head(10).to_string(index=False))
    lines.append("")
    lines.append("[long examples]")
    lines.append(res["target_sorted_long"].head(10).to_string(index=False))
    lines.append("")
    lines.append("[aggressive examples]")
    lines.append(res["target_sorted_aggr"].head(10).to_string(index=False))

    report_file.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 120)
    print("STEP2-4 PART4 | ONE CLUSTER DIAGNOSIS EXPORT")
    print("=" * 120)
    print("[saved files]")
    print("summary_file               =", summary_file)
    print("target_abs_z_file          =", target_abs_z_file)
    print("baseline_mix_file          =", baseline_mix_file)
    print("target_vs_overall_file     =", target_vs_overall_file)
    print("target_episode_summary_file=", target_episode_summary_file)
    print("target_short_file          =", target_short_file)
    print("target_long_file           =", target_long_file)
    print("target_aggr_file           =", target_aggr_file)
    print("report_file                =", report_file)

    return {
        **res,
        "save_dir": str(save_dir),
        "summary_file": str(summary_file),
        "target_abs_z_file": str(target_abs_z_file),
        "baseline_mix_file": str(baseline_mix_file),
        "target_vs_overall_file": str(target_vs_overall_file),
        "target_episode_summary_file": str(target_episode_summary_file),
        "target_short_file": str(target_short_file),
        "target_long_file": str(target_long_file),
        "target_aggr_file": str(target_aggr_file),
        "report_file": str(report_file),
    }


# =============================================================================
# Step2-4 utils：global seed
# =============================================================================

import random
import numpy as np
import torch


def set_step2_global_seed(seed: int = 42) -> None:
    """
    为 Step2 learned encoder 实验设置全局随机种子。
    目标：
    - Python random
    - numpy
    - torch cpu / cuda
    """
    print("\n" + "=" * 120)
    print("STEP2-4 | SET GLOBAL SEED")
    print("=" * 120)
    print(f"[seed] = {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 这里先不强行开 deterministic algorithms，
    # 避免某些 CUDA kernel 变慢或报不支持。
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Step2-4 Part 5：seed-to-seed learned cluster stability
# =============================================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd


def compare_two_learned_cluster_runs(
    run_a_assignment_csv: str,
    run_a_profile_csv: str,
    run_b_assignment_csv: str,
    run_b_profile_csv: str,
    show_top_n: int = 3,
) -> dict:
    """
    比较两个 learned clustering run（例如 seed42 vs seed7）的稳定性。

    输出：
    1) profile correlation matrix
    2) A->B / B->A best match
    3) greedy one-to-one profile match
    4) episode-level overlap table（在 greedy match 下）
    """

    print("\n" + "=" * 120)
    print("STEP2-4 PART5 | COMPARE TWO LEARNED CLUSTER RUNS")
    print("=" * 120)
    print(f"[run_a_assignment_csv] = {run_a_assignment_csv}")
    print(f"[run_a_profile_csv]    = {run_a_profile_csv}")
    print(f"[run_b_assignment_csv] = {run_b_assignment_csv}")
    print(f"[run_b_profile_csv]    = {run_b_profile_csv}")

    a_assign = pd.read_csv(run_a_assignment_csv)
    a_prof = pd.read_csv(run_a_profile_csv)
    b_assign = pd.read_csv(run_b_assignment_csv)
    b_prof = pd.read_csv(run_b_profile_csv)

    for df_name, df in [
        ("a_assign", a_assign), ("b_assign", b_assign),
        ("a_prof", a_prof), ("b_prof", b_prof),
    ]:
        if "cluster_kmeans" not in df.columns:
            raise ValueError(f"{df_name} missing cluster_kmeans")

    for df_name, df in [("a_assign", a_assign), ("b_assign", b_assign)]:
        if "episode_id" not in df.columns:
            raise ValueError(f"{df_name} missing episode_id")

    a_assign["episode_id"] = pd.to_numeric(a_assign["episode_id"], errors="coerce")
    b_assign["episode_id"] = pd.to_numeric(b_assign["episode_id"], errors="coerce")
    a_assign["cluster_kmeans"] = pd.to_numeric(a_assign["cluster_kmeans"], errors="coerce")
    b_assign["cluster_kmeans"] = pd.to_numeric(b_assign["cluster_kmeans"], errors="coerce")

    a_assign = a_assign.dropna(subset=["episode_id", "cluster_kmeans"]).copy()
    b_assign = b_assign.dropna(subset=["episode_id", "cluster_kmeans"]).copy()
    a_assign["episode_id"] = a_assign["episode_id"].astype(int)
    b_assign["episode_id"] = b_assign["episode_id"].astype(int)
    a_assign["cluster_kmeans"] = a_assign["cluster_kmeans"].astype(int)
    b_assign["cluster_kmeans"] = b_assign["cluster_kmeans"].astype(int)

    a_prof["cluster_kmeans"] = pd.to_numeric(a_prof["cluster_kmeans"], errors="coerce").astype(int)
    b_prof["cluster_kmeans"] = pd.to_numeric(b_prof["cluster_kmeans"], errors="coerce").astype(int)

    exclude_cols = {"cluster_kmeans", "cluster_size"}
    a_cols = [c for c in a_prof.columns if c not in exclude_cols]
    b_cols = [c for c in b_prof.columns if c not in exclude_cols]
    common_profile_cols = [c for c in a_cols if c in b_cols]
    if len(common_profile_cols) == 0:
        raise ValueError("No common profile columns between run A and run B")

    for c in common_profile_cols:
        a_prof[c] = pd.to_numeric(a_prof[c], errors="coerce")
        b_prof[c] = pd.to_numeric(b_prof[c], errors="coerce")

    # within-run zscore on cluster profiles
    def _zprof(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            x = pd.to_numeric(out[c], errors="coerce")
            mu = x.mean()
            sd = x.std(ddof=0)
            if pd.isna(sd) or sd < 1e-12:
                out[c] = 0.0
            else:
                out[c] = (x - mu) / sd
        return out

    def _corr_1d(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) != len(y):
            raise ValueError("length mismatch")
        if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    a_prof = a_prof.sort_values("cluster_kmeans").reset_index(drop=True)
    b_prof = b_prof.sort_values("cluster_kmeans").reset_index(drop=True)

    a_z = _zprof(a_prof[["cluster_kmeans"] + common_profile_cols], common_profile_cols)
    b_z = _zprof(b_prof[["cluster_kmeans"] + common_profile_cols], common_profile_cols)

    a_ids = a_z["cluster_kmeans"].tolist()
    b_ids = b_z["cluster_kmeans"].tolist()

    corr_mat = np.zeros((len(a_ids), len(b_ids)), dtype=float)
    for i, aid in enumerate(a_ids):
        av = a_z.loc[a_z["cluster_kmeans"] == aid, common_profile_cols].iloc[0].to_numpy(dtype=float)
        for j, bid in enumerate(b_ids):
            bv = b_z.loc[b_z["cluster_kmeans"] == bid, common_profile_cols].iloc[0].to_numpy(dtype=float)
            corr_mat[i, j] = _corr_1d(av, bv)

    corr_df = pd.DataFrame(
        corr_mat,
        index=[f"runA_{x}" for x in a_ids],
        columns=[f"runB_{x}" for x in b_ids],
    )

    # A -> B best match
    a_best_rows = []
    for i, aid in enumerate(a_ids):
        row = corr_mat[i, :]
        order = np.argsort(-row)
        best_j = int(order[0])
        a_best_rows.append({
            "runA_cluster": int(aid),
            "best_runB_cluster": int(b_ids[best_j]),
            "best_corr": float(row[best_j]),
            "top_matches": "; ".join(
                [f"runB_{int(b_ids[j])}:{row[j]:.4f}" for j in order[:show_top_n]]
            ),
        })
    a_best_df = pd.DataFrame(a_best_rows).sort_values("runA_cluster").reset_index(drop=True)

    # B -> A best match
    b_best_rows = []
    for j, bid in enumerate(b_ids):
        col = corr_mat[:, j]
        order = np.argsort(-col)
        best_i = int(order[0])
        b_best_rows.append({
            "runB_cluster": int(bid),
            "best_runA_cluster": int(a_ids[best_i]),
            "best_corr": float(col[best_i]),
            "top_matches": "; ".join(
                [f"runA_{int(a_ids[i])}:{col[i]:.4f}" for i in order[:show_top_n]]
            ),
        })
    b_best_df = pd.DataFrame(b_best_rows).sort_values("runB_cluster").reset_index(drop=True)

    # greedy one-to-one
    all_pairs = []
    for i, aid in enumerate(a_ids):
        for j, bid in enumerate(b_ids):
            all_pairs.append((int(aid), int(bid), float(corr_mat[i, j])))
    all_pairs = sorted(all_pairs, key=lambda x: x[2], reverse=True)

    used_a, used_b = set(), set()
    greedy_rows = []
    for aid, bid, corr in all_pairs:
        if aid in used_a or bid in used_b:
            continue
        greedy_rows.append({
            "runA_cluster": aid,
            "runB_cluster": bid,
            "corr": corr,
        })
        used_a.add(aid)
        used_b.add(bid)

    greedy_df = pd.DataFrame(greedy_rows).sort_values("runA_cluster").reset_index(drop=True)

    # episode-level overlap under greedy mapping
    ep_common = a_assign.merge(
        b_assign,
        on="episode_id",
        how="inner",
        suffixes=("_runA", "_runB"),
    )

    overlap_rows = []
    for _, row in greedy_df.iterrows():
        aid = int(row["runA_cluster"])
        bid = int(row["runB_cluster"])
        n_a = int((ep_common["cluster_kmeans_runA"] == aid).sum())
        n_b = int((ep_common["cluster_kmeans_runB"] == bid).sum())
        n_inter = int(((ep_common["cluster_kmeans_runA"] == aid) & (ep_common["cluster_kmeans_runB"] == bid)).sum())
        precision = n_inter / n_a if n_a > 0 else np.nan
        recall = n_inter / n_b if n_b > 0 else np.nan
        jaccard = n_inter / (n_a + n_b - n_inter) if (n_a + n_b - n_inter) > 0 else np.nan
        overlap_rows.append({
            "runA_cluster": aid,
            "runB_cluster": bid,
            "profile_corr": float(row["corr"]),
            "runA_size": n_a,
            "runB_size": n_b,
            "intersection": n_inter,
            "precision_runA_to_runB": precision,
            "recall_runB_from_runA": recall,
            "jaccard": jaccard,
        })
    overlap_df = pd.DataFrame(overlap_rows).sort_values("runA_cluster").reset_index(drop=True)

    summary = {
        "n_common_profile_cols": int(len(common_profile_cols)),
        "mean_best_corr_runA_to_runB": float(np.nanmean(a_best_df["best_corr"])),
        "min_best_corr_runA_to_runB": float(np.nanmin(a_best_df["best_corr"])),
        "max_best_corr_runA_to_runB": float(np.nanmax(a_best_df["best_corr"])),
        "mean_greedy_profile_corr": float(np.nanmean(greedy_df["corr"])),
        "mean_greedy_jaccard": float(np.nanmean(overlap_df["jaccard"])),
        "min_greedy_jaccard": float(np.nanmin(overlap_df["jaccard"])),
        "max_greedy_jaccard": float(np.nanmax(overlap_df["jaccard"])),
        "n_common_episodes": int(len(ep_common)),
    }

    print("\n[summary]")
    print(summary)

    print("\n[runA -> runB best match]")
    print(a_best_df)

    print("\n[runB -> runA best match]")
    print(b_best_df)

    print("\n[greedy one-to-one profile match]")
    print(greedy_df)

    print("\n[episode overlap under greedy match]")
    print(overlap_df)

    return {
        "corr_df": corr_df,
        "runA_best_df": a_best_df,
        "runB_best_df": b_best_df,
        "greedy_df": greedy_df,
        "overlap_df": overlap_df,
        "summary": summary,
        "common_profile_cols": common_profile_cols,
    }


def export_two_learned_cluster_run_comparison(
    run_a_assignment_csv: str,
    run_a_profile_csv: str,
    run_b_assignment_csv: str,
    run_b_profile_csv: str,
    out_dir: str,
    run_name: str = "learned_seed42_vs_seed7_comparison",
) -> dict:
    res = compare_two_learned_cluster_runs(
        run_a_assignment_csv=run_a_assignment_csv,
        run_a_profile_csv=run_a_profile_csv,
        run_b_assignment_csv=run_b_assignment_csv,
        run_b_profile_csv=run_b_profile_csv,
        show_top_n=3,
    )

    save_dir = Path(out_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    corr_file = save_dir / "seed_profile_corr_matrix.csv"
    runA_best_file = save_dir / "runA_to_runB_best_match.csv"
    runB_best_file = save_dir / "runB_to_runA_best_match.csv"
    greedy_file = save_dir / "greedy_profile_match.csv"
    overlap_file = save_dir / "greedy_match_episode_overlap.csv"
    summary_file = save_dir / "seed_comparison_summary.json"
    report_file = save_dir / "seed_comparison_report.txt"

    res["corr_df"].to_csv(corr_file)
    res["runA_best_df"].to_csv(runA_best_file, index=False)
    res["runB_best_df"].to_csv(runB_best_file, index=False)
    res["greedy_df"].to_csv(greedy_file, index=False)
    res["overlap_df"].to_csv(overlap_file, index=False)
    summary_file.write_text(json.dumps(res["summary"], ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("STEP2-4 SEED-TO-SEED LEARNED CLUSTER COMPARISON")
    lines.append("=" * 80)
    lines.append("")
    lines.append("[summary]")
    for k, v in res["summary"].items():
        lines.append(f"{k} = {v}")
    lines.append("")
    lines.append("[common_profile_cols]")
    lines.append(", ".join(res["common_profile_cols"]))
    lines.append("")
    lines.append("[runA -> runB best match]")
    lines.append(res["runA_best_df"].to_string(index=False))
    lines.append("")
    lines.append("[runB -> runA best match]")
    lines.append(res["runB_best_df"].to_string(index=False))
    lines.append("")
    lines.append("[greedy one-to-one profile match]")
    lines.append(res["greedy_df"].to_string(index=False))
    lines.append("")
    lines.append("[episode overlap under greedy match]")
    lines.append(res["overlap_df"].to_string(index=False))
    report_file.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 120)
    print("STEP2-4 PART5 | SEED COMPARISON EXPORT")
    print("=" * 120)
    print("[saved files]")
    print("corr_file     =", corr_file)
    print("runA_best_file=", runA_best_file)
    print("runB_best_file=", runB_best_file)
    print("greedy_file   =", greedy_file)
    print("overlap_file  =", overlap_file)
    print("summary_file  =", summary_file)
    print("report_file   =", report_file)

    return {
        **res,
        "save_dir": str(save_dir),
        "corr_file": str(corr_file),
        "runA_best_file": str(runA_best_file),
        "runB_best_file": str(runB_best_file),
        "greedy_file": str(greedy_file),
        "overlap_file": str(overlap_file),
        "summary_file": str(summary_file),
        "report_file": str(report_file),
    }


# =============================================================================
# Step2-4 Part 6：stable core vs residual region
# =============================================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd


def build_stable_core_vs_residual_view(
    seed_comparison_greedy_csv: str,
    seed_comparison_overlap_csv: str,
    run_a_assignment_csv: str,
    run_b_assignment_csv: str,
    run_a_profile_csv: str,
    run_b_profile_csv: str,
    out_dir: str,
    run_name: str = "stable_core_vs_residual_view",
    profile_corr_threshold: float = 0.85,
    jaccard_threshold: float = 0.60,
) -> dict:
    """
    基于 seed42 vs seed7 的比较结果，把 learned clusters 分成：
    1) stable core clusters
    2) unstable residual region

    判定规则（可调）：
    - greedy matched pair 的 profile corr >= profile_corr_threshold
    - greedy matched pair 的 jaccard >= jaccard_threshold

    满足上述两个条件的 pair，视为 stable core。
    其他 cluster 统一视为 unstable / residual。
    """

    print("\n" + "=" * 120)
    print("STEP2-4 PART6 | STABLE CORE VS RESIDUAL VIEW")
    print("=" * 120)
    print(f"[seed_comparison_greedy_csv]  = {seed_comparison_greedy_csv}")
    print(f"[seed_comparison_overlap_csv] = {seed_comparison_overlap_csv}")
    print(f"[profile_corr_threshold]      = {profile_corr_threshold}")
    print(f"[jaccard_threshold]           = {jaccard_threshold}")

    greedy_df = pd.read_csv(seed_comparison_greedy_csv)
    overlap_df = pd.read_csv(seed_comparison_overlap_csv)

    run_a_assign = pd.read_csv(run_a_assignment_csv)
    run_b_assign = pd.read_csv(run_b_assignment_csv)
    run_a_prof = pd.read_csv(run_a_profile_csv)
    run_b_prof = pd.read_csv(run_b_profile_csv)

    # merge greedy + overlap
    pair_df = greedy_df.merge(
        overlap_df,
        on=["runA_cluster", "runB_cluster"],
        how="inner",
    )

    # 标记 stable pairs
    pair_df["is_stable_core_pair"] = (
        (pd.to_numeric(pair_df["corr"], errors="coerce") >= profile_corr_threshold) &
        (pd.to_numeric(pair_df["jaccard"], errors="coerce") >= jaccard_threshold)
    )

    stable_pairs = pair_df[pair_df["is_stable_core_pair"]].copy()
    stable_runA_clusters = set(stable_pairs["runA_cluster"].astype(int).tolist())
    stable_runB_clusters = set(stable_pairs["runB_cluster"].astype(int).tolist())

    # assignment 规范化
    for df in [run_a_assign, run_b_assign]:
        df["episode_id"] = pd.to_numeric(df["episode_id"], errors="coerce")
        df["cluster_kmeans"] = pd.to_numeric(df["cluster_kmeans"], errors="coerce")
        df.dropna(subset=["episode_id", "cluster_kmeans"], inplace=True)
        df["episode_id"] = df["episode_id"].astype(int)
        df["cluster_kmeans"] = df["cluster_kmeans"].astype(int)

    run_a_assign = run_a_assign.rename(columns={"cluster_kmeans": "runA_cluster"})
    run_b_assign = run_b_assign.rename(columns={"cluster_kmeans": "runB_cluster"})

    ep_df = run_a_assign.merge(run_b_assign, on="episode_id", how="inner")

    ep_df["runA_is_stable_core"] = ep_df["runA_cluster"].isin(stable_runA_clusters)
    ep_df["runB_is_stable_core"] = ep_df["runB_cluster"].isin(stable_runB_clusters)

    # 双边都落在 stable core pair 中，才算 truly stable core episode
    stable_pair_lookup = set(
        zip(
            stable_pairs["runA_cluster"].astype(int).tolist(),
            stable_pairs["runB_cluster"].astype(int).tolist(),
        )
    )
    ep_df["pair_tuple"] = list(zip(ep_df["runA_cluster"], ep_df["runB_cluster"]))
    ep_df["is_stable_core_episode"] = ep_df["pair_tuple"].isin(stable_pair_lookup)

    ep_df["region_label"] = np.where(
        ep_df["is_stable_core_episode"],
        "stable_core",
        "residual",
    )

    # ------------------------------------------------------------
    # stable core pair summary
    # ------------------------------------------------------------
    stable_pair_summary = stable_pairs[[
        "runA_cluster",
        "runB_cluster",
        "corr",
        "jaccard",
        "intersection",
        "runA_size",
        "runB_size",
        "precision_runA_to_runB",
        "recall_runB_from_runA",
    ]].copy().sort_values(["corr", "jaccard"], ascending=[False, False])

    # ------------------------------------------------------------
    # episode-level summary
    # ------------------------------------------------------------
    region_summary = (
        ep_df["region_label"]
        .value_counts(dropna=False)
        .rename("n_episodes")
        .reset_index()
        .rename(columns={"index": "region_label"})
    )
    region_summary["ratio"] = region_summary["n_episodes"] / region_summary["n_episodes"].sum()

    # stable core pair coverage
    stable_core_coverage = {
        "n_common_episodes": int(len(ep_df)),
        "n_stable_core_episodes": int(ep_df["is_stable_core_episode"].sum()),
        "n_residual_episodes": int((~ep_df["is_stable_core_episode"]).sum()),
        "stable_core_ratio": float(ep_df["is_stable_core_episode"].mean()),
        "n_stable_pairs": int(len(stable_pair_summary)),
    }

    # ------------------------------------------------------------
    # 给 runA / runB cluster 打标签
    # ------------------------------------------------------------
    run_a_cluster_labels = pd.DataFrame({
        "runA_cluster": sorted(run_a_assign["runA_cluster"].unique())
    })
    run_a_cluster_labels["cluster_region"] = np.where(
        run_a_cluster_labels["runA_cluster"].isin(stable_runA_clusters),
        "stable_core",
        "residual",
    )

    run_b_cluster_labels = pd.DataFrame({
        "runB_cluster": sorted(run_b_assign["runB_cluster"].unique())
    })
    run_b_cluster_labels["cluster_region"] = np.where(
        run_b_cluster_labels["runB_cluster"].isin(stable_runB_clusters),
        "stable_core",
        "residual",
    )

    # ------------------------------------------------------------
    # profile 摘要（看 stable core 的 cluster 长什么样）
    # ------------------------------------------------------------
    for df in [run_a_prof, run_b_prof]:
        df["cluster_kmeans"] = pd.to_numeric(df["cluster_kmeans"], errors="coerce")
        df.dropna(subset=["cluster_kmeans"], inplace=True)
        df["cluster_kmeans"] = df["cluster_kmeans"].astype(int)

    run_a_prof_tagged = run_a_prof.merge(
        run_a_cluster_labels.rename(columns={"runA_cluster": "cluster_kmeans"}),
        on="cluster_kmeans",
        how="left",
    )
    run_b_prof_tagged = run_b_prof.merge(
        run_b_cluster_labels.rename(columns={"runB_cluster": "cluster_kmeans"}),
        on="cluster_kmeans",
        how="left",
    )

    # ------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------
    save_dir = Path(out_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    stable_pair_file = save_dir / "stable_core_pairs.csv"
    region_summary_file = save_dir / "region_summary.csv"
    episode_region_file = save_dir / "episode_region_label.csv"
    run_a_cluster_label_file = save_dir / "runA_cluster_region_label.csv"
    run_b_cluster_label_file = save_dir / "runB_cluster_region_label.csv"
    run_a_profile_tagged_file = save_dir / "runA_profile_with_region.csv"
    run_b_profile_tagged_file = save_dir / "runB_profile_with_region.csv"
    summary_file = save_dir / "stable_core_summary.json"
    report_file = save_dir / "stable_core_report.txt"

    stable_pair_summary.to_csv(stable_pair_file, index=False)
    region_summary.to_csv(region_summary_file, index=False)
    ep_df[["episode_id", "runA_cluster", "runB_cluster", "region_label", "is_stable_core_episode"]].to_csv(
        episode_region_file, index=False
    )
    run_a_cluster_labels.to_csv(run_a_cluster_label_file, index=False)
    run_b_cluster_labels.to_csv(run_b_cluster_label_file, index=False)
    run_a_prof_tagged.to_csv(run_a_profile_tagged_file, index=False)
    run_b_prof_tagged.to_csv(run_b_profile_tagged_file, index=False)

    summary_obj = {
        **stable_core_coverage,
        "profile_corr_threshold": profile_corr_threshold,
        "jaccard_threshold": jaccard_threshold,
        "stable_runA_clusters": sorted(list(stable_runA_clusters)),
        "stable_runB_clusters": sorted(list(stable_runB_clusters)),
    }
    summary_file.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("STEP2-4 STABLE CORE VS RESIDUAL VIEW")
    lines.append("=" * 80)
    lines.append("")
    lines.append("[summary]")
    for k, v in summary_obj.items():
        lines.append(f"{k} = {v}")
    lines.append("")
    lines.append("[stable core pairs]")
    lines.append(stable_pair_summary.to_string(index=False))
    lines.append("")
    lines.append("[region summary]")
    lines.append(region_summary.to_string(index=False))

    report_file.write_text("\n".join(lines), encoding="utf-8")

    print("\n[summary]")
    print(summary_obj)

    print("\n[stable core pairs]")
    print(stable_pair_summary)

    print("\n[region summary]")
    print(region_summary)

    print("\n" + "=" * 120)
    print("STEP2-4 PART6 | STABLE CORE EXPORT")
    print("=" * 120)
    print("[saved files]")
    print("stable_pair_file         =", stable_pair_file)
    print("region_summary_file      =", region_summary_file)
    print("episode_region_file      =", episode_region_file)
    print("run_a_cluster_label_file =", run_a_cluster_label_file)
    print("run_b_cluster_label_file =", run_b_cluster_label_file)
    print("summary_file             =", summary_file)
    print("report_file              =", report_file)

    return {
        "stable_pair_summary": stable_pair_summary,
        "region_summary": region_summary,
        "episode_region_df": ep_df,
        "run_a_cluster_labels": run_a_cluster_labels,
        "run_b_cluster_labels": run_b_cluster_labels,
        "summary": summary_obj,
        "save_dir": str(save_dir),
    }

# =============================================================================
# Step2-4 Time-Aware A：minimal time-aware transformer
# =============================================================================

import torch
import torch.nn as nn


class SmallEpisodeTimeAwareTransformer(nn.Module):
    """
    最小版 time-aware Transformer。
    核心区别：
    - token_emb
    - cont_proj(cont_feats)
    - time_mlp(dt_log1p_seq 单独通道)
    - pos_emb
    四者相加后送入 encoder

    约定：
    cont_feats[:, :, 0] == dt_log1p_seq
    """

    def __init__(
        self,
        vocab_size: int,
        cont_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_token_id: int = 0,
        norm_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.cont_dim = cont_dim

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.cont_proj = nn.Linear(cont_dim, d_model)

        # 关键新增：只对 dt 走一条独立时间投影
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, token_ids, cont_feats, attention_mask):
        """
        token_ids: [B, T]
        cont_feats: [B, T, C]
        attention_mask: [B, T], True = valid
        """
        B, T = token_ids.shape

        x_tok = self.token_emb(token_ids)              # [B, T, D]
        x_cont = self.cont_proj(cont_feats)            # [B, T, D]

        # 约定 cont_feats 第 0 维就是 dt_log1p_seq
        dt_feat = cont_feats[:, :, 0:1]                # [B, T, 1]
        x_time = self.time_mlp(dt_feat)                # [B, T, D]

        x = x_tok + x_cont + x_time + self.pos_emb[:, :T, :]
        x = self.dropout(x)

        key_padding_mask = ~attention_mask
        hidden = self.encoder(x, src_key_padding_mask=key_padding_mask)
        hidden = self.out_norm(hidden)

        mask_f = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

        return hidden, pooled
    
class SmallEpisodeTimeAwareTransformerMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        cont_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_token_id: int = 0,
        norm_first: bool = True,
    ):
        super().__init__()
        self.backbone = SmallEpisodeTimeAwareTransformer(
            vocab_size=vocab_size,
            cont_dim=cont_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            pad_token_id=pad_token_id,
            norm_first=norm_first,
        )
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, cont_feats, attention_mask):
        hidden, pooled = self.backbone(token_ids, cont_feats, attention_mask)
        logits = self.mlm_head(hidden)
        return logits, hidden, pooled

# =============================================================================
# Step2-4 Time-Aware A：independent MLM masking helper
# =============================================================================

def mask_tokens_for_mlm_timeaware(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int,
    mask_token_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
):
    """
    独立于原有 vanilla helper 的最小 MLM masking。
    规则：
    - 只在 valid token 上 mask（attention_mask=True 且 token != PAD）
    - 被选中的位置：
        80% -> [MASK]
        10% -> random token
        10% -> keep original
    - labels 中未选中位置 = -100
    """
    if token_ids.ndim != 2:
        raise ValueError("token_ids must be [B, T]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be [B, T]")

    device = token_ids.device
    masked_input_ids = token_ids.clone()
    labels = torch.full_like(token_ids, fill_value=-100)

    valid_pos = attention_mask.bool() & token_ids.ne(pad_token_id)

    rand = torch.rand(token_ids.shape, device=device)
    mask_pos = valid_pos & (rand < mask_prob)

    labels[mask_pos] = token_ids[mask_pos]

    if mask_pos.sum().item() == 0:
        return masked_input_ids, labels

    choice = torch.rand(token_ids.shape, device=device)

    # 80% -> [MASK]
    pos_mask_token = mask_pos & (choice < 0.8)
    masked_input_ids[pos_mask_token] = mask_token_id

    # 10% -> random token
    pos_random_token = mask_pos & (choice >= 0.8) & (choice < 0.9)
    n_random = int(pos_random_token.sum().item())
    if n_random > 0:
        random_tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=(n_random,),
            device=device,
        )
        masked_input_ids[pos_random_token] = random_tokens

    # 剩余 10% 保持原 token，不用改 masked_input_ids

    return masked_input_ids, labels
    

def train_small_timeaware_transformer_mlm(
    out_dir: str,
    token_min_freq: int = 1,
    max_seq_len: int = 64,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 8,
    max_batches_per_epoch: int | None = None,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    max_len: int = 128,
    mask_prob: float = 0.15,
    device: str = "cpu",
):
    """
    最小版 time-aware MLM 训练入口。
    复用你现有:
    - build_torch_episode_dataset_from_artifacts
    - make_episode_sequence_collate_fn
    - mask_tokens_for_mlm（如果你已有）
    的逻辑。
    """

    print("\n" + "=" * 120)
    print("STEP2-4 TIME-AWARE A | MLM TRAINING")
    print("=" * 120)
    print("[train] device =", device)

    dataset, token_to_id, id_to_token, seq_df = build_torch_episode_dataset_from_artifacts(
        out_dir=out_dir,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
    )

    print("[train] dataset size =", len(dataset))
    print("[train] vocab size   =", len(token_to_id))
    print("[train] batch_size   =", batch_size)

    pad_token_id = token_to_id["[PAD]"]
    collate_fn = make_episode_sequence_collate_fn(pad_token_id=pad_token_id)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 跟 vanilla 一样，给 MLM 补 [MASK]
    token_to_id_train = dict(token_to_id)
    id_to_token_train = dict(id_to_token)

    if "[MASK]" not in token_to_id_train:
        mask_id = len(token_to_id_train)
        token_to_id_train["[MASK]"] = mask_id
        id_to_token_train[mask_id] = "[MASK]"

    vocab_size_train = len(token_to_id_train)
    mask_token_id = token_to_id_train["[MASK]"]

    sample_batch = collate_fn([dataset[0]])
    cont_dim = int(sample_batch["cont_feats"].shape[-1])

    model = SmallEpisodeTimeAwareTransformerMLM(
        vocab_size=vocab_size_train,
        cont_dim=cont_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
        pad_token_id=pad_token_id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_masked = 0

        for batch_idx, batch in enumerate(loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break

            token_ids = batch["token_ids"].to(device)
            cont_feats = batch["cont_feats"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 这里直接复用你现有 vanilla 的 mask_tokens_for_mlm
            masked_input_ids, mlm_labels = mask_tokens_for_mlm_timeaware(
                token_ids=token_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                mask_token_id=mask_token_id,
                vocab_size=vocab_size_train,
                mask_prob=mask_prob,
            )

            logits, hidden, pooled = model(
                token_ids=masked_input_ids,
                cont_feats=cont_feats,
                attention_mask=attention_mask,
            )

            loss = criterion(logits.view(-1, logits.size(-1)), mlm_labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_masked += int((mlm_labels != -100).sum().item())

        n_batches_ran = batch_idx + 1 if "batch_idx" in locals() else 0
        avg_loss = epoch_loss / max(n_batches_ran, 1)

        print(
            f"[epoch {epoch}/{num_epochs}] "
            f"avg_loss={avg_loss:.4f} "
            f"masked_tokens={epoch_masked} "
            f"batches_ran={n_batches_ran}"
        )

    return model, token_to_id_train, id_to_token_train

@torch.no_grad()
def extract_timeaware_episode_embeddings(
    model,
    out_dir: str,
    token_to_id: dict,
    token_min_freq: int = 1,
    max_seq_len: int = 64,
    batch_size: int = 128,
    device: str = "cpu",
):
    print("\n" + "=" * 120)
    print("STEP2-4 TIME-AWARE A | EMBEDDING EXPORT")
    print("=" * 120)

    dataset, _, _, seq_df = build_torch_episode_dataset_from_artifacts(
        out_dir=out_dir,
        token_min_freq=token_min_freq,
        max_seq_len=max_seq_len,
        token_to_id_override=token_to_id,
    )

    pad_token_id = token_to_id["[PAD]"]
    collate_fn = make_episode_sequence_collate_fn(pad_token_id=pad_token_id)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    model.eval()
    rows = []

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        cont_feats = batch["cont_feats"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        episode_id = batch["episode_id"].cpu().numpy()

        # 注意：backbone 在 model.backbone
        hidden, pooled = model.backbone(
            token_ids=token_ids,
            cont_feats=cont_feats,
            attention_mask=attention_mask,
        )

        pooled_np = pooled.detach().cpu().numpy()

        for i in range(len(episode_id)):
            row = {"episode_id": int(episode_id[i])}
            for j in range(pooled_np.shape[1]):
                row[f"emb_{j}"] = float(pooled_np[i, j])
            rows.append(row)

    embedding_df = pd.DataFrame(rows).sort_values("episode_id").reset_index(drop=True)
    return embedding_df

def remap_seq_df_token_ids_to_external_vocab(
    seq_df: pd.DataFrame,
    cached_id_to_token: dict,
    external_token_to_id: dict,
) -> pd.DataFrame:
    seq_df = seq_df.copy()

    unk_id = external_token_to_id["[UNK]"]

    def _remap_one_seq(token_id_seq):
        out = []
        for old_id in token_id_seq:
            tok = cached_id_to_token.get(int(old_id), "[UNK]")
            new_id = external_token_to_id.get(tok, unk_id)
            out.append(int(new_id))
        return out

    seq_df["event_token_id_seq"] = seq_df["event_token_id_seq"].apply(_remap_one_seq)
    return seq_df


