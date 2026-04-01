# -*- coding: utf-8 -*-

import pandas as pd

from step0 import build_event_stream_preview, build_snapshot_context_v2


# =============================================================================
# Step1-2. 正式排序主事件流并生成事件索引
# =============================================================================

def build_event_stream_step1_ordered(trade_date: str | int, symbol: str) -> pd.DataFrame:
    """
    Step1-2:
    在 Step0 的 event_stream_v1 基础上，构造 Step1 正式使用的事件序列。

    目标：
    1) 固定 Step1 的正式排序规则
    2) 生成稳定的 event_idx
    3) 为后续 Step1 状态递推提供统一事件顺序基准

    说明：
    - 不宣称恢复交易所绝对真顺序
    - 只定义一个稳定、可复现、可用于 online 递推的处理顺序
    - SH order 同毫秒内可借助 event_seq_proxy（即 BizIndex）细排
    - mixed-source 同毫秒只做稳定处理，不做强机制断言
    """

    es = build_event_stream_preview(trade_date=trade_date, symbol=symbol).copy()

    # -------------------------------------------------------------------------
    # 1) 保证关键字段存在
    # -------------------------------------------------------------------------
    required_cols = [
        "symbol", "market", "source_table",
        "event_time_str", "event_time_ms",
        "event_family", "event_type_raw_std",
        "event_price", "event_qty", "event_side_raw_std",
        "event_seq_proxy"
    ]
    missing = [c for c in required_cols if c not in es.columns]
    if missing:
        raise ValueError(f"Step1-2 missing required columns: {missing}")

    # -------------------------------------------------------------------------
    # 2) 构造排序辅助字段
    # -------------------------------------------------------------------------
    # source_table 只作为稳定 tie-breaker，不代表交易所机制真顺序
    source_rank_map = {
        "order": 0,
        "trade": 1,
    }
    es["_source_rank"] = es["source_table"].map(source_rank_map).fillna(99).astype(int)

    # event_seq_proxy 只对 SH order 同毫秒细排有意义
    # 其他情况统一放到一个很大的值，避免干扰主排序
    seq_num = pd.to_numeric(es["event_seq_proxy"], errors="coerce")

    is_sh_order = (es["market"] == "SH") & (es["source_table"] == "order") & seq_num.notna()
    es["_seq_rank"] = seq_num.where(is_sh_order, 10**18)

    # 原始行号作为最后一级稳定排序锚，保证排序可复现
    es["_raw_row_id"] = range(len(es))

    # -------------------------------------------------------------------------
    # 3) 正式排序
    # -------------------------------------------------------------------------
    es = es.sort_values(
        by=[
            "event_time_ms",   # 第一层：时间
            "_source_rank",    # 第二层：来源稳定 tie-breaker
            "_seq_rank",       # 第三层：SH order 同毫秒用 event_seq_proxy 细排
            "_raw_row_id",     # 最后一级：稳定可复现
        ],
        ascending=[True, True, True, True],
        kind="stable"
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 4) 生成 Step1 正式事件索引
    # -------------------------------------------------------------------------
    es["event_idx"] = range(len(es))

    # -------------------------------------------------------------------------
    # 5) 调整列顺序：把 event_idx 放到前面
    # -------------------------------------------------------------------------
    front_cols = [
        "event_idx",
        "symbol", "market", "source_table",
        "event_time_str", "event_time_ms",
        "event_family", "event_type_raw_std",
        "event_price", "event_qty", "event_side_raw_std",
        "event_seq_proxy",
        "order_id", "buy_order_id", "sell_order_id",
        "trade_price", "trade_qty",
        "is_direct_observation", "is_inferred_from_trade",
        "buy_order_seen_in_order", "sell_order_seen_in_order",
        "has_inferred_side",
    ]
    keep_cols = [c for c in front_cols if c in es.columns] + [
        c for c in es.columns
        if c not in front_cols + ["_source_rank", "_seq_rank", "_raw_row_id"]
    ]
    es = es[keep_cols].copy()

    return es


def inspect_event_stream_step1_ordered(
    trade_date: str,
    symbol: str,
    n: int = 20,
    show_same_ms: int = 5,
    show_order_same_ms: int = 5
):
    """
    Step1-2 审计预览：
    1) 查看正式排序后的主事件流
    2) 检查 event_idx 是否稳定生成
    3) 查看 same-ms 事件分布
    4) 额外检查 SH order 同毫秒且 event_seq_proxy 非空的排序结果
    """
    es = build_event_stream_step1_ordered(trade_date=trade_date, symbol=symbol)

    print("\n" + "=" * 120)
    print(f"STEP1-2 ORDERED EVENT STREAM | SYMBOL={symbol}")
    print("=" * 120)

    print("[shape]")
    print(es.shape)

    print("\n[columns]")
    print(es.columns.tolist())

    print("\n[head]")
    print(es.head(n))

    print("\n[tail]")
    print(es.tail(n))

    # ---------------------------------------------------------------------
    # A. 所有 same-ms group 概览
    # ---------------------------------------------------------------------
    per_ms = es.groupby("event_time_ms").size().rename("n_events")
    same_ms = per_ms[per_ms > 1].sort_values(ascending=False)

    print("\n[same-ms event_time_ms count]")
    print(len(same_ms))

    if len(same_ms) > 0:
        print(f"\n[top {show_same_ms} same-ms groups]")
        print(same_ms.head(show_same_ms))

        for ms in same_ms.head(show_same_ms).index.tolist():
            print(f"\n--- ordered events at event_time_ms = {ms} ---")
            sub = es.loc[es["event_time_ms"] == ms].copy()

            cols = [c for c in [
                "event_idx",
                "event_time_str", "event_time_ms",
                "source_table",
                "event_family", "event_type_raw_std",
                "event_price", "event_qty", "event_side_raw_std",
                "event_seq_proxy",
                "order_id", "buy_order_id", "sell_order_id",
                "trade_price", "trade_qty"
            ] if c in sub.columns]

            print(sub[cols])

    # ---------------------------------------------------------------------
    # B. 额外检查：SH order 同毫秒且 event_seq_proxy 非空
    # ---------------------------------------------------------------------
    sh_order = es[
        (es["market"] == "SH") &
        (es["source_table"] == "order") &
        (pd.to_numeric(es["event_seq_proxy"], errors="coerce").notna())
    ].copy()

    if len(sh_order) > 0:
        sh_order_per_ms = sh_order.groupby("event_time_ms").size().rename("n_order")
        sh_order_same_ms = sh_order_per_ms[sh_order_per_ms >= 2].sort_values(ascending=False)

        print("\n[SH order same-ms groups with non-null event_seq_proxy]")
        print(len(sh_order_same_ms))

        if len(sh_order_same_ms) > 0:
            print(f"\n[top {show_order_same_ms} SH order same-ms groups]")
            print(sh_order_same_ms.head(show_order_same_ms))

            for ms in sh_order_same_ms.head(show_order_same_ms).index.tolist():
                print(f"\n--- SH order same-ms ordered sample at event_time_ms = {ms} ---")
                sub = es[
                    (es["event_time_ms"] == ms) &
                    (es["market"] == "SH")
                ].copy()

                cols = [c for c in [
                    "event_idx",
                    "event_time_str", "event_time_ms",
                    "source_table",
                    "event_family", "event_type_raw_std",
                    "event_price", "event_qty", "event_side_raw_std",
                    "event_seq_proxy",
                    "order_id", "buy_order_id", "sell_order_id",
                    "trade_price", "trade_qty"
                ] if c in sub.columns]

                print(sub[cols].sort_values(
                    by=["source_table", "event_seq_proxy", "event_idx"],
                    ascending=[True, True, True],
                    kind="stable"
                ))


# =============================================================================
# Step1-3. 按时间窗给每条事件打 session 标签
# =============================================================================

def map_time_to_session_bucket(time_str: str) -> str:
    """
    根据 event_time_str 映射标准化交易时段标签。

    当前版本：
    - 不依赖 trading_phase_raw
    - 只使用时间窗 fallback
    - 面向普通 A 股日内时段划分
    """
    if pd.isna(time_str):
        return "unknown_time"

    t = str(time_str)

    if t < "09:15:00":
        return "pre_open_other"

    if "09:15:00" <= t < "09:25:00":
        return "open_call"

    if "09:25:00" <= t < "09:30:00":
        return "between_open_and_continuous"

    if "09:30:00" <= t < "11:30:00":
        return "continuous_am"

    if "11:30:00" <= t < "13:00:00":
        return "lunch_break"

    if "13:00:00" <= t < "14:57:00":
        return "continuous_pm"

    if "14:57:00" <= t < "15:00:00":
        return "close_call"

    if t >= "15:00:00":
        return "post_close"

    return "unknown_time"


def build_event_stream_step1_with_session(trade_date: str | int, symbol: str) -> pd.DataFrame:
    """
    Step1-3:
    在 Step1-2 正式排序后的主事件流基础上，为每条事件打标准化 session 标签。

    新增字段：
    - session_bucket_std
    - session_source
    """
    es = build_event_stream_step1_ordered(trade_date=trade_date, symbol=symbol).copy()

    # -------------------------------------------------------------------------
    # 1) 基础字段检查
    # -------------------------------------------------------------------------
    required_cols = ["event_idx", "event_time_str", "event_time_ms", "symbol", "market"]
    missing = [c for c in required_cols if c not in es.columns]
    if missing:
        raise ValueError(f"Step1-3 missing required columns: {missing}")

    # -------------------------------------------------------------------------
    # 2) 基于时间窗映射标准 session
    # -------------------------------------------------------------------------
    es["session_bucket_std"] = es["event_time_str"].map(map_time_to_session_bucket)

    # 当前数据下 session 统一来自时间规则 fallback
    es["session_source"] = "time_fallback"

    return es


def inspect_event_stream_step1_with_session(trade_date: str | int, symbol: str, n: int = 20):
    """
    Step1-3 审计预览：
    1) 查看 session 标签分布
    2) 查看各 session 的头部样本
    3) 检查 event_time_str 到 session 的映射是否符合预期
    """
    es = build_event_stream_step1_with_session(trade_date=trade_date, symbol=symbol)

    print("\n" + "=" * 120)
    print(f"STEP1-3 EVENT STREAM WITH SESSION | SYMBOL={symbol}")
    print("=" * 120)

    print("[shape]")
    print(es.shape)

    print("\n[session_bucket_std counts]")
    print(es["session_bucket_std"].value_counts(dropna=False))

    print("\n[session_source counts]")
    print(es["session_source"].value_counts(dropna=False))

    print("\n[head]")
    cols = [c for c in [
        "event_idx",
        "event_time_str", "event_time_ms",
        "source_table",
        "event_family", "event_type_raw_std",
        "session_bucket_std", "session_source",
        "event_price", "event_qty", "event_side_raw_std",
        "event_seq_proxy"
    ] if c in es.columns]
    print(es[cols].head(n))

    print("\n[sample first rows per session]")
    for sess in [
        "pre_open_other",
        "open_call",
        "between_open_and_continuous",
        "continuous_am",
        "lunch_break",
        "continuous_pm",
        "close_call",
        "post_close",
        "unknown_time",
    ]:
        sub = es.loc[es["session_bucket_std"] == sess, cols]
        if len(sub) == 0:
            continue
        print(f"\n--- session_bucket_std = {sess} ---")
        print(sub.head(10))

# =============================================================================
# Step1-4. 给每条主事件对齐最近可用 snapshot
# =============================================================================

from step0 import build_snapshot_context_v2


def build_event_stream_step1_with_snapshot(trade_date: str | int, symbol: str) -> pd.DataFrame:
    """
    Step1-4:
    在 Step1-3 的事件流基础上，对齐最近可用 snapshot（只能向过去找）。

    新增字段：
    - ref_snapshot_time_str
    - ref_snapshot_time_ms
    - snapshot_lag_ms
    - snap_last
    - snap_cum_trade_count
    - snap_cum_volume
    - snap_cum_amount
    - snap_bid1_price
    - snap_bid1_qty
    - snap_ask1_price
    - snap_ask1_qty
    """
    es = build_event_stream_step1_with_session(trade_date=trade_date, symbol=symbol).copy()

    # 读取 Step0 snapshot，并只保留可用上下文
    _, snap = build_snapshot_context_v2(trade_date=trade_date, symbol=symbol)
    snap = snap.loc[snap["is_usable_snapshot_context"] == 1].copy()

    # 基础字段检查
    es_required = ["symbol", "event_time_ms", "event_time_str", "event_idx"]
    snap_required = [
        "symbol", "event_time_ms", "event_time_str",
        "last", "cum_trade_count", "cum_volume", "cum_amount",
        "bid1_price", "bid1_qty", "ask1_price", "ask1_qty"
    ]
    miss_es = [c for c in es_required if c not in es.columns]
    miss_snap = [c for c in snap_required if c not in snap.columns]
    if miss_es:
        raise ValueError(f"Step1-4 missing event columns: {miss_es}")
    if miss_snap:
        raise ValueError(f"Step1-4 missing snapshot columns: {miss_snap}")

    # snapshot 重命名，避免和事件字段混淆
    snap = snap.rename(columns={
        "event_time_str": "ref_snapshot_time_str",
        "event_time_ms": "ref_snapshot_time_ms",
        "last": "snap_last",
        "cum_trade_count": "snap_cum_trade_count",
        "cum_volume": "snap_cum_volume",
        "cum_amount": "snap_cum_amount",
        "bid1_price": "snap_bid1_price",
        "bid1_qty": "snap_bid1_qty",
        "ask1_price": "snap_ask1_price",
        "ask1_qty": "snap_ask1_qty",
    })

    # merge_asof 之前必须排序
    es = es.sort_values(["symbol", "event_time_ms", "event_idx"], kind="stable").reset_index(drop=True)
    snap = snap.sort_values(["symbol", "ref_snapshot_time_ms"], kind="stable").reset_index(drop=True)

    # 只向过去找最近 snapshot
    out = pd.merge_asof(
        es,
        snap[[
            "symbol",
            "ref_snapshot_time_ms",
            "ref_snapshot_time_str",
            "snap_last",
            "snap_cum_trade_count",
            "snap_cum_volume",
            "snap_cum_amount",
            "snap_bid1_price",
            "snap_bid1_qty",
            "snap_ask1_price",
            "snap_ask1_qty",
        ]],
        left_on="event_time_ms",
        right_on="ref_snapshot_time_ms",
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    )

    out["snapshot_lag_ms"] = out["event_time_ms"] - out["ref_snapshot_time_ms"]

    return out


def inspect_event_stream_step1_with_snapshot(trade_date: str | int, symbol: str, n: int = 20):
    """
    Step1-4 审计预览：
    1) 查看对齐后表结构
    2) 检查 snapshot_lag_ms 是否非负
    3) 查看不同 session 下 snapshot 覆盖情况
    """
    es = build_event_stream_step1_with_snapshot(trade_date=trade_date, symbol=symbol)

    print("\n" + "=" * 120)
    print(f"STEP1-4 EVENT STREAM WITH SNAPSHOT | SYMBOL={symbol}")
    print("=" * 120)

    print("[shape]")
    print(es.shape)

    print("\n[head]")
    cols = [c for c in [
        "event_idx",
        "event_time_str", "event_time_ms",
        "session_bucket_std",
        "event_family", "event_type_raw_std",
        "ref_snapshot_time_str", "ref_snapshot_time_ms", "snapshot_lag_ms",
        "snap_last",
        "snap_cum_trade_count", "snap_cum_volume", "snap_cum_amount",
        "snap_bid1_price", "snap_bid1_qty", "snap_ask1_price", "snap_ask1_qty"
    ] if c in es.columns]
    print(es[cols].head(n))

    print("\n[snapshot matched count]")
    print(es["ref_snapshot_time_ms"].notna().value_counts(dropna=False))

    print("\n[snapshot_lag_ms summary]")
    print(es["snapshot_lag_ms"].describe())

    neg_lag = es.loc[es["snapshot_lag_ms"].notna() & (es["snapshot_lag_ms"] < 0)]
    print("\n[negative snapshot_lag_ms rows]")
    print(len(neg_lag))

    print("\n[snapshot coverage by session]")
    coverage = (
        es.assign(has_snapshot=es["ref_snapshot_time_ms"].notna().astype(int))
          .groupby("session_bucket_std")["has_snapshot"]
          .agg(["count", "sum", "mean"])
          .sort_index()
    )
    print(coverage)

    print("\n[sample rows without snapshot]")
    print(es.loc[es["ref_snapshot_time_ms"].isna(), cols].head(20))


# =============================================================================
# Step1-5. 从 snapshot 原始字段构造最小市场状态字段
# =============================================================================

def build_event_stream_step1_with_market_state(trade_date: str | int, symbol: str) -> pd.DataFrame:
    """
    Step1-5:
    在 Step1-4 基础上，构造最小市场状态字段。

    新增字段：
    - has_snapshot
    - snapshot_is_stale
    - snap_mid_price
    - snap_spread_abs
    - snap_spread_bps
    - snap_top_depth_total
    - snap_top_depth_imbalance
    """
    es = build_event_stream_step1_with_snapshot(trade_date=trade_date, symbol=symbol).copy()

    # -------------------------------------------------------------------------
    # 1) 基础可用性标记
    # -------------------------------------------------------------------------
    es["has_snapshot"] = es["ref_snapshot_time_ms"].notna().astype(int)

    # 先用一个简单且透明的 stale 定义：
    # 有 snapshot 且 lag > 5000ms 记为 stale
    es["snapshot_is_stale"] = (
        (es["ref_snapshot_time_ms"].notna()) &
        (pd.to_numeric(es["snapshot_lag_ms"], errors="coerce") > 5000)
    ).astype(int)

    # -------------------------------------------------------------------------
    # 2) 数值化，避免后面算术报错
    # -------------------------------------------------------------------------
    bid1_p = pd.to_numeric(es["snap_bid1_price"], errors="coerce")
    ask1_p = pd.to_numeric(es["snap_ask1_price"], errors="coerce")
    bid1_q = pd.to_numeric(es["snap_bid1_qty"], errors="coerce")
    ask1_q = pd.to_numeric(es["snap_ask1_qty"], errors="coerce")

    # -------------------------------------------------------------------------
    # 3) 盘口价格结构
    # -------------------------------------------------------------------------
    es["snap_mid_price"] = (bid1_p + ask1_p) / 2.0
    es["snap_spread_abs"] = ask1_p - bid1_p

    mid = pd.to_numeric(es["snap_mid_price"], errors="coerce")
    es["snap_spread_bps"] = pd.NA
    valid_mid = mid.notna() & (mid > 0)
    es.loc[valid_mid, "snap_spread_bps"] = (
        es.loc[valid_mid, "snap_spread_abs"] / mid.loc[valid_mid] * 10000.0
    )

    # -------------------------------------------------------------------------
    # 4) 顶层深度结构
    # -------------------------------------------------------------------------
    es["snap_top_depth_total"] = bid1_q + ask1_q

    denom = pd.to_numeric(es["snap_top_depth_total"], errors="coerce")
    es["snap_top_depth_imbalance"] = pd.NA
    valid_denom = denom.notna() & (denom > 0)
    es.loc[valid_denom, "snap_top_depth_imbalance"] = (
        (bid1_q.loc[valid_denom] - ask1_q.loc[valid_denom]) / denom.loc[valid_denom]
    )

    return es


def inspect_event_stream_step1_with_market_state(trade_date: str | int, symbol: str, n: int = 20):
    """
    Step1-5 审计预览：
    1) 查看最小市场状态字段
    2) 检查 spread / imbalance 的基本分布
    3) 检查 has_snapshot / snapshot_is_stale
    """
    es = build_event_stream_step1_with_market_state(trade_date=trade_date, symbol=symbol)

    print("\n" + "=" * 120)
    print(f"STEP1-5 EVENT STREAM WITH MARKET STATE | SYMBOL={symbol}")
    print("=" * 120)

    print("[shape]")
    print(es.shape)

    cols = [c for c in [
        "event_idx",
        "event_time_str", "event_time_ms",
        "session_bucket_std",
        "event_family", "event_type_raw_std",
        "ref_snapshot_time_str", "snapshot_lag_ms",
        "has_snapshot", "snapshot_is_stale",
        "snap_bid1_price", "snap_bid1_qty",
        "snap_ask1_price", "snap_ask1_qty",
        "snap_mid_price",
        "snap_spread_abs", "snap_spread_bps",
        "snap_top_depth_total", "snap_top_depth_imbalance",
    ] if c in es.columns]

    print("\n[head]")
    print(es[cols].head(n))

    print("\n[has_snapshot counts]")
    print(es["has_snapshot"].value_counts(dropna=False))

    print("\n[snapshot_is_stale counts]")
    print(es["snapshot_is_stale"].value_counts(dropna=False))

    print("\n[snap_spread_abs summary]")
    print(pd.to_numeric(es["snap_spread_abs"], errors="coerce").describe())

    print("\n[snap_spread_bps summary]")
    print(pd.to_numeric(es["snap_spread_bps"], errors="coerce").describe())

    print("\n[snap_top_depth_total summary]")
    print(pd.to_numeric(es["snap_top_depth_total"], errors="coerce").describe())

    print("\n[snap_top_depth_imbalance summary]")
    print(pd.to_numeric(es["snap_top_depth_imbalance"], errors="coerce").describe())

    neg_spread = es.loc[
        pd.to_numeric(es["snap_spread_abs"], errors="coerce").notna() &
        (pd.to_numeric(es["snap_spread_abs"], errors="coerce") < 0)
    ]
    print("\n[negative spread rows]")
    print(len(neg_spread))

    bad_imb = es.loc[
        pd.to_numeric(es["snap_top_depth_imbalance"], errors="coerce").notna() &
        (
            (pd.to_numeric(es["snap_top_depth_imbalance"], errors="coerce") < -1) |
            (pd.to_numeric(es["snap_top_depth_imbalance"], errors="coerce") > 1)
        )
    ]
    print("\n[imbalance outside [-1, 1]]")
    print(len(bad_imb))

    print("\n[stale ratio by session]")
    stale_by_session = (
        es.groupby("session_bucket_std")["snapshot_is_stale"]
          .mean()
          .sort_index()
    )
    print(stale_by_session)

    print("\n[sample stale rows]")
    print(es.loc[es["snapshot_is_stale"] == 1, cols].head(20))

# =============================================================================
# Step1-6 (Function 1). 状态变化层
# =============================================================================

def build_event_stream_step1_with_state_change(trade_date: str | int, symbol: str) -> pd.DataFrame:
    """
    Step1-6 / Function 1:
    在 Step1-5 基础上，构造最小状态变化层。

    变化参考规则：
    - 只在同一 session_bucket_std 内比较
    - 只使用 has_snapshot == 1 的事件作为可比较状态基准
    - 当前事件若无 snapshot，则不计算变化量
    - session 内第一条可比较事件没有前值，变化量记 NA

    新增字段：
    - prev_ref_snapshot_time_ms
    - prev_snap_last
    - prev_snap_spread_abs
    - prev_snap_top_depth_total
    - prev_snap_top_depth_imbalance
    - state_change_is_valid
    - d_snap_last
    - d_snap_spread_abs
    - d_snap_top_depth_total
    - d_snap_top_depth_imbalance
    """
    es = build_event_stream_step1_with_market_state(trade_date=trade_date, symbol=symbol).copy()

    # -------------------------------------------------------------------------
    # 1) 只在 has_snapshot == 1 的事件上建立“可比较状态序列”
    # -------------------------------------------------------------------------
    has_snap_mask = es["has_snapshot"] == 1
    state_base = es.loc[has_snap_mask, [
        "event_idx",
        "session_bucket_std",
        "ref_snapshot_time_ms",
        "snap_last",
        "snap_spread_abs",
        "snap_top_depth_total",
        "snap_top_depth_imbalance",
    ]].copy()

    # session 内按 event_idx 排序后，取前一个可比较状态
    state_base = state_base.sort_values(
        ["session_bucket_std", "event_idx"],
        ascending=[True, True],
        kind="stable"
    ).reset_index(drop=True)

    state_base["prev_ref_snapshot_time_ms"] = (
        state_base.groupby("session_bucket_std")["ref_snapshot_time_ms"].shift(1)
    )
    state_base["prev_snap_last"] = (
        state_base.groupby("session_bucket_std")["snap_last"].shift(1)
    )
    state_base["prev_snap_spread_abs"] = (
        state_base.groupby("session_bucket_std")["snap_spread_abs"].shift(1)
    )
    state_base["prev_snap_top_depth_total"] = (
        state_base.groupby("session_bucket_std")["snap_top_depth_total"].shift(1)
    )
    state_base["prev_snap_top_depth_imbalance"] = (
        state_base.groupby("session_bucket_std")["snap_top_depth_imbalance"].shift(1)
    )

    # 只把前值字段 merge 回原表
    prev_cols = [
        "event_idx",
        "prev_ref_snapshot_time_ms",
        "prev_snap_last",
        "prev_snap_spread_abs",
        "prev_snap_top_depth_total",
        "prev_snap_top_depth_imbalance",
    ]
    es = es.merge(
        state_base[prev_cols],
        on="event_idx",
        how="left",
        sort=False
    )

    # -------------------------------------------------------------------------
    # 2) 定义变化量是否可用
    # -------------------------------------------------------------------------
    es["state_change_is_valid"] = (
        (es["has_snapshot"] == 1) &
        es["prev_ref_snapshot_time_ms"].notna()
    ).astype(int)

    # -------------------------------------------------------------------------
    # 3) 计算最小变化量
    # -------------------------------------------------------------------------
    cur_last = pd.to_numeric(es["snap_last"], errors="coerce")
    prev_last = pd.to_numeric(es["prev_snap_last"], errors="coerce")

    cur_spread = pd.to_numeric(es["snap_spread_abs"], errors="coerce")
    prev_spread = pd.to_numeric(es["prev_snap_spread_abs"], errors="coerce")

    cur_depth = pd.to_numeric(es["snap_top_depth_total"], errors="coerce")
    prev_depth = pd.to_numeric(es["prev_snap_top_depth_total"], errors="coerce")

    cur_imb = pd.to_numeric(es["snap_top_depth_imbalance"], errors="coerce")
    prev_imb = pd.to_numeric(es["prev_snap_top_depth_imbalance"], errors="coerce")

    es["d_snap_last"] = pd.NA
    es["d_snap_spread_abs"] = pd.NA
    es["d_snap_top_depth_total"] = pd.NA
    es["d_snap_top_depth_imbalance"] = pd.NA

    valid = es["state_change_is_valid"] == 1

    es.loc[valid, "d_snap_last"] = cur_last.loc[valid] - prev_last.loc[valid]
    es.loc[valid, "d_snap_spread_abs"] = cur_spread.loc[valid] - prev_spread.loc[valid]
    es.loc[valid, "d_snap_top_depth_total"] = cur_depth.loc[valid] - prev_depth.loc[valid]
    es.loc[valid, "d_snap_top_depth_imbalance"] = cur_imb.loc[valid] - prev_imb.loc[valid]

    return es


def inspect_event_stream_step1_with_state_change(trade_date: str | int, symbol: str, n: int = 20):
    """
    Step1-6 / Function 1 审计：
    1) 检查变化量是否只在有前参考状态时产生
    2) 检查各 session 首条可比较状态是否没有变化量
    3) 查看主要变化量分布
    """
    es = build_event_stream_step1_with_state_change(trade_date=trade_date, symbol=symbol)

    print("\n" + "=" * 120)
    print(f"STEP1-6 EVENT STREAM WITH STATE CHANGE | SYMBOL={symbol}")
    print("=" * 120)

    print("[shape]")
    print(es.shape)

    cols = [c for c in [
        "event_idx",
        "event_time_str", "event_time_ms",
        "session_bucket_std",
        "has_snapshot",
        "snapshot_is_stale",
        "ref_snapshot_time_ms",
        "prev_ref_snapshot_time_ms",
        "state_change_is_valid",
        "snap_last", "prev_snap_last", "d_snap_last",
        "snap_spread_abs", "prev_snap_spread_abs", "d_snap_spread_abs",
        "snap_top_depth_total", "prev_snap_top_depth_total", "d_snap_top_depth_total",
        "snap_top_depth_imbalance", "prev_snap_top_depth_imbalance", "d_snap_top_depth_imbalance",
    ] if c in es.columns]

    print("\n[head]")
    print(es[cols].head(n))

    print("\n[state_change_is_valid counts]")
    print(es["state_change_is_valid"].value_counts(dropna=False))

    print("\n[state_change_is_valid by session]")
    by_session = (
        es.groupby("session_bucket_std")["state_change_is_valid"]
          .agg(["count", "sum", "mean"])
          .sort_index()
    )
    print(by_session)

    print("\n[d_snap_last summary]")
    print(pd.to_numeric(es["d_snap_last"], errors="coerce").describe())

    print("\n[d_snap_spread_abs summary]")
    print(pd.to_numeric(es["d_snap_spread_abs"], errors="coerce").describe())

    print("\n[d_snap_top_depth_total summary]")
    print(pd.to_numeric(es["d_snap_top_depth_total"], errors="coerce").describe())

    print("\n[d_snap_top_depth_imbalance summary]")
    print(pd.to_numeric(es["d_snap_top_depth_imbalance"], errors="coerce").describe())

    # -------------------------------------------------------------------------
    # 检查：无前参考状态时，不应出现 valid=1
    # -------------------------------------------------------------------------
    bad_valid = es.loc[
        (es["state_change_is_valid"] == 1) &
        (es["prev_ref_snapshot_time_ms"].isna())
    ]
    print("\n[bad valid rows: valid=1 but prev_ref_snapshot_time_ms is NA]")
    print(len(bad_valid))

    # -------------------------------------------------------------------------
    # 检查：session 内首条可比较状态
    # 这些行应该 has_snapshot=1，但 prev_ref_snapshot_time_ms 为空，valid=0
    # -------------------------------------------------------------------------
    first_comparable = (
        es.loc[es["has_snapshot"] == 1]
          .sort_values(["session_bucket_std", "event_idx"], kind="stable")
          .groupby("session_bucket_std", as_index=False)
          .head(1)
    )

    print("\n[first comparable row per session]")
    first_cols = [c for c in [
        "event_idx",
        "event_time_str",
        "session_bucket_std",
        "has_snapshot",
        "ref_snapshot_time_ms",
        "prev_ref_snapshot_time_ms",
        "state_change_is_valid",
        "d_snap_last",
        "d_snap_spread_abs",
        "d_snap_top_depth_total",
        "d_snap_top_depth_imbalance",
    ] if c in es.columns]
    print(first_comparable[first_cols])

    print("\n[sample valid change rows]")
    print(es.loc[es["state_change_is_valid"] == 1, cols].head(20))

import numpy as np
import pandas as pd


def build_event_stream_step1_with_price_position(trade_date: str | int, symbol: str) -> pd.DataFrame:
    """
    在 Step1-6 基础上，给每条事件补：
    1) event_price_position_std
    2) event_price_aggressive_flag

    说明：
    - 这里只做“事件价格相对当时 best bid / best ask 的位置”
    - 不直接把它当成最终买卖主动方向
    - 对缺快照 / 缺价格 / 异常盘口 保守给 unknown
    """
    es = build_event_stream_step1_with_state_change(trade_date=trade_date, symbol=symbol).copy()

    p = pd.to_numeric(es["event_price"], errors="coerce")
    b = pd.to_numeric(es["snap_bid1_price"], errors="coerce")
    a = pd.to_numeric(es["snap_ask1_price"], errors="coerce")

    valid = p.notna() & b.notna() & a.notna() & (b <= a)

    pos = pd.Series("unknown", index=es.index, dtype="object")

    pos.loc[valid & (p < b)] = "lt_bid1"
    pos.loc[valid & (p == b)] = "at_bid1"
    pos.loc[valid & (p > b) & (p < a)] = "inside_spread"
    pos.loc[valid & (p == a)] = "at_ask1"
    pos.loc[valid & (p > a)] = "gt_ask1"

    flag = pd.Series("unknown", index=es.index, dtype="object")
    flag.loc[valid & (p <= b)] = "passive"
    flag.loc[valid & (p > b) & (p < a)] = "inside"
    flag.loc[valid & (p >= a)] = "aggressive"

    es["event_price_position_std"] = pos
    es["event_price_aggressive_flag"] = flag

    return es


def inspect_event_stream_step1_with_price_position(trade_date: str | int, symbol: str, n: int = 20) -> None:
    es = build_event_stream_step1_with_price_position(trade_date=trade_date, symbol=symbol)

    print("\n" + "=" * 120)
    print(f"STEP1-7 EVENT STREAM WITH PRICE POSITION | SYMBOL={symbol}")
    print("=" * 120)

    print("[shape]")
    print(es.shape)

    show_cols = [
        "event_idx",
        "event_time_str",
        "event_time_ms",
        "session_bucket_std",
        "source_table",
        "event_family",
        "event_type_raw_std",
        "event_side_raw_std",
        "event_price",
        "snap_bid1_price",
        "snap_ask1_price",
        "event_price_position_std",
        "event_price_aggressive_flag",
    ]

    print("\n[head]")
    print(es[show_cols].head(n))

    print("\n[event_price_position_std counts]")
    print(es["event_price_position_std"].value_counts(dropna=False))

    print("\n[event_price_aggressive_flag counts]")
    print(es["event_price_aggressive_flag"].value_counts(dropna=False))

    print("\n[position by source_table]")
    print(
        es.pivot_table(
            index="event_price_position_std",
            columns="source_table",
            values="event_idx",
            aggfunc="count",
            fill_value=0,
        )
    )

    print("\n[position by event_family]")
    print(
        es.pivot_table(
            index="event_price_position_std",
            columns="event_family",
            values="event_idx",
            aggfunc="count",
            fill_value=0,
        )
    )

    print("\n[sample inside_spread rows]")
    inside = es[es["event_price_position_std"] == "inside_spread"][show_cols]
    print(inside.head(20))

    print("\n[sample aggressive rows]")
    aggr = es[es["event_price_aggressive_flag"] == "aggressive"][show_cols]
    print(aggr.head(20))

    print("\n[sample unknown rows]")
    unk = es[es["event_price_position_std"] == "unknown"][show_cols]
    print(unk.head(20))


import numpy as np
import pandas as pd


def build_event_stream_step1_out_of_quote_audit(trade_date: str | int, symbol: str) -> pd.DataFrame:
    """
    Step1-8:
    对 Step1-7 的结果做 trade 越界审计：
    只看有 snapshot 的 trade，筛出
      - trade_price > ask1
      - trade_price < bid1
    """
    es = build_event_stream_step1_with_price_position(trade_date=trade_date, symbol=symbol).copy()

    trade_mask = es["source_table"].eq("trade")
    has_snap_mask = es["has_snapshot"].eq(1)

    pos_mask = es["event_price_position_std"].isin(["gt_ask1", "lt_bid1"])

    out = es.loc[trade_mask & has_snap_mask & pos_mask].copy()

    if out.empty:
        out["out_of_quote_side"] = pd.Series(dtype="object")
        out["abs_out_of_quote_dist"] = pd.Series(dtype="float64")
        out["out_of_quote_bps"] = pd.Series(dtype="float64")
        return out

    out["out_of_quote_side"] = np.where(
        out["event_price_position_std"].eq("gt_ask1"),
        "trade_above_ask1",
        "trade_below_bid1",
    )

    out["abs_out_of_quote_dist"] = np.where(
        out["event_price_position_std"].eq("gt_ask1"),
        out["trade_price"] - out["snap_ask1_price"],
        out["snap_bid1_price"] - out["trade_price"],
    )

    ref_price = np.where(
        out["event_price_position_std"].eq("gt_ask1"),
        out["snap_ask1_price"],
        out["snap_bid1_price"],
    )

    out["out_of_quote_bps"] = np.where(
        pd.notna(ref_price) & (ref_price != 0),
        out["abs_out_of_quote_dist"] / ref_price * 10000.0,
        np.nan,
    )

    return out


def inspect_event_stream_step1_out_of_quote_audit(trade_date: str | int, symbol: str, n: int = 20) -> None:
    """
    Step1-8 审计输出：
    - 越界 trade 数量
    - 分 session 分布
    - lag 分布
    - 越界距离分布
    - 具体样本
    """
    out = build_event_stream_step1_out_of_quote_audit(trade_date=trade_date, symbol=symbol)

    print("\n" + "=" * 120)
    print(f"STEP1-8 OUT-OF-QUOTE TRADE AUDIT | SYMBOL={symbol}")
    print("=" * 120)

    print("[shape]")
    print(out.shape)

    if out.empty:
        print("\n[no out-of-quote trade rows found]")
        return

    show_cols = [
        "event_idx",
        "event_time_str",
        "event_time_ms",
        "session_bucket_std",
        "event_price_position_std",
        "out_of_quote_side",
        "trade_price",
        "snap_bid1_price",
        "snap_ask1_price",
        "snapshot_lag_ms",
        "abs_out_of_quote_dist",
        "out_of_quote_bps",
        "buy_order_id",
        "sell_order_id",
    ]

    print("\n[out_of_quote_side counts]")
    print(out["out_of_quote_side"].value_counts(dropna=False))

    print("\n[position counts]")
    print(out["event_price_position_std"].value_counts(dropna=False))

    print("\n[out-of-quote by session]")
    print(
        out.groupby(["session_bucket_std", "out_of_quote_side"])
        .size()
        .unstack(fill_value=0)
    )

    print("\n[snapshot_lag_ms summary]")
    print(out["snapshot_lag_ms"].describe())

    print("\n[snapshot_lag_ms by side]")
    print(
        out.groupby("out_of_quote_side")["snapshot_lag_ms"]
        .describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    )

    print("\n[abs_out_of_quote_dist summary]")
    print(out["abs_out_of_quote_dist"].describe())

    print("\n[out_of_quote_bps summary]")
    print(out["out_of_quote_bps"].describe())

    print("\n[top 20 largest out_of_quote_bps rows]")
    print(
        out.sort_values(["out_of_quote_bps", "snapshot_lag_ms"], ascending=[False, False])[
            show_cols
        ].head(n)
    )

    print("\n[top 20 largest snapshot_lag_ms rows]")
    print(
        out.sort_values(["snapshot_lag_ms", "out_of_quote_bps"], ascending=[False, False])[
            show_cols
        ].head(n)
    )

    print("\n[first 20 out-of-quote rows]")
    print(out[show_cols].head(n))

# =============================================================================
# Step1 artifact cache for downstream Step2
# =============================================================================

from pathlib import Path


def build_and_save_step1_final_artifact(
    trade_date: str | int,
    symbol: str,
    out_dir: str | Path,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 120)
    print("STEP1 FINAL ARTIFACT BUILD")
    print("=" * 120)
    print(f"[artifact] out_dir = {out_path.resolve()}")
    print(f"[artifact] trade_date = {trade_date}")
    print(f"[artifact] symbol = {symbol}")

    es = build_event_stream_step1_with_price_position(
        trade_date=trade_date,
        symbol=symbol,
    ).copy()

    artifact_file = out_path / "step1_event_stream_final.pkl"
    meta_file = out_path / "step1_event_stream_final_meta.txt"

    es.to_pickle(artifact_file)

    meta_text = f"""trade_date={trade_date}
symbol={symbol}
shape={es.shape}
columns={list(es.columns)}
"""
    meta_file.write_text(meta_text, encoding="utf-8")

    print("[artifact] saved successfully")
    print(f"[artifact] artifact_file = {artifact_file}")
    print(f"[artifact] es shape = {es.shape}")

    return es


def load_step1_final_artifact(out_dir: str) -> pd.DataFrame:
    """
    读取 Step1 最终事件流 artifact。
    """
    out_path = Path(out_dir)
    artifact_file = out_path / "step1_event_stream_final.pkl"

    if not artifact_file.exists():
        raise FileNotFoundError(f"missing step1 artifact file: {artifact_file}")

    print("\n" + "=" * 120)
    print("STEP1 FINAL ARTIFACT LOAD")
    print("=" * 120)
    print(f"[artifact] loading from = {artifact_file}")

    es = pd.read_pickle(artifact_file)

    print(f"[artifact] es shape = {es.shape}")
    return es


def inspect_step1_final_artifact(out_dir: str, n: int = 10) -> None:
    es = load_step1_final_artifact(out_dir)

    print("\n[head]")
    print(es.head(n))

    print("\n[columns]")
    print(es.columns.tolist())

    print("\n[dtypes]")
    print(es.dtypes)
