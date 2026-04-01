# -*- coding: utf-8 -*-
"""
Step0 
================================

目标：
1) raw order / raw trade -> event_stream_v1 (preview / builder)
2) raw tick -> build_snapshot_context_v2 (preview / builder)
3) clean order / clean trade -> order_lifecycle_expost_final_v1 (boundary preview)
4) 明确 online / ex post 边界
5) 提供关键审计入口：
   - order-trade link audit
   - event ordering audit
   - tick around close audit

当前约束：
- 单日单票样本审计与预览
- 不推进 Step1
- 不做 episode 切分
- 不做 token 学习
- 不做意图推断
- 不做全量批处理
"""

from pathlib import Path
import pandas as pd


# =============================================================================
# 配置
# =============================================================================
RAW_ROOT = Path("/data105/Level2_Data/ftr_data")
CLEAN_ROOT = Path("/data105/Level2_Data/clean_data")

DEFAULT_SAMPLES = {
    "SZ": "000001",
    "SH": "600000",
}


# =============================================================================
# 通用工具
# =============================================================================
def hhmmss_to_ms(x: str):
    if pd.isna(x):
        return pd.NA
    s = str(x)
    hh, mm, rest = s.split(":")
    if "." in rest:
        ss, ms = rest.split(".")
        ms = ms.ljust(3, "0")[:3]
    else:
        ss, ms = rest, "000"
    return (
        int(hh) * 3600 * 1000
        + int(mm) * 60 * 1000
        + int(ss) * 1000
        + int(ms)
    )


def infer_market_from_symbol(symbol: str):
    symbol = str(symbol)
    if symbol.startswith("6"):
        return "SH"
    if symbol.startswith("0") or symbol.startswith("3"):
        return "SZ"
    raise ValueError(f"Cannot infer market from symbol: {symbol}")


def load_raw(trade_date: str | int, symbol: str, table: str):
    trade_date = str(trade_date)
    path = RAW_ROOT / trade_date / table / f"{symbol}.ftr"
    return pd.read_feather(path)


def load_clean(trade_date: str | int, symbol: str, table: str):
    trade_date = str(trade_date)
    path = CLEAN_ROOT / trade_date / table / f"{symbol}.ftr"
    return pd.read_feather(path)


def pct(num, den):
    if den == 0:
        return None
    return num / den


def print_ratio(title, hit, total):
    ratio = pct(hit, total)
    if ratio is None:
        print(f"{title}: total=0")
    else:
        print(f"{title}: hit={hit}, total={total}, ratio={ratio:.6f}")


# =============================================================================
# Part A. 原始字段标准化
# =============================================================================
def validate_market_by_columns(df: pd.DataFrame, table: str, inferred_market: str):
    cols = set(df.columns)

    if table == "order":
        looks_sz = {"ApplSeqNum", "OrderQty", "Side", "OrdType"}.issubset(cols)
        looks_sh = {"OrderNO", "Balance", "OrderBSFlag", "BizIndex"}.issubset(cols)
    elif table == "trade":
        looks_sz = {"BidApplSeqNum", "OfferApplSeqNum", "ExecType"}.issubset(cols)
        looks_sh = {"TradeBuyNo", "TradeSellNo", "TradeBSFlag"}.issubset(cols)
    elif table == "tick":
        # tick 不强做列校验；旧数据沪深快照字段差异没 order/trade 那么稳定
        return
    else:
        return

    if inferred_market == "SZ" and looks_sh and not looks_sz:
        raise ValueError(f"Symbol says SZ but columns look SH for table={table}")
    if inferred_market == "SH" and looks_sz and not looks_sh:
        raise ValueError(f"Symbol says SH but columns look SZ for table={table}")


def get_raw_time_col(df: pd.DataFrame, table: str, market: str):
    cols = set(df.columns)

    if table == "order":
        if market == "SZ" and "TransactTime" in cols:
            return "TransactTime"
        if market == "SH" and "OrderTime" in cols:
            return "OrderTime"

    if table == "trade":
        if market == "SZ" and "TransactTime" in cols:
            return "TransactTime"
        if market == "SH" and "TradTime" in cols:
            return "TradTime"

    if table == "tick":
        if "UpdateTime" in cols:
            return "UpdateTime"

    raise ValueError(f"Cannot find raw time column for market={market}, table={table}")


def add_standard_fields(df: pd.DataFrame, symbol: str, table: str):
    market = infer_market_from_symbol(symbol)
    validate_market_by_columns(df, table, market)
    time_col = get_raw_time_col(df, table, market)

    out = df.copy()

    out.insert(0, "symbol", symbol)
    out.insert(1, "market", market)
    out.insert(2, "source_table", table)
    out.insert(3, "raw_time_col", time_col)

    out["event_time_str"] = out[time_col].astype(str)
    out["event_time_ms"] = out["event_time_str"].map(hhmmss_to_ms)

    if table == "order":
        if market == "SZ" and "ApplSeqNum" in out.columns:
            out["order_id"] = out["ApplSeqNum"]
        elif market == "SH" and "OrderNO" in out.columns:
            out["order_id"] = out["OrderNO"]

    if table == "trade":
        if market == "SZ":
            if "BidApplSeqNum" in out.columns:
                out["buy_order_id"] = out["BidApplSeqNum"]
            if "OfferApplSeqNum" in out.columns:
                out["sell_order_id"] = out["OfferApplSeqNum"]
            if "LastPx" in out.columns:
                out["trade_price"] = out["LastPx"]
            if "LastQty" in out.columns:
                out["trade_qty"] = out["LastQty"]

        elif market == "SH":
            if "TradeBuyNo" in out.columns:
                out["buy_order_id"] = out["TradeBuyNo"]
            if "TradeSellNo" in out.columns:
                out["sell_order_id"] = out["TradeSellNo"]
            if "TradPrice" in out.columns:
                out["trade_price"] = out["TradPrice"]
            if "TradVolume" in out.columns:
                out["trade_qty"] = out["TradVolume"]

    return out


def add_event_type_fields(df: pd.DataFrame):
    """
    最终版本：
    SZ order: order_add_like
    SZ trade: ExecType 70 -> trade_like, 52 -> cancel_like
    SH order: OrderType A -> order_add_like, D -> order_cancel_like
    SH trade: trade_like
    """
    out = df.copy()
    market = out["market"].iloc[0]
    table = out["source_table"].iloc[0]

    out["event_family"] = "unknown"
    out["event_type_raw_std"] = "unknown"

    if market == "SZ" and table == "order":
        out["event_family"] = "order"
        out["event_type_raw_std"] = "order_add_like"

    elif market == "SZ" and table == "trade":
        if "ExecType" not in out.columns:
            raise ValueError("SZ trade missing ExecType")

        exec_str = out["ExecType"].astype(str)

        out.loc[exec_str == "70", "event_family"] = "trade"
        out.loc[exec_str == "70", "event_type_raw_std"] = "trade_like"

        out.loc[exec_str == "52", "event_family"] = "cancel"
        out.loc[exec_str == "52", "event_type_raw_std"] = "cancel_like"

        unknown_mask = ~exec_str.isin(["70", "52"])
        out.loc[unknown_mask, "event_family"] = "unknown"
        out.loc[unknown_mask, "event_type_raw_std"] = "unknown_exec_type"

    elif market == "SH" and table == "order":
        if "OrderType" not in out.columns:
            raise ValueError("SH order missing OrderType")

        order_type = out["OrderType"].astype(str).str.strip()

        out.loc[order_type == "A", "event_family"] = "order"
        out.loc[order_type == "A", "event_type_raw_std"] = "order_add_like"

        out.loc[order_type == "D", "event_family"] = "cancel"
        out.loc[order_type == "D", "event_type_raw_std"] = "order_cancel_like"

        unknown_mask = ~order_type.isin(["A", "D"])
        out.loc[unknown_mask, "event_family"] = "unknown"
        out.loc[unknown_mask, "event_type_raw_std"] = "unknown_order_type"

    elif market == "SH" and table == "trade":
        out["event_family"] = "trade"
        out["event_type_raw_std"] = "trade_like"

    else:
        raise ValueError(f"Unhandled combination: market={market}, table={table}")

    return out


def inspect_raw_standardization(trade_date: str | int, symbol: str, table: str, n=10):
    path = RAW_ROOT / trade_date / table / f"{symbol}.ftr"

    print("\n" + "=" * 120)
    print(f"RAW {table.upper()} | SYMBOL={symbol}")
    print(f"PATH={path}")
    print("=" * 120)

    df = pd.read_feather(path)
    std_df = add_standard_fields(df, symbol, table)
    out = add_event_type_fields(std_df)

    print("[before shape]", df.shape)
    print("[after  shape]", out.shape)

    print("\n[event type counts]")
    print(out[["event_family", "event_type_raw_std"]].value_counts(dropna=False))

    preview_cols = [
        c for c in [
            "symbol", "market", "source_table",
            "event_time_str", "event_time_ms",
            "order_id", "buy_order_id", "sell_order_id",
            "trade_price", "trade_qty",
            "ExecType", "OrderType",
            "event_family", "event_type_raw_std"
        ] if c in out.columns
    ]

    print("\n[preview]")
    print(out[preview_cols].head(n))

    if table == "trade" and "ExecType" in out.columns:
        print("\n[ExecType counts]")
        print(out["ExecType"].astype(str).value_counts(dropna=False).head(20))

    if table == "order" and "OrderType" in out.columns:
        print("\n[OrderType counts]")
        print(out["OrderType"].astype(str).value_counts(dropna=False).head(20))
    
def normalize_side_raw_std(series, market: str, table: str):
    """
    统一原始方向到:
    B / S / N / UNK

    说明：
    - order/cancel: 尽量从原始订单方向字段直接标准化
    - trade:
        SH 用 TradeBSFlag 原样标准化（通常为 N）
        SZ 当前没有可靠单边方向字段，统一给 UNK
    """
    s = series.astype(str).str.strip()

    if market == "SZ" and table == "order":
        # 深市 Side: 49=买, 50=卖
        out = pd.Series("UNK", index=series.index, dtype="object")
        out.loc[s == "49"] = "B"
        out.loc[s == "50"] = "S"
        return out

    if market == "SH" and table == "order":
        # 沪市 OrderBSFlag: B/S
        out = pd.Series("UNK", index=series.index, dtype="object")
        out.loc[s == "B"] = "B"
        out.loc[s == "S"] = "S"
        return out

    if market == "SH" and table == "trade":
        # 沪市 clean/raw trade 目前样本多为 N，不强推主动方向
        out = pd.Series("UNK", index=series.index, dtype="object")
        out.loc[s == "B"] = "B"
        out.loc[s == "S"] = "S"
        out.loc[s == "N"] = "N"
        return out

    if market == "SZ" and table == "trade":
        # 当前不伪造方向
        return pd.Series("UNK", index=series.index, dtype="object")

    return pd.Series("UNK", index=series.index, dtype="object")


# =============================================================================
# Part B. event_stream_v1
# =============================================================================
def get_order_id_set(trade_date: str | int, symbol: str):
    market = infer_market_from_symbol(symbol)
    order = load_raw(trade_date, symbol, "order")

    if market == "SZ":
        return set(order["ApplSeqNum"].dropna().astype(int).tolist())
    elif market == "SH":
        return set(order["OrderNO"].dropna().astype(int).tolist())
    else:
        raise ValueError(market)


def build_order_events(trade_date: str | int, symbol: str):
    market = infer_market_from_symbol(symbol)
    df = load_raw(trade_date, symbol, "order").copy()

    if market == "SZ":
        df["event_time_str"] = df["TransactTime"].astype(str)
        df["event_time_ms"] = df["event_time_str"].map(hhmmss_to_ms)

        df["order_id"] = df["ApplSeqNum"]
        df["event_family"] = "order"
        df["event_type_raw_std"] = "order_add_like"

        # 新增：统一最小数值锚
        df["event_price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["event_qty"] = pd.to_numeric(df["OrderQty"], errors="coerce")

        # 新增：方向标准化
        df["event_side_raw_std"] = normalize_side_raw_std(df["Side"], market="SZ", table="order")

        # 新增：排序代理（深市当前没有可靠等价字段）
        df["event_seq_proxy"] = pd.NA

    elif market == "SH":
        df["event_time_str"] = df["OrderTime"].astype(str)
        df["event_time_ms"] = df["event_time_str"].map(hhmmss_to_ms)

        df["order_id"] = df["OrderNO"]

        order_type = df["OrderType"].astype(str).str.strip()
        df["event_family"] = "unknown"
        df["event_type_raw_std"] = "unknown"

        df.loc[order_type == "A", "event_family"] = "order"
        df.loc[order_type == "A", "event_type_raw_std"] = "order_add_like"

        df.loc[order_type == "D", "event_family"] = "cancel"
        df.loc[order_type == "D", "event_type_raw_std"] = "order_cancel_like"

        # 新增：统一最小数值锚
        # 注意：SH 的 Balance 在 A/D 行都作为当前事件行对应的数量锚使用
        df["event_price"] = pd.to_numeric(df["OrderPrice"], errors="coerce")
        df["event_qty"] = pd.to_numeric(df["Balance"], errors="coerce")

        # 新增：方向标准化
        df["event_side_raw_std"] = normalize_side_raw_std(df["OrderBSFlag"], market="SH", table="order")

        # 新增：排序代理
        df["event_seq_proxy"] = pd.to_numeric(df["BizIndex"], errors="coerce")

    else:
        raise ValueError(market)

    df["symbol"] = symbol
    df["market"] = market
    df["source_table"] = "order"

    df["buy_order_id"] = pd.NA
    df["sell_order_id"] = pd.NA
    df["trade_price"] = pd.NA
    df["trade_qty"] = pd.NA

    df["is_direct_observation"] = 1
    df["is_inferred_from_trade"] = 0
    df["buy_order_seen_in_order"] = pd.NA
    df["sell_order_seen_in_order"] = pd.NA
    df["has_inferred_side"] = 0

    keep_cols = [
        "symbol", "market", "source_table",
        "event_time_str", "event_time_ms",
        "event_family", "event_type_raw_std",

        # 新增的统一字段
        "event_price", "event_qty", "event_side_raw_std", "event_seq_proxy",

        "order_id", "buy_order_id", "sell_order_id",
        "trade_price", "trade_qty",
        "is_direct_observation", "is_inferred_from_trade",
        "buy_order_seen_in_order", "sell_order_seen_in_order",
        "has_inferred_side"
    ]
    return df[keep_cols].copy()


def build_trade_events(trade_date: str | int, symbol: str):
    market = infer_market_from_symbol(symbol)
    df = load_raw(trade_date, symbol, "trade").copy()
    order_ids = get_order_id_set(trade_date, symbol)

    if market == "SZ":
        df["event_time_str"] = df["TransactTime"].astype(str)
        df["event_time_ms"] = df["event_time_str"].map(hhmmss_to_ms)

        df["order_id"] = pd.NA
        df["buy_order_id"] = df["BidApplSeqNum"]
        df["sell_order_id"] = df["OfferApplSeqNum"]
        df["trade_price"] = df["LastPx"]
        df["trade_qty"] = df["LastQty"]

        exec_str = df["ExecType"].astype(str)
        df["event_family"] = "unknown"
        df["event_type_raw_std"] = "unknown"

        df.loc[exec_str == "70", "event_family"] = "trade"
        df.loc[exec_str == "70", "event_type_raw_std"] = "trade_like"

        df.loc[exec_str == "52", "event_family"] = "cancel"
        df.loc[exec_str == "52", "event_type_raw_std"] = "cancel_like"

        # 新增：统一数值锚
        # 对 trade/cancel 都先保留原始行上的 price/qty
        df["event_price"] = pd.to_numeric(df["LastPx"], errors="coerce")
        df["event_qty"] = pd.to_numeric(df["LastQty"], errors="coerce")

        # 新增：方向标准化（当前不伪造）
        df["event_side_raw_std"] = normalize_side_raw_std(
            pd.Series(index=df.index, data=pd.NA), market="SZ", table="trade"
        )

        # 新增：排序代理（当前无可靠等价字段）
        df["event_seq_proxy"] = pd.NA

    elif market == "SH":
        df["event_time_str"] = df["TradTime"].astype(str)
        df["event_time_ms"] = df["event_time_str"].map(hhmmss_to_ms)

        df["order_id"] = pd.NA
        df["buy_order_id"] = df["TradeBuyNo"]
        df["sell_order_id"] = df["TradeSellNo"]
        df["trade_price"] = df["TradPrice"]
        df["trade_qty"] = df["TradVolume"]

        df["event_family"] = "trade"
        df["event_type_raw_std"] = "trade_like"

        # 新增：统一数值锚
        df["event_price"] = pd.to_numeric(df["TradPrice"], errors="coerce")
        df["event_qty"] = pd.to_numeric(df["TradVolume"], errors="coerce")

        # 新增：方向标准化（保守保留 N/B/S/UNK）
        df["event_side_raw_std"] = normalize_side_raw_std(df["TradeBSFlag"], market="SH", table="trade")

        # 新增：trade 当前没有可靠排序代理
        df["event_seq_proxy"] = pd.NA

    else:
        raise ValueError(market)

    df["symbol"] = symbol
    df["market"] = market
    df["source_table"] = "trade"

    df["is_direct_observation"] = 1

    def seen(x):
        if pd.isna(x):
            return pd.NA
        try:
            x = int(x)
        except Exception:
            return pd.NA
        if market == "SZ" and x == 0:
            return 0
        return 1 if x in order_ids else 0

    df["buy_order_seen_in_order"] = df["buy_order_id"].map(seen)
    df["sell_order_seen_in_order"] = df["sell_order_id"].map(seen)

    df["has_inferred_side"] = (
        (df["buy_order_seen_in_order"] == 0) | (df["sell_order_seen_in_order"] == 0)
    ).astype(int)

    df["is_inferred_from_trade"] = df["has_inferred_side"]

    keep_cols = [
        "symbol", "market", "source_table",
        "event_time_str", "event_time_ms",
        "event_family", "event_type_raw_std",

        # 新增的统一字段
        "event_price", "event_qty", "event_side_raw_std", "event_seq_proxy",

        "order_id", "buy_order_id", "sell_order_id",
        "trade_price", "trade_qty",
        "is_direct_observation", "is_inferred_from_trade",
        "buy_order_seen_in_order", "sell_order_seen_in_order",
        "has_inferred_side"
    ]
    return df[keep_cols].copy()


def build_event_stream_preview(trade_date: str | int, symbol: str):
    order_events = build_order_events(trade_date, symbol)
    trade_events = build_trade_events(trade_date, symbol)

    out = pd.concat([order_events, trade_events], axis=0, ignore_index=True)

    # 注意：这里只做轻排序，不宣称最终真顺序
    # SZ mixed-source 同毫秒不能硬排
    # SH order 内部后续可借助 BizIndex 做细排序
    out = out.sort_values(
        by=["event_time_ms", "source_table"],
        ascending=[True, True],
        kind="stable"
    ).reset_index(drop=True)

    return out


def inspect_event_stream(trade_date: str | int, symbol: str, n=20):
    es = build_event_stream_preview(trade_date=trade_date, symbol=symbol)

    print("\n" + "=" * 120)
    print(f"EVENT_STREAM PREVIEW | SYMBOL={symbol}")
    print("=" * 120)

    print("[shape]")
    print(es.shape)

    print("\n[event counts]")
    print(es[["source_table", "event_family", "event_type_raw_std"]].value_counts(dropna=False))

    print("\n[inferred-side stats]")
    print(es["has_inferred_side"].value_counts(dropna=False))

    print("\n[preview head]")
    print(es.head(n))

    print("\n[preview tail]")
    print(es.tail(n))


# =============================================================================
# Part C. order-trade link audit
# =============================================================================
def audit_order_trade_link_sz(trade_date: str | int, symbol: str):
    order = load_raw(trade_date, symbol, "order")
    trade = load_raw(trade_date, symbol, "trade")

    print("\n" + "=" * 120)
    print(f"SZ ORDER-TRADE LINK AUDIT | SYMBOL={symbol}")
    print("=" * 120)

    order_ids = set(order["ApplSeqNum"].dropna().astype(int).tolist())

    bid_ids_all = set(trade["BidApplSeqNum"].dropna().astype(int).tolist())
    offer_ids_all = set(trade["OfferApplSeqNum"].dropna().astype(int).tolist())

    bid_ids_nonzero = {x for x in bid_ids_all if x != 0}
    offer_ids_nonzero = {x for x in offer_ids_all if x != 0}

    bid_hit = len(bid_ids_nonzero & order_ids)
    offer_hit = len(offer_ids_nonzero & order_ids)

    print("\n[all trade rows]")
    print("order unique ids:", len(order_ids))
    print("trade unique bid ids (all):", len(bid_ids_all))
    print("trade unique offer ids (all):", len(offer_ids_all))
    print("trade unique bid ids (nonzero):", len(bid_ids_nonzero))
    print("trade unique offer ids (nonzero):", len(offer_ids_nonzero))

    print_ratio("bid id in order", bid_hit, len(bid_ids_nonzero))
    print_ratio("offer id in order", offer_hit, len(offer_ids_nonzero))

    for exec_type in ["70", "52"]:
        sub = trade[trade["ExecType"].astype(str) == exec_type].copy()

        bid_ids = set(sub["BidApplSeqNum"].dropna().astype(int).tolist())
        offer_ids = set(sub["OfferApplSeqNum"].dropna().astype(int).tolist())

        bid_ids_nonzero = {x for x in bid_ids if x != 0}
        offer_ids_nonzero = {x for x in offer_ids if x != 0}

        bid_hit = len(bid_ids_nonzero & order_ids)
        offer_hit = len(offer_ids_nonzero & order_ids)

        print(f"\n[ExecType={exec_type}]")
        print("rows:", len(sub))
        print("unique bid ids (nonzero):", len(bid_ids_nonzero))
        print("unique offer ids (nonzero):", len(offer_ids_nonzero))
        print_ratio("bid id in order", bid_hit, len(bid_ids_nonzero))
        print_ratio("offer id in order", offer_hit, len(offer_ids_nonzero))

        bid_miss = sorted(list(bid_ids_nonzero - order_ids))[:20]
        offer_miss = sorted(list(offer_ids_nonzero - order_ids))[:20]
        print("sample bid misses:", bid_miss)
        print("sample offer misses:", offer_miss)


def audit_order_trade_link_sh(trade_date: str | int, symbol: str):
    order = load_raw(trade_date, symbol, "order")
    trade = load_raw(trade_date, symbol, "trade")

    print("\n" + "=" * 120)
    print(f"SH ORDER-TRADE LINK AUDIT | SYMBOL={symbol}")
    print("=" * 120)

    # 只用 A 订单做 order 直接可观测集合更合理
    order_ids = set(order.loc[order["OrderType"].astype(str).str.strip() == "A", "OrderNO"].dropna().astype(int).tolist())

    buy_ids = set(trade["TradeBuyNo"].dropna().astype(int).tolist())
    sell_ids = set(trade["TradeSellNo"].dropna().astype(int).tolist())

    buy_hit = len(buy_ids & order_ids)
    sell_hit = len(sell_ids & order_ids)

    print("\n[all trade rows]")
    print("order unique ids (A only):", len(order_ids))
    print("trade unique buy ids:", len(buy_ids))
    print("trade unique sell ids:", len(sell_ids))

    print_ratio("buy id in order", buy_hit, len(buy_ids))
    print_ratio("sell id in order", sell_hit, len(sell_ids))

    buy_miss = sorted(list(buy_ids - order_ids))[:30]
    sell_miss = sorted(list(sell_ids - order_ids))[:30]

    print("\n[sample buy misses]")
    print(buy_miss)

    print("\n[sample sell misses]")
    print(sell_miss)

    trade["buy_in_order"] = trade["TradeBuyNo"].astype(int).isin(order_ids)
    trade["sell_in_order"] = trade["TradeSellNo"].astype(int).isin(order_ids)

    print("\n[trade row-level link status]")
    both_hit = ((trade["buy_in_order"]) & (trade["sell_in_order"])).sum()
    buy_only = ((trade["buy_in_order"]) & (~trade["sell_in_order"])).sum()
    sell_only = ((~trade["buy_in_order"]) & (trade["sell_in_order"])).sum()
    both_miss = ((~trade["buy_in_order"]) & (~trade["sell_in_order"])).sum()

    print("both_hit :", int(both_hit))
    print("buy_only :", int(buy_only))
    print("sell_only:", int(sell_only))
    print("both_miss:", int(both_miss))

    print("\n[sample both_miss rows]")
    cols = [c for c in [
        "TradeIndex", "TradTime", "TradPrice", "TradVolume",
        "TradeBuyNo", "TradeSellNo", "TradeBSFlag"
    ] if c in trade.columns]
    print(trade.loc[(~trade["buy_in_order"]) & (~trade["sell_in_order"]), cols].head(20))


def audit_order_trade_link(trade_date: str | int, symbol: str):
    market = infer_market_from_symbol(symbol)
    if market == "SZ":
        audit_order_trade_link_sz(trade_date, symbol)
    else:
        audit_order_trade_link_sh(trade_date, symbol)


# =============================================================================
# Part D. event ordering audit
# =============================================================================
def build_min_event_stream_for_ordering(trade_date: str | int, symbol: str):
    market = infer_market_from_symbol(symbol)

    order = load_raw(trade_date, symbol, "order").copy()
    if market == "SZ":
        order["event_time_str"] = order["TransactTime"].astype(str)
        order["event_time_ms"] = order["event_time_str"].map(hhmmss_to_ms)
        order["event_family"] = "order"
        order["event_type_raw_std"] = "order_add_like"
        order["biz_index"] = pd.NA
    else:
        order["event_time_str"] = order["OrderTime"].astype(str)
        order["event_time_ms"] = order["event_time_str"].map(hhmmss_to_ms)
        order_type = order["OrderType"].astype(str).str.strip()
        order["event_family"] = "unknown"
        order["event_type_raw_std"] = "unknown"
        order.loc[order_type == "A", "event_family"] = "order"
        order.loc[order_type == "A", "event_type_raw_std"] = "order_add_like"
        order.loc[order_type == "D", "event_family"] = "cancel"
        order.loc[order_type == "D", "event_type_raw_std"] = "order_cancel_like"
        order["biz_index"] = order["BizIndex"]

    order["symbol"] = symbol
    order["market"] = market
    order["source_table"] = "order"
    order = order[[
        "symbol", "market", "source_table",
        "event_time_str", "event_time_ms",
        "event_family", "event_type_raw_std",
        "biz_index"
    ]].copy()

    trade = load_raw(trade_date, symbol, "trade").copy()
    if market == "SZ":
        trade["event_time_str"] = trade["TransactTime"].astype(str)
        trade["event_time_ms"] = trade["event_time_str"].map(hhmmss_to_ms)
        exec_str = trade["ExecType"].astype(str)
        trade["event_family"] = "unknown"
        trade["event_type_raw_std"] = "unknown"
        trade.loc[exec_str == "70", "event_family"] = "trade"
        trade.loc[exec_str == "70", "event_type_raw_std"] = "trade_like"
        trade.loc[exec_str == "52", "event_family"] = "cancel"
        trade.loc[exec_str == "52", "event_type_raw_std"] = "cancel_like"
        trade["biz_index"] = pd.NA
    else:
        trade["event_time_str"] = trade["TradTime"].astype(str)
        trade["event_time_ms"] = trade["event_time_str"].map(hhmmss_to_ms)
        trade["event_family"] = "trade"
        trade["event_type_raw_std"] = "trade_like"
        trade["biz_index"] = pd.NA

    trade["symbol"] = symbol
    trade["market"] = market
    trade["source_table"] = "trade"
    trade = trade[[
        "symbol", "market", "source_table",
        "event_time_str", "event_time_ms",
        "event_family", "event_type_raw_std",
        "biz_index"
    ]].copy()

    return pd.concat([order, trade], ignore_index=True)


def audit_event_ordering(trade_date: str | int, symbol: str):
    es = build_min_event_stream_for_ordering(trade_date, symbol)
    market = infer_market_from_symbol(symbol)

    print("\n" + "=" * 120)
    print(f"EVENT ORDERING AUDIT | SYMBOL={symbol} | MARKET={market}")
    print("=" * 120)

    per_ms = es.groupby("event_time_ms").size().rename("n_events")
    print("\n[events per ms summary]")
    print(per_ms.describe())

    print("\n[top 20 busiest ms]")
    print(per_ms.sort_values(ascending=False).head(20))

    by_ms_source = es.groupby(["event_time_ms", "source_table"]).size().unstack(fill_value=0)
    if "order" not in by_ms_source.columns:
        by_ms_source["order"] = 0
    if "trade" not in by_ms_source.columns:
        by_ms_source["trade"] = 0

    mixed = by_ms_source[(by_ms_source["order"] > 0) & (by_ms_source["trade"] > 0)]
    order_only = by_ms_source[(by_ms_source["order"] > 0) & (by_ms_source["trade"] == 0)]
    trade_only = by_ms_source[(by_ms_source["order"] == 0) & (by_ms_source["trade"] > 0)]

    print("\n[millisecond source-mix counts]")
    print("mixed(order+trade):", len(mixed))
    print("order_only        :", len(order_only))
    print("trade_only        :", len(trade_only))

    print("\n[sample mixed ms rows]")
    print(mixed.head(20))

    print("\n[sample mixed ms event details]")
    for ms in mixed.head(5).index.tolist():
        print(f"\n--- event_time_ms = {ms} ---")
        sub = es[es["event_time_ms"] == ms].copy()
        print(sub.sort_values(["source_table", "event_family", "event_type_raw_std"]).head(30))

    if market == "SH":
        sh_order = load_raw(symbol, "order").copy()
        sh_order["event_time_ms"] = sh_order["OrderTime"].astype(str).map(hhmmss_to_ms)

        order_per_ms = sh_order.groupby("event_time_ms").size().rename("n_order").sort_values(ascending=False)
        print("\n[SH order per ms top 20]")
        print(order_per_ms.head(20))

        multi_ms = order_per_ms[order_per_ms > 1].index.tolist()[:10]

        print("\n[SH same-ms order rows with BizIndex]")
        for ms in multi_ms:
            print(f"\n--- SH order event_time_ms = {ms} ---")
            sub = sh_order[sh_order["event_time_ms"] == ms].copy().sort_values("BizIndex")
            cols = [c for c in ["OrderTime", "OrderType", "OrderNO", "OrderPrice", "Balance", "OrderBSFlag", "BizIndex"] if c in sub.columns]
            print(sub[cols].head(30))

            biz = sub["BizIndex"].dropna().tolist()
            print("BizIndex unique count:", len(set(biz)), " / rows:", len(biz))
            is_strict_inc = all(biz[i] < biz[i + 1] for i in range(len(biz) - 1)) if len(biz) > 1 else True
            print("BizIndex strictly increasing after sort:", is_strict_inc)


# =============================================================================
# Part E. snapshot_context_v1
# =============================================================================
def build_snapshot_context_v2(trade_date: str | int, symbol: str):
    market = infer_market_from_symbol(symbol)
    raw = load_raw(trade_date, symbol, "tick").copy()

    df = raw.copy()
    df["symbol"] = symbol
    df["market"] = market
    df["source_table"] = "tick"
    df["event_time_str"] = df["UpdateTime"].astype(str)
    df["event_time_ms"] = df["event_time_str"].map(hhmmss_to_ms)

    if market == "SZ":
        df["preclose"] = df["PreCloPrice"]
        df["open"] = df["OpenPrice"]
        df["high"] = df["HighPrice"]
        df["low"] = df["LowPrice"]
        df["last"] = df["LastPrice"]
        df["cum_trade_count"] = df["TurnNum"]
        df["cum_volume"] = df["Volume"]
        df["cum_amount"] = df["Turnover"]
    elif market == "SH":
        df["preclose"] = df["PreCloPrice"]
        df["open"] = df["OpenPrice"]
        df["high"] = df["HighPrice"]
        df["low"] = df["LowPrice"]
        df["last"] = df["LastPrice"]
        df["cum_trade_count"] = df["TradNumber"]
        df["cum_volume"] = df["TradVolume"]
        df["cum_amount"] = df["Turnover"]
    else:
        raise ValueError(market)

    df["ask1_price"] = pd.to_numeric(df["AskPrice1"], errors="coerce")
    df["ask1_qty"] = pd.to_numeric(df["AskVolume1"], errors="coerce")
    df["bid1_price"] = pd.to_numeric(df["BidPrice1"], errors="coerce")
    df["bid1_qty"] = pd.to_numeric(df["BidVolume1"], errors="coerce")

    phase_candidates = ["TradingPhaseCode", "InstruStatus"]
    found_phase_col = None
    for c in phase_candidates:
        if c in df.columns:
            found_phase_col = c
            break

    if found_phase_col is not None:
        df["trading_phase_raw"] = df[found_phase_col].astype(str)
    else:
        df["trading_phase_raw"] = pd.NA

    def nz(series):
        return pd.to_numeric(series, errors="coerce").fillna(0)

    has_book = df["bid1_price"].notna() | df["ask1_price"].notna()
    has_trade_activity = (nz(df["last"]) > 0) | (nz(df["cum_volume"]) > 0) | (nz(df["cum_amount"]) > 0)
    df["has_core_state"] = (has_book | has_trade_activity).astype(int)

    t = df["event_time_str"]
    df["is_post_close_snapshot"] = (t >= "15:00:00").astype(int)

    # 当前项目保守定义：连续竞价上下文只取 09:30:00 <= t < 14:57:00
    df["is_continuous_context_by_time"] = ((t >= "09:30:00") & (t < "14:57:00")).astype(int)

    df["is_usable_snapshot_context"] = (
        (df["is_post_close_snapshot"] == 0)
        & (df["is_continuous_context_by_time"] == 1)
        & (df["has_core_state"] == 1)
    ).astype(int)

    df["is_raw_snapshot"] = 1

    keep_cols = [
        "symbol", "market", "source_table",
        "event_time_str", "event_time_ms",
        "trading_phase_raw",
        "preclose", "open", "high", "low", "last",
        "cum_trade_count", "cum_volume", "cum_amount",
        "bid1_price", "bid1_qty", "ask1_price", "ask1_qty",
        "is_raw_snapshot",
        "has_core_state",
        "is_post_close_snapshot",
        "is_continuous_context_by_time",
        "is_usable_snapshot_context",
    ]
    return raw, df[keep_cols].copy()


def inspect_snapshot_context(trade_date: str | int, symbol: str, n=20):
    raw, out = build_snapshot_context_v2(trade_date, symbol)

    print("\n" + "=" * 120)
    print(f"SNAPSHOT_CONTEXT PREVIEW V2 | SYMBOL={symbol}")
    print("=" * 120)

    print("[before shape]")
    print(raw.shape)

    print("\n[after shape]")
    print(out.shape)

    print("\n[flag counts]")
    for c in [
        "has_core_state",
        "is_post_close_snapshot",
        "is_continuous_context_by_time",
        "is_usable_snapshot_context",
    ]:
        print(f"\n{c}")
        print(out[c].value_counts(dropna=False))

    print("\n[trading_phase_raw sample counts]")
    print(out["trading_phase_raw"].value_counts(dropna=False).head(20))

    print("\n[preview head]")
    print(out.head(n))

    print("\n[preview tail]")
    print(out.tail(n))

    print("\n[post-close preview]")
    print(out[out["is_post_close_snapshot"] == 1].head(20))

    print("\n[usable snapshot preview]")
    print(out[out["is_usable_snapshot_context"] == 1].head(20))


# =============================================================================
# Part F. tick around close audit
# =============================================================================
def build_tick_minimal(trade_date: str | int, symbol: str) -> pd.DataFrame:
    market = infer_market_from_symbol(symbol)
    raw = load_raw(trade_date, symbol, "tick").copy()

    out = raw.copy()
    out["symbol"] = symbol
    out["market"] = market
    out["event_time_str"] = out["UpdateTime"].astype(str)

    if market == "SZ":
        out["preclose"] = out["PreCloPrice"]
        out["open"] = out["OpenPrice"]
        out["high"] = out["HighPrice"]
        out["low"] = out["LowPrice"]
        out["last"] = out["LastPrice"]
        out["cum_trade_count"] = out["TurnNum"]
        out["cum_volume"] = out["Volume"]
        out["cum_amount"] = out["Turnover"]
    else:
        out["preclose"] = out["PreCloPrice"]
        out["open"] = out["OpenPrice"]
        out["high"] = out["HighPrice"]
        out["low"] = out["LowPrice"]
        out["last"] = out["LastPrice"]
        out["cum_trade_count"] = out["TradNumber"]
        out["cum_volume"] = out["TradVolume"]
        out["cum_amount"] = out["Turnover"]

    out["bid1_price"] = pd.to_numeric(out["BidPrice1"], errors="coerce")
    out["bid1_qty"] = pd.to_numeric(out["BidVolume1"], errors="coerce")
    out["ask1_price"] = pd.to_numeric(out["AskPrice1"], errors="coerce")
    out["ask1_qty"] = pd.to_numeric(out["AskVolume1"], errors="coerce")

    cols = [
        "symbol", "market", "event_time_str",
        "preclose", "open", "high", "low", "last",
        "cum_trade_count", "cum_volume", "cum_amount",
        "bid1_price", "bid1_qty", "ask1_price", "ask1_qty"
    ]
    return out[cols].copy()


def audit_tick_from_145950(trade_date: str | int, symbol: str, n: int = 60):
    df = build_tick_minimal(trade_date, symbol)

    print("\n" + "=" * 120)
    print(f"TICK FROM 14:59:50 | SYMBOL={symbol} | MARKET={infer_market_from_symbol(symbol)}")
    print("=" * 120)

    sub = df[df["event_time_str"] >= "14:59:50"].copy()

    print("\n[rows from 14:59:50 onward]")
    print(len(sub))

    print(f"\n[first {n} rows from 14:59:50]")
    print(sub.head(n))

    print("\n[time counts from 14:59:50 onward]")
    print(sub["event_time_str"].value_counts().sort_index())

    state_cols = [
        "last", "cum_trade_count", "cum_volume", "cum_amount",
        "bid1_price", "bid1_qty", "ask1_price", "ask1_qty"
    ]
    distinct = sub[["event_time_str"] + state_cols].drop_duplicates()

    print("\n[distinct rows from 14:59:50 onward]")
    print(len(distinct))
    print(distinct)

    small = df[(df["event_time_str"] >= "14:59:50") & (df["event_time_str"] <= "15:00:10")].copy()
    print("\n[rows between 14:59:50 and 15:00:10]")
    print(len(small))
    print(small)


# =============================================================================
# Part G. ex post boundary / order_lifecycle_expost_final_v1
# =============================================================================
def order_field_role_map():
    return {
        "OrderNO": "id_anchor",
        "SecurityID": "id_anchor",

        "OrderTimer": "expost_lifecycle_core",
        "FirstTradTimer": "expost_lifecycle_core",
        "LastTradTimer": "expost_lifecycle_core",
        "OrderFinishTimer": "expost_lifecycle_core",

        "OrderType": "order_static_attr",
        "PoN": "order_static_attr",
        "OrderPrice": "order_static_attr",
        "OrderSize": "order_static_attr",
        "OrderBSFlag": "order_static_attr",

        "ExecPrice": "expost_trade_outcome",
        "ExecVwap": "expost_trade_outcome",
        "TradeRate": "expost_trade_outcome",
        "TradPrice": "expost_trade_outcome",
        "Balance": "expost_trade_outcome",
        "PosRate": "expost_trade_outcome",
    }


def trade_field_role_map():
    return {
        "SecurityID": "id_anchor",

        "TradTime": "expost_trade_fact",
        "TradTimer": "expost_trade_fact",
        "TradPrice": "expost_trade_fact",
        "TradVolume": "expost_trade_fact",
        "TradeMoney": "expost_trade_fact",

        "TradeBuyNo": "counterparty_order_ref",
        "TradeSellNo": "counterparty_order_ref",

        "TradeBSFlag": "annotation_only",
    }


def weak_supervision_candidates_order():
    return {
        "FirstTradTimer",
        "LastTradTimer",
        "OrderFinishTimer",
        "ExecPrice",
        "ExecVwap",
        "TradeRate",
        "TradPrice",
        "Balance",
        "PosRate",
    }


def weak_supervision_candidates_trade():
    return {
        "TradPrice",
        "TradVolume",
        "TradeMoney",
        "TradeBSFlag",
    }


def forbidden_for_online_order():
    return {
        "FirstTradTimer",
        "LastTradTimer",
        "OrderFinishTimer",
        "ExecPrice",
        "ExecVwap",
        "TradeRate",
        "TradPrice",
        "Balance",
        "PosRate",
    }


def forbidden_for_online_trade():
    return {
        "TradTime",
        "TradTimer",
        "TradPrice",
        "TradVolume",
        "TradeMoney",
        "TradeBuyNo",
        "TradeSellNo",
        "TradeBSFlag",
    }


def build_boundary_table(df: pd.DataFrame, table: str):
    if table == "order":
        role_map = order_field_role_map()
        weak_set = weak_supervision_candidates_order()
        forbid_set = forbidden_for_online_order()
    elif table == "trade":
        role_map = trade_field_role_map()
        weak_set = weak_supervision_candidates_trade()
        forbid_set = forbidden_for_online_trade()
    else:
        raise ValueError(table)

    rows = []
    for col in df.columns:
        rows.append({
            "field_name": col,
            "field_role": role_map.get(col, "unclassified"),
            "is_weak_supervision_candidate": int(col in weak_set),
            "forbidden_for_online": int(col in forbid_set),
            "dtype": str(df[col].dtype),
            "null_ratio": float(df[col].isna().mean()),
        })

    out = pd.DataFrame(rows).sort_values(
        by=["forbidden_for_online", "field_role", "field_name"],
        ascending=[False, True, True],
        kind="stable"
    ).reset_index(drop=True)

    return out


def inspect_expost_boundary(symbol: str, n=50):
    market = infer_market_from_symbol(symbol)
    clean_order = load_clean(symbol, "order")
    clean_trade = load_clean(symbol, "trade")

    order_boundary = build_boundary_table(clean_order, "order")
    trade_boundary = build_boundary_table(clean_trade, "trade")

    print("\n" + "=" * 120)
    print(f"EXPOST BOUNDARY PREVIEW | SYMBOL={symbol} | MARKET={market}")
    print("=" * 120)

    print("\n[CLEAN ORDER shape]")
    print(clean_order.shape)

    print("\n[CLEAN TRADE shape]")
    print(clean_trade.shape)

    print("\n[ORDER boundary role counts]")
    print(order_boundary["field_role"].value_counts(dropna=False))

    print("\n[ORDER forbidden_for_online counts]")
    print(order_boundary["forbidden_for_online"].value_counts(dropna=False))

    print("\n[ORDER boundary table]")
    print(order_boundary.head(n))

    print("\n[TRADE boundary role counts]")
    print(trade_boundary["field_role"].value_counts(dropna=False))

    print("\n[TRADE forbidden_for_online counts]")
    print(trade_boundary["forbidden_for_online"].value_counts(dropna=False))

    print("\n[TRADE boundary table]")
    print(trade_boundary.head(n))


# =============================================================================
# Part H. 一键入口
# =============================================================================
def run_all_for_symbol(symbol: str):
    print("\n" + "#" * 120)
    print(f"RUN STEP0 FINAL PIPELINE FOR SYMBOL={symbol}")
    print("#" * 120)

    for table in ["order", "trade"]:
        inspect_raw_standardization(symbol, table, n=10)

    inspect_event_stream(symbol, n=20)
    audit_order_trade_link(symbol)
    audit_event_ordering(symbol)
    inspect_snapshot_context(symbol, n=20)
    audit_tick_from_145950(symbol, n=60)
    inspect_expost_boundary(symbol, n=50)


def run_all_default_samples():
    for symbol in DEFAULT_SAMPLES.values():
        run_all_for_symbol(symbol)
