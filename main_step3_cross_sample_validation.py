# -*- coding: utf-8 -*-
"""
Step3: cross-sample validation
目标：
1) 不重新训练 Step2 模型
2) 对多个 (trade_date, symbol) 样本复用冻结 Step2 protocol
3) 输出 baseline / vanilla25 / time-aware25 的统一比较结果

当前第一版策略：
- 先做 smoke test: n_samples = 50
- 之后流程稳定再扩到 1000
"""

from __future__ import annotations

import json
import random
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import os

import io
import contextlib


# =============================================================================
# 路径配置（你先按你自己的真实路径改）
# =============================================================================
STEP3_ROOT = Path("/home/bu-yuting/新建文件夹/step3_cross_sample_validation")

# baseline 参考目录：用于 learned-vs-baseline matching
FINAL_BASELINE_DIR = Path(
    "/home/bu-yuting/新建文件夹/step2_outputs_validate_baseline_v2/final_baseline_no_delta_depth_total"
)

# vanilla25 已训练模型目录
VANILLA25_MODEL_DIR = Path(
    "/home/bu-yuting/新建文件夹/step2_outputs_validate_baseline_v2/learned_small_transformer_v1_epoch25"
)

# time-aware25 已训练模型目录
TIMEAWARE25_MODEL_DIR = Path(
    "/home/bu-yuting/新建文件夹/step2_outputs_validate_baseline_v2/learned_timeaware_transformer_a_v1_seed7"
)

# Step3 如果要缓存单样本 Step1 / Step2 中间产物，统一放这里
STEP3_ARTIFACT_ROOT = STEP3_ROOT / "sample_artifacts"

FINAL_BASELINE_FEATURE_CSV = FINAL_BASELINE_DIR / "final_feature_model_with_cluster.csv"
FINAL_BASELINE_PROFILE_CSV = FINAL_BASELINE_DIR / "final_cluster_profile.csv"

KNOWN_STEP1_ARTIFACT_LOOKUP = {
    "000001": "/home/bu-yuting/新建文件夹/行为单元/step1_outputs_cached_000001",
}

KNOWN_STEP2_ARTIFACT_LOOKUP = {
    "000001": "/home/bu-yuting/新建文件夹/step2_outputs_validate_baseline_v2",
}


# =============================================================================
# Step3 冻结协议
# =============================================================================
GLOBAL_RANDOM_STATE = 42
SMOKE_TEST_N_SAMPLES = 50

BASELINE_PROTOCOL = {
    "drop_feature_cols": ["delta_depth_total"],
    "pca_n_components": 8,
    "n_clusters": 8,
    "random_state": 42,
    "n_init": 1,
    "max_iter": 30,
    "silhouette_sample_size": 2000,
}

SMOKE_TEST_SILHOUETTE_SAMPLE_SIZE = 500

STABILITY_PROTOCOL = {
    "profile_corr_threshold": 0.85,
    "jaccard_threshold": 0.60,
}

RECENT_WINDOW_MONTHS = 12

UNIVERSE_TARGET_MAX_TRADE_DATES = 120
UNIVERSE_TARGET_MAX_SYMBOLS_PER_DATE = 80
UNIVERSE_TARGET_MAX_ROWS = 9600

STEP3_TARGET_N_SAMPLES = 1000

STEP3_FULL_SAMPLE_LIST_CSV = "/home/bu-yuting/新建文件夹/行为单元/step3_sample_list_recent1y_1000.csv"
STEP3_AUDITED_FULL_CSV = "/home/bu-yuting/新建文件夹/行为单元/step3_sample_list_1000_audited.csv"
STEP3_BUILT_FULL_CSV = "/home/bu-yuting/新建文件夹/行为单元/step3_sample_list_1000_with_artifacts.csv"

STEP3_BATCH_OUT_CSV = "/home/bu-yuting/新建文件夹/step3_cross_sample_validation/recent1y_1000_results.csv"
STEP3_BATCH_OUT_JSON = "/home/bu-yuting/新建文件夹/step3_cross_sample_validation/recent1y_1000_summary.json"

# =============================================================================
# 小工具函数
# =============================================================================
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int = GLOBAL_RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_sample_tag(trade_date: str | int, symbol: str) -> str:
    """
    统一 sample 命名，例如:
    20170321_000001
    """
    return f"{str(trade_date)}_{str(symbol).zfill(6)}"


def get_sample_output_dir(trade_date: str | int, symbol: str) -> Path:
    """
    每个 sample 单独一个目录，避免产物混在一起。
    """
    sample_tag = make_sample_tag(trade_date, symbol)
    return ensure_dir(STEP3_ARTIFACT_ROOT / sample_tag)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None
    

# =============================================================================
# 导入你现有工程中的 Step1 / Step2 函数
# 这里如果函数名和你本地文件不完全一样，你运行时报错后把报错贴我，我再按你的真实接口改
# =============================================================================
from step1 import build_and_save_step1_final_artifact, load_step1_final_artifact

from step2 import (
    build_event_stream_step2_ready,
    build_episode_candidates,
    build_episode_feature_table,
    build_and_save_step2_artifacts,
    load_step2_artifacts,
    run_episode_baseline_kmeans,
    SmallEpisodeTransformerMLM,
    SmallEpisodeTimeAwareTransformerMLM,
    extract_episode_embeddings,
    extract_timeaware_episode_embeddings,
    export_step2_learned_embedding_cluster_audit,
    export_learned_vs_baseline_profile_matching,
)

def run_step1_for_one_sample(
    trade_date: str | int,
    symbol: str,
    sample_out_dir: Path,
    rebuild_step1: bool = False,
    existing_step1_artifact_dir: str | Path | None = None,
) -> Dict[str, Any]:
    default_step1_out_dir = ensure_dir(sample_out_dir / "step1_artifact")

    result = {
        "status": "success",
        "step1_out_dir": str(default_step1_out_dir),
        "trade_date": str(trade_date),
        "symbol": str(symbol).zfill(6),
        "used_cached_step1": False,
        "used_external_step1_artifact": False,
    }

    try:
        if existing_step1_artifact_dir is not None:
            existing_step1_artifact_dir = Path(existing_step1_artifact_dir)

            # 如果用户误传了 pkl 文件，就自动取父目录
            if existing_step1_artifact_dir.is_file():
                existing_step1_artifact_dir = existing_step1_artifact_dir.parent

            artifact = load_step1_final_artifact(existing_step1_artifact_dir)

            result["step1_out_dir"] = str(existing_step1_artifact_dir)
            result["used_cached_step1"] = True
            result["used_external_step1_artifact"] = True
            result["artifact_keys"] = list(artifact.keys()) if isinstance(artifact, dict) else None
            return result

        if not rebuild_step1:
            try:
                artifact = load_step1_final_artifact(default_step1_out_dir)
                if artifact is not None:
                    result["used_cached_step1"] = True
                    result["artifact_keys"] = list(artifact.keys()) if isinstance(artifact, dict) else None
                    return result
            except Exception:
                pass

        build_and_save_step1_final_artifact(
            trade_date=str(trade_date),
            symbol=str(symbol).zfill(6),
            out_dir=default_step1_out_dir,
        )

        artifact = load_step1_final_artifact(default_step1_out_dir)
        result["artifact_keys"] = list(artifact.keys()) if isinstance(artifact, dict) else None

    except KeyboardInterrupt:
        result["status"] = "failed"
        result["fail_stage"] = "step1"
        result["fail_reason"] = "keyboard_interrupt_during_step1_build"
        result["traceback"] = traceback.format_exc()
        return result

    except Exception as e:
        result["status"] = "failed"
        result["fail_stage"] = "step1"
        result["fail_reason"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result

def build_step2_inputs_for_one_sample(
    trade_date: str | int,
    symbol: str,
    step1_out_dir: Path,
    existing_step2_artifact_dir: str | Path | None = None,
) -> Dict[str, Any]:
    result = {
        "status": "success",
        "trade_date": str(trade_date),
        "symbol": str(symbol).zfill(6),
        "step1_out_dir": str(step1_out_dir),
        "used_external_step2_artifact": False,
    }

    try:
        if existing_step2_artifact_dir is not None:
            existing_step2_artifact_dir = Path(existing_step2_artifact_dir)
            es, episode_table, episode_feature_table = load_step2_artifacts(existing_step2_artifact_dir)

            result.update({
                "used_external_step2_artifact": True,
                "step2_artifact_dir": str(existing_step2_artifact_dir),
                "n_events": safe_int(len(es) if es is not None else 0),
                "n_episodes": safe_int(len(episode_table) if episode_table is not None else 0),
                "event_stream_step2_ready": es,
                "episode_table": episode_table,
                "episode_feature_table": episode_feature_table,
            })
            return result

        # 新增：把现算结果直接保存成 Step2 artifact
        step2_artifact_dir = ensure_dir(get_sample_output_dir(trade_date, symbol) / "step2_artifact")

        es, episode_table, episode_feature_table = build_and_save_step2_artifacts(
            symbol=str(symbol).zfill(6),
            out_dir=str(step2_artifact_dir),
            keep_sessions=["continuous_am", "continuous_pm"],
            gap_hard_ms=3000,
            lag_hard_ms=5000,
            max_events_per_episode=64,
            max_duration_ms=10000,
            min_events_soft_cut=8,
            min_events_episode_flag=4,
            step1_artifact_dir=str(step1_out_dir),
        )

        result.update({
            "step2_artifact_dir": str(step2_artifact_dir),
            "n_events": safe_int(len(es) if es is not None else 0),
            "n_episodes": safe_int(len(episode_table) if episode_table is not None else 0),
            "event_stream_step2_ready": es,
            "episode_table": episode_table,
            "episode_feature_table": episode_feature_table,
        })

        if result["n_events"] is None or result["n_events"] <= 0:
            result["status"] = "failed"
            result["fail_stage"] = "step2_ready"
            result["fail_reason"] = "no events after step2-ready construction"
            return result

        if result["n_episodes"] is None or result["n_episodes"] <= 0:
            result["status"] = "failed"
            result["fail_stage"] = "episode_build"
            result["fail_reason"] = "no episodes generated"
            return result

        if episode_feature_table is None or len(episode_feature_table) <= 0:
            result["status"] = "failed"
            result["fail_stage"] = "feature_table"
            result["fail_reason"] = "empty episode_feature_table"
            return result

    except Exception as e:
        result["status"] = "failed"
        result["fail_stage"] = "step2_ready"
        result["fail_reason"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result

def run_baseline_for_one_sample(
    trade_date: str | int,
    symbol: str,
    step2_res: Dict[str, Any],
    sample_out_dir: Path,
) -> Dict[str, Any]:
    """
    对单个 sample 运行冻结版 baseline clustering。
    如果当前 sample 直接复用的是冻结版 Step2 artifact 目录，则优先直接复用 baseline 指标。
    """
    result = {
        "status": "success",
        "trade_date": str(trade_date),
        "symbol": str(symbol).zfill(6),
    }

    try:
        if step2_res.get("status") != "success":
            result["status"] = "failed"
            result["fail_stage"] = "baseline"
            result["fail_reason"] = "step2_res is not success"
            return result

        step2_artifact_dir = step2_res.get("step2_artifact_dir")
        if not step2_artifact_dir:
            result["status"] = "failed"
            result["fail_stage"] = "baseline"
            result["fail_reason"] = "step2_artifact_dir missing in step2_res"
            return result

        baseline_out_dir = ensure_dir(sample_out_dir / "baseline")

        # 1) 优先复用冻结版 baseline 指标
        frozen_summary = get_frozen_baseline_summary_for_known_artifact_dir(step2_artifact_dir)
        if frozen_summary is not None:
            result.update({
                "baseline_out_dir": str(baseline_out_dir),
                "baseline_source_step2_artifact_dir": str(step2_artifact_dir),
                "baseline_source": frozen_summary["baseline_source"],
                "baseline_silhouette": frozen_summary["baseline_silhouette"],
                "baseline_db": frozen_summary["baseline_db"],
                "baseline_feature_csv": str(FINAL_BASELINE_FEATURE_CSV),
                "baseline_profile_csv": str(FINAL_BASELINE_PROFILE_CSV),
            })
            save_json(result, baseline_out_dir / "baseline_summary.json")
            return result

        # 2) 否则才真正重跑 baseline
        res = run_episode_baseline_kmeans(
            out_dir=str(step2_artifact_dir),
            clip_quantile_low=0.01,
            clip_quantile_high=0.99,
            pca_n_components=BASELINE_PROTOCOL["pca_n_components"],
            kmeans_k=BASELINE_PROTOCOL["n_clusters"],
            random_state=BASELINE_PROTOCOL["random_state"],
            kmeans_n_init=BASELINE_PROTOCOL["n_init"],
            kmeans_max_iter=BASELINE_PROTOCOL["max_iter"],
            silhouette_sample_size=SMOKE_TEST_SILHOUETTE_SAMPLE_SIZE,
        )

        metrics = res.get("metrics", {}) if isinstance(res, dict) else {}

        result["baseline_out_dir"] = str(baseline_out_dir)
        result["baseline_source_step2_artifact_dir"] = str(step2_artifact_dir)
        result["baseline_source"] = "rerun"
        result["baseline_pkg_keys"] = list(res.keys()) if isinstance(res, dict) else None
        result["baseline_silhouette"] = safe_float(metrics.get("silhouette_score"))
        result["baseline_db"] = safe_float(metrics.get("davies_bouldin_score"))

        feature_model = res["feature_model"]
        cluster_profile = res["cluster_profile"]

        baseline_feature_csv = baseline_out_dir / "final_feature_model_with_cluster.csv"
        baseline_profile_csv = baseline_out_dir / "final_cluster_profile.csv"

        feature_model.to_csv(baseline_feature_csv, index=False)
        cluster_profile.to_csv(baseline_profile_csv, index=False)

        result["baseline_feature_csv"] = str(baseline_feature_csv)
        result["baseline_profile_csv"] = str(baseline_profile_csv)

        save_json(result, baseline_out_dir / "baseline_summary.json")

    except Exception as e:
        result["status"] = "failed"
        result["fail_stage"] = "baseline"
        result["fail_reason"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result

def get_frozen_baseline_summary_for_known_artifact_dir(
    step2_artifact_dir: str | Path,
) -> Dict[str, Any] | None:
    """
    如果传入的是当前冻结版 baseline 的 Step2 artifact 目录，
    直接返回已知 baseline 指标，不再重跑 baseline clustering。
    """
    step2_artifact_dir = str(Path(step2_artifact_dir).resolve())
    frozen_dir = str(Path("/home/bu-yuting/新建文件夹/step2_outputs_validate_baseline_v2").resolve())

    if step2_artifact_dir != frozen_dir:
        return None

    return {
        "baseline_source": "frozen_reuse",
        "baseline_silhouette": 0.15686923265457153,
        "baseline_db": 1.5479367547658727,
    }

def load_vanilla25_model_bundle(device: str = "cpu") -> Dict[str, Any]:
    """
    加载已经训练好的 vanilla25 模型、vocab、config。
    不做训练，只做 inference。
    """
    model_file = VANILLA25_MODEL_DIR / "small_transformer_mlm_state_dict.pt"
    vocab_file = VANILLA25_MODEL_DIR / "vocab.json"
    config_file = VANILLA25_MODEL_DIR / "train_config.json"

    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab_obj = json.load(f)

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    token_to_id = vocab_obj["token_to_id"]
    id_to_token = {int(k): v for k, v in vocab_obj["id_to_token"].items()}

    pad_token_id = token_to_id["[PAD]"]

    model = SmallEpisodeTransformerMLM(
        vocab_size=len(token_to_id),
        cont_dim=4,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        max_len=cfg["max_len"],
        pad_token_id=pad_token_id,
    )

    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "cfg": cfg,
        "model_file": str(model_file),
        "vocab_file": str(vocab_file),
        "config_file": str(config_file),
    }

def load_timeaware25_model_bundle(device: str = "cpu") -> Dict[str, Any]:
    model_file = TIMEAWARE25_MODEL_DIR / "timeaware_transformer_mlm_state_dict.pt"
    vocab_file = TIMEAWARE25_MODEL_DIR / "vocab.json"
    config_file = TIMEAWARE25_MODEL_DIR / "train_config.json"

    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab_obj = json.load(f)

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    token_to_id = vocab_obj["token_to_id"]
    id_to_token = {int(k): v for k, v in vocab_obj["id_to_token"].items()}
    pad_token_id = token_to_id["[PAD]"]

    model = SmallEpisodeTimeAwareTransformerMLM(
        vocab_size=len(token_to_id),
        cont_dim=4,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        max_len=cfg["max_len"],
        pad_token_id=pad_token_id,
    )

    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "cfg": cfg,
        "model_file": str(model_file),
        "vocab_file": str(vocab_file),
        "config_file": str(config_file),
    }

def run_vanilla25_for_one_sample(
    trade_date: str | int,
    symbol: str,
    step2_res: Dict[str, Any],
    sample_out_dir: Path,
    baseline_feature_csv: str | Path,
    baseline_profile_csv: str | Path,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    对单个 sample 运行 vanilla25:
    1) load trained model
    2) extract embeddings
    3) clustering
    4) matching to final baseline
    """
    result = {
        "status": "success",
        "trade_date": str(trade_date),
        "symbol": str(symbol).zfill(6),
    }

    try:
        if step2_res.get("status") != "success":
            result["status"] = "failed"
            result["fail_stage"] = "vanilla25"
            result["fail_reason"] = "step2_res is not success"
            return result

        step2_artifact_dir = step2_res.get("step2_artifact_dir")
        if not step2_artifact_dir:
            result["status"] = "failed"
            result["fail_stage"] = "vanilla25"
            result["fail_reason"] = "step2_artifact_dir missing in step2_res"
            return result

        vanilla_out_dir = ensure_dir(sample_out_dir / "vanilla25")

        bundle = load_vanilla25_model_bundle(device=device)
        model = bundle["model"]
        token_to_id = bundle["token_to_id"]
        cfg = bundle["cfg"]

        # 1) embedding extraction
        embedding_df = extract_episode_embeddings(
            model=model,
            out_dir=str(step2_artifact_dir),
            token_to_id=token_to_id,
            token_min_freq=cfg["token_min_freq"],
            max_seq_len=cfg["max_seq_len"],
            batch_size=cfg.get("extract_batch_size", 128),
            device=device,
        )

        embedding_df = embedding_df.sort_values("episode_id").reset_index(drop=True)

        embedding_csv = vanilla_out_dir / "episode_embedding_small_transformer_v1.csv"
        embedding_pkl = vanilla_out_dir / "episode_embedding_small_transformer_v1.pkl"
        embedding_meta_file = vanilla_out_dir / "embedding_meta.json"

        embedding_df.to_csv(embedding_csv, index=False)
        embedding_df.to_pickle(embedding_pkl)

        emb_meta = {
            "n_rows": int(embedding_df.shape[0]),
            "n_cols": int(embedding_df.shape[1]),
            "n_embedding_dims": int(embedding_df.shape[1] - 1),
            "episode_id_min": int(embedding_df["episode_id"].min()),
            "episode_id_max": int(embedding_df["episode_id"].max()),
        }
        save_json(emb_meta, embedding_meta_file)

        # 2) clustering
        cluster_run_name = "vanilla25_cluster"

        cluster_res = export_step2_learned_embedding_cluster_audit(
            embedding_csv=str(embedding_csv),
            baseline_feature_csv=str(baseline_feature_csv),
            out_dir=str(vanilla_out_dir),
            run_name=cluster_run_name,
            pca_n_components=BASELINE_PROTOCOL["pca_n_components"],
            kmeans_k=BASELINE_PROTOCOL["n_clusters"],
            random_state=BASELINE_PROTOCOL["random_state"],
            kmeans_n_init=BASELINE_PROTOCOL["n_init"],
            kmeans_max_iter=BASELINE_PROTOCOL["max_iter"],
            silhouette_sample_size=SMOKE_TEST_SILHOUETTE_SAMPLE_SIZE,
        )

        # 3) matching
        matching_run_name = "vanilla25_vs_final_baseline_profile_matching"

        match_res = export_learned_vs_baseline_profile_matching(
            learned_profile_csv=cluster_res["profile_file"],
            baseline_profile_csv=str(baseline_profile_csv),
            out_dir=str(vanilla_out_dir),
            run_name=matching_run_name,
        )

        metrics = cluster_res.get("metrics", {})
        match_summary = match_res.get("summary", {})

        result.update({
            "vanilla25_out_dir": str(vanilla_out_dir),
            "vanilla25_source_step2_artifact_dir": str(step2_artifact_dir),
            "vanilla25_embedding_csv": str(embedding_csv),
            "vanilla25_cluster_metrics": metrics,
            "vanilla25_matching_summary": match_summary,
            "vanilla25_silhouette": safe_float(metrics.get("silhouette_score")),
            "vanilla25_db": safe_float(metrics.get("davies_bouldin_score")),
            "vanilla25_mean_best_corr_to_baseline": safe_float(
                match_summary.get("mean_best_corr_learned_to_baseline")
            ),
            "vanilla25_min_best_corr_to_baseline": safe_float(
                match_summary.get("min_best_corr_learned_to_baseline")
            ),
            "vanilla25_max_best_corr_to_baseline": safe_float(
                match_summary.get("max_best_corr_learned_to_baseline")
            ),
        })

        save_json(result, vanilla_out_dir / "vanilla25_summary.json")

    except Exception as e:
        result["status"] = "failed"
        result["fail_stage"] = "vanilla25"
        result["fail_reason"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result

def run_timeaware25_for_one_sample(
    trade_date: str | int,
    symbol: str,
    step2_res: Dict[str, Any],
    sample_out_dir: Path,
    baseline_feature_csv: str | Path,
    baseline_profile_csv: str | Path,
    device: str = "cpu",
) -> Dict[str, Any]:
    result = {
        "status": "success",
        "trade_date": str(trade_date),
        "symbol": str(symbol).zfill(6),
    }

    try:
        if step2_res.get("status") != "success":
            result["status"] = "failed"
            result["fail_stage"] = "timeaware25"
            result["fail_reason"] = "step2_res is not success"
            return result

        step2_artifact_dir = step2_res.get("step2_artifact_dir")
        if not step2_artifact_dir:
            result["status"] = "failed"
            result["fail_stage"] = "timeaware25"
            result["fail_reason"] = "step2_artifact_dir missing in step2_res"
            return result

        ta_out_dir = ensure_dir(sample_out_dir / "timeaware25")

        bundle = load_timeaware25_model_bundle(device=device)
        model = bundle["model"]
        token_to_id = bundle["token_to_id"]
        cfg = bundle["cfg"]

        embedding_df = extract_timeaware_episode_embeddings(
            model=model,
            out_dir=str(step2_artifact_dir),
            token_to_id=token_to_id,
            token_min_freq=cfg["token_min_freq"],
            max_seq_len=cfg["max_seq_len"],
            batch_size=cfg.get("extract_batch_size", 128),
            device=device,
        )

        embedding_df = embedding_df.sort_values("episode_id").reset_index(drop=True)

        embedding_csv = ta_out_dir / "episode_embedding_timeaware_transformer_v1.csv"
        embedding_pkl = ta_out_dir / "episode_embedding_timeaware_transformer_v1.pkl"
        embedding_meta_file = ta_out_dir / "embedding_meta.json"

        embedding_df.to_csv(embedding_csv, index=False)
        embedding_df.to_pickle(embedding_pkl)

        emb_meta = {
            "n_rows": int(embedding_df.shape[0]),
            "n_cols": int(embedding_df.shape[1]),
            "n_embedding_dims": int(embedding_df.shape[1] - 1),
            "episode_id_min": int(embedding_df["episode_id"].min()),
            "episode_id_max": int(embedding_df["episode_id"].max()),
        }
        save_json(emb_meta, embedding_meta_file)

        cluster_run_name = "timeaware25_cluster"
        cluster_res = export_step2_learned_embedding_cluster_audit(
            embedding_csv=str(embedding_csv),
            baseline_feature_csv=str(baseline_feature_csv),
            out_dir=str(ta_out_dir),
            run_name=cluster_run_name,
            pca_n_components=BASELINE_PROTOCOL["pca_n_components"],
            kmeans_k=BASELINE_PROTOCOL["n_clusters"],
            random_state=BASELINE_PROTOCOL["random_state"],
            kmeans_n_init=BASELINE_PROTOCOL["n_init"],
            kmeans_max_iter=BASELINE_PROTOCOL["max_iter"],
            silhouette_sample_size=SMOKE_TEST_SILHOUETTE_SAMPLE_SIZE,
        )

        matching_run_name = "timeaware25_vs_final_baseline_profile_matching"
        match_res = export_learned_vs_baseline_profile_matching(
            learned_profile_csv=cluster_res["profile_file"],
            baseline_profile_csv=str(baseline_profile_csv),
            out_dir=str(ta_out_dir),
            run_name=matching_run_name,
        )

        metrics = cluster_res.get("metrics", {})
        match_summary = match_res.get("summary", {})

        result.update({
            "timeaware25_out_dir": str(ta_out_dir),
            "timeaware25_source_step2_artifact_dir": str(step2_artifact_dir),
            "timeaware25_embedding_csv": str(embedding_csv),
            "timeaware25_cluster_metrics": metrics,
            "timeaware25_matching_summary": match_summary,
            "timeaware25_silhouette": safe_float(metrics.get("silhouette_score")),
            "timeaware25_db": safe_float(metrics.get("davies_bouldin_score")),
            "timeaware25_mean_best_corr_to_baseline": safe_float(
                match_summary.get("mean_best_corr_learned_to_baseline")
            ),
            "timeaware25_min_best_corr_to_baseline": safe_float(
                match_summary.get("min_best_corr_learned_to_baseline")
            ),
            "timeaware25_max_best_corr_to_baseline": safe_float(
                match_summary.get("max_best_corr_learned_to_baseline")
            ),
        })

        save_json(result, ta_out_dir / "timeaware25_summary.json")

    except Exception as e:
        result["status"] = "failed"
        result["fail_stage"] = "timeaware25"
        result["fail_reason"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result

# def smoke_test_step1_step2_baseline_vanilla25_timeaware25(
#     trade_date: str | int,
#     symbol: str,
#     rebuild_step1: bool = False,
#     existing_step1_artifact_dir: str | Path | None = None,
#     existing_step2_artifact_dir: str | Path | None = None,
#     device: str = "cpu",
# ) -> Dict[str, Any]:
#     sample_out_dir = get_sample_output_dir(trade_date, symbol)

#     step1_res = run_step1_for_one_sample(
#         trade_date=trade_date,
#         symbol=symbol,
#         sample_out_dir=sample_out_dir,
#         rebuild_step1=rebuild_step1,
#         existing_step1_artifact_dir=existing_step1_artifact_dir,
#     )
#     if step1_res["status"] != "success":
#         return step1_res

#     step2_res = build_step2_inputs_for_one_sample(
#         trade_date=trade_date,
#         symbol=symbol,
#         step1_out_dir=Path(step1_res["step1_out_dir"]),
#         existing_step2_artifact_dir=existing_step2_artifact_dir,
#     )
#     if step2_res["status"] != "success":
#         return step2_res

#     baseline_res = run_baseline_for_one_sample(
#         trade_date=trade_date,
#         symbol=symbol,
#         step2_res=step2_res,
#         sample_out_dir=sample_out_dir,
#     )
#     if baseline_res["status"] != "success":
#         return baseline_res

#     vanilla_res = run_vanilla25_for_one_sample(
#         trade_date=trade_date,
#         symbol=symbol,
#         step2_res=step2_res,
#         sample_out_dir=sample_out_dir,
#         baseline_feature_csv=baseline_res["baseline_feature_csv"],
#         baseline_profile_csv=baseline_res["baseline_profile_csv"],
#         device=device,
#     )
#     if vanilla_res["status"] != "success":
#         return vanilla_res

#     timeaware_res = run_timeaware25_for_one_sample(
#         trade_date=trade_date,
#         symbol=symbol,
#         step2_res=step2_res,
#         sample_out_dir=sample_out_dir,
#         baseline_feature_csv=baseline_res["baseline_feature_csv"],
#         baseline_profile_csv=baseline_res["baseline_profile_csv"],
#         device=device,
#     )
#     if timeaware_res["status"] != "success":
#         return timeaware_res

#     summary = {
#         "status": "success",
#         "trade_date": str(trade_date),
#         "symbol": str(symbol).zfill(6),
#         "sample_out_dir": str(sample_out_dir),
#         "step1_out_dir": step1_res["step1_out_dir"],
#         "used_cached_step1": step1_res.get("used_cached_step1", False),
#         "used_external_step1_artifact": step1_res.get("used_external_step1_artifact", False),
#         "used_external_step2_artifact": step2_res.get("used_external_step2_artifact", False),
#         "n_events": step2_res.get("n_events"),
#         "n_episodes": step2_res.get("n_episodes"),
#         "baseline_silhouette": baseline_res.get("baseline_silhouette"),
#         "baseline_db": baseline_res.get("baseline_db"),
#         "vanilla25_silhouette": vanilla_res.get("vanilla25_silhouette"),
#         "vanilla25_db": vanilla_res.get("vanilla25_db"),
#         "vanilla25_mean_best_corr_to_baseline": vanilla_res.get("vanilla25_mean_best_corr_to_baseline"),
#         "timeaware25_silhouette": timeaware_res.get("timeaware25_silhouette"),
#         "timeaware25_db": timeaware_res.get("timeaware25_db"),
#         "timeaware25_mean_best_corr_to_baseline": timeaware_res.get("timeaware25_mean_best_corr_to_baseline"),
#     }

#     save_json(summary, sample_out_dir / "smoke_test_step1_step2_baseline_vanilla25_timeaware25_summary.json")
#     return summary

# def smoke_test_step1_step2_baseline_vanilla25(
#     trade_date: str | int,
#     symbol: str,
#     rebuild_step1: bool = False,
#     existing_step1_artifact_dir: str | Path | None = None,
#     existing_step2_artifact_dir: str | Path | None = None,
#     device: str = "cpu",
# ) -> Dict[str, Any]:
#     sample_out_dir = get_sample_output_dir(trade_date, symbol)

#     step1_res = run_step1_for_one_sample(
#         trade_date=trade_date,
#         symbol=symbol,
#         sample_out_dir=sample_out_dir,
#         rebuild_step1=rebuild_step1,
#         existing_step1_artifact_dir=existing_step1_artifact_dir,
#     )
#     if step1_res["status"] != "success":
#         return step1_res

#     step2_res = build_step2_inputs_for_one_sample(
#         trade_date=trade_date,
#         symbol=symbol,
#         step1_out_dir=Path(step1_res["step1_out_dir"]),
#         existing_step2_artifact_dir=existing_step2_artifact_dir,
#     )
#     if step2_res["status"] != "success":
#         return step2_res

#     baseline_res = run_baseline_for_one_sample(
#         trade_date=trade_date,
#         symbol=symbol,
#         step2_res=step2_res,
#         sample_out_dir=sample_out_dir,
#     )
#     if baseline_res["status"] != "success":
#         return baseline_res

#     vanilla_res = run_vanilla25_for_one_sample(
#         trade_date=trade_date,
#         symbol=symbol,
#         step2_res=step2_res,
#         sample_out_dir=sample_out_dir,
#         baseline_feature_csv=FINAL_BASELINE_FEATURE_CSV,
#         baseline_profile_csv=FINAL_BASELINE_PROFILE_CSV,
#         device=device,
#     )
#     if vanilla_res["status"] != "success":
#         return vanilla_res

#     summary = {
#         "status": "success",
#         "trade_date": str(trade_date),
#         "symbol": str(symbol).zfill(6),
#         "sample_out_dir": str(sample_out_dir),
#         "step1_out_dir": step1_res["step1_out_dir"],
#         "used_cached_step1": step1_res.get("used_cached_step1", False),
#         "used_external_step1_artifact": step1_res.get("used_external_step1_artifact", False),
#         "used_external_step2_artifact": step2_res.get("used_external_step2_artifact", False),
#         "n_events": step2_res.get("n_events"),
#         "n_episodes": step2_res.get("n_episodes"),
#         "baseline_silhouette": baseline_res.get("baseline_silhouette"),
#         "baseline_db": baseline_res.get("baseline_db"),
#         "vanilla25_silhouette": vanilla_res.get("vanilla25_silhouette"),
#         "vanilla25_db": vanilla_res.get("vanilla25_db"),
#         "vanilla25_mean_best_corr_to_baseline": vanilla_res.get("vanilla25_mean_best_corr_to_baseline"),
#     }

#     save_json(summary, sample_out_dir / "smoke_test_step1_step2_baseline_vanilla25_summary.json")
#     return summary

# def smoke_test_step1_step2_baseline(
#     trade_date: str | int,
#     symbol: str,
#     rebuild_step1: bool = False,
#     existing_step1_artifact_dir: str | Path | None = None,
#     existing_step2_artifact_dir: str | Path | None = None,
# ) -> Dict[str, Any]:
#     sample_out_dir = get_sample_output_dir(trade_date, symbol)

#     step1_res = run_step1_for_one_sample(
#         trade_date=trade_date,
#         symbol=symbol,
#         sample_out_dir=sample_out_dir,
#         rebuild_step1=rebuild_step1,
#         existing_step1_artifact_dir=existing_step1_artifact_dir,
#     )
#     if step1_res["status"] != "success":
#         return step1_res

#     step2_res = build_step2_inputs_for_one_sample(
#         trade_date=trade_date,
#         symbol=symbol,
#         step1_out_dir=Path(step1_res["step1_out_dir"]),
#         existing_step2_artifact_dir=existing_step2_artifact_dir,
#     )
#     if step2_res["status"] != "success":
#         return step2_res

#     baseline_res = run_baseline_for_one_sample(
#         trade_date=trade_date,
#         symbol=symbol,
#         step2_res=step2_res,
#         sample_out_dir=sample_out_dir,
#     )
#     if baseline_res["status"] != "success":
#         return baseline_res

#     summary = {
#         "status": "success",
#         "trade_date": str(trade_date),
#         "symbol": str(symbol).zfill(6),
#         "sample_out_dir": str(sample_out_dir),
#         "step1_out_dir": step1_res["step1_out_dir"],
#         "used_cached_step1": step1_res.get("used_cached_step1", False),
#         "used_external_step1_artifact": step1_res.get("used_external_step1_artifact", False),
#         "used_external_step2_artifact": step2_res.get("used_external_step2_artifact", False),
#         "n_events": step2_res.get("n_events"),
#         "n_episodes": step2_res.get("n_episodes"),
#         "episode_feature_table_shape": (
#             list(step2_res["episode_feature_table"].shape)
#             if step2_res.get("episode_feature_table") is not None
#             else None
#         ),
#         "baseline_silhouette": baseline_res.get("baseline_silhouette"),
#         "baseline_db": baseline_res.get("baseline_db"),
#     }

#     save_json(summary, sample_out_dir / "smoke_test_step1_step2_baseline_summary.json")
#     return summary

# def smoke_test_step1_step2(
#     trade_date: str | int,
#     symbol: str,
#     rebuild_step1: bool = False,
#     existing_step1_artifact_dir: str | Path | None = None,
#     existing_step2_artifact_dir: str | Path | None = None,
# ) -> Dict[str, Any]:
#     sample_out_dir = get_sample_output_dir(trade_date, symbol)

#     step1_res = run_step1_for_one_sample(
#         trade_date=trade_date,
#         symbol=symbol,
#         sample_out_dir=sample_out_dir,
#         rebuild_step1=rebuild_step1,
#         existing_step1_artifact_dir=existing_step1_artifact_dir,
#     )
#     if step1_res["status"] != "success":
#         return step1_res

#     step2_res = build_step2_inputs_for_one_sample(
#         trade_date=trade_date,
#         symbol=symbol,
#         step1_out_dir=Path(step1_res["step1_out_dir"]),
#         existing_step2_artifact_dir=existing_step2_artifact_dir,
#     )
#     if step2_res["status"] != "success":
#         return step2_res

#     summary = {
#         "status": "success",
#         "trade_date": str(trade_date),
#         "symbol": str(symbol).zfill(6),
#         "sample_out_dir": str(sample_out_dir),
#         "step1_out_dir": step1_res["step1_out_dir"],
#         "used_cached_step1": step1_res.get("used_cached_step1", False),
#         "used_external_step1_artifact": step1_res.get("used_external_step1_artifact", False),
#         "n_events": step2_res.get("n_events"),
#         "n_episodes": step2_res.get("n_episodes"),
#         "episode_feature_table_shape": (
#             list(step2_res["episode_feature_table"].shape)
#             if step2_res.get("episode_feature_table") is not None
#             else None
#         ),
#     }
#     save_json(summary, sample_out_dir / "smoke_test_step1_step2_summary.json")
#     return summary

def run_step3_one_sample(
    trade_date: str | int,
    symbol: str,
    rebuild_step1: bool = False,
    existing_step1_artifact_dir: str | Path | None = None,
    existing_step2_artifact_dir: str | Path | None = None,
    baseline_feature_csv: str | Path = FINAL_BASELINE_FEATURE_CSV,
    baseline_profile_csv: str | Path = FINAL_BASELINE_PROFILE_CSV,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Step3 单样本正式 runner。

    当前版本目标：
    1) 复用 Step1 artifact（若提供）
    2) 复用 Step2 artifact（若提供）
    3) baseline / vanilla25 / time-aware25 全链条跑通
    4) 产出统一 sample-level summary

    注意：
    - 当前 smoke 版仍允许 baseline_feature_csv / baseline_profile_csv 指向 frozen baseline
    - 正式 cross-sample 版时，这两个参数应替换为“当前 sample 自己的 baseline 产物”
    """
    sample_out_dir = get_sample_output_dir(trade_date, symbol)

    summary = {
        "status": "success",
        "trade_date": str(trade_date),
        "symbol": str(symbol).zfill(6),
        "sample_out_dir": str(sample_out_dir),
    }

    try:
        # ---------------------------------------------------------------------
        # Step1
        # ---------------------------------------------------------------------
        with contextlib.redirect_stdout(io.StringIO()):
            step1_res = run_step1_for_one_sample(
                trade_date=trade_date,
                symbol=symbol,
                sample_out_dir=sample_out_dir,
                rebuild_step1=rebuild_step1,
                existing_step1_artifact_dir=existing_step1_artifact_dir,
            )
        if step1_res["status"] != "success":
            return step1_res

        # ---------------------------------------------------------------------
        # Step2
        # ---------------------------------------------------------------------
        with contextlib.redirect_stdout(io.StringIO()):
            step2_res = build_step2_inputs_for_one_sample(
                trade_date=trade_date,
                symbol=symbol,
                step1_out_dir=Path(step1_res["step1_out_dir"]),
                existing_step2_artifact_dir=existing_step2_artifact_dir,
            )
        if step2_res["status"] != "success":
            return step2_res

        # ---------------------------------------------------------------------
        # baseline
        # ---------------------------------------------------------------------
        with contextlib.redirect_stdout(io.StringIO()):
            baseline_res = run_baseline_for_one_sample(
                trade_date=trade_date,
                symbol=symbol,
                step2_res=step2_res,
                sample_out_dir=sample_out_dir,
            )
        if baseline_res["status"] != "success":
            return baseline_res

        # ---------------------------------------------------------------------
        # vanilla25
        # ---------------------------------------------------------------------
        with contextlib.redirect_stdout(io.StringIO()):
            vanilla_res = run_vanilla25_for_one_sample(
                trade_date=trade_date,
                symbol=symbol,
                step2_res=step2_res,
                sample_out_dir=sample_out_dir,
                baseline_feature_csv=baseline_res["baseline_feature_csv"],
                baseline_profile_csv=baseline_res["baseline_profile_csv"],
                device=device,
            )
        if vanilla_res["status"] != "success":
            return vanilla_res

        # ---------------------------------------------------------------------
        # time-aware25
        # ---------------------------------------------------------------------
        with contextlib.redirect_stdout(io.StringIO()):
            timeaware_res = run_timeaware25_for_one_sample(
                trade_date=trade_date,
                symbol=symbol,
                step2_res=step2_res,
                sample_out_dir=sample_out_dir,
                baseline_feature_csv=baseline_res["baseline_feature_csv"],
                baseline_profile_csv=baseline_res["baseline_profile_csv"],
                device=device,
            )
        if timeaware_res["status"] != "success":
            return timeaware_res

        # ---------------------------------------------------------------------
        # unified summary
        # ---------------------------------------------------------------------
        summary.update({
            "step1_out_dir": step1_res["step1_out_dir"],
            "used_cached_step1": step1_res.get("used_cached_step1", False),
            "used_external_step1_artifact": step1_res.get("used_external_step1_artifact", False),

            "step2_artifact_dir": step2_res.get("step2_artifact_dir"),
            "used_external_step2_artifact": step2_res.get("used_external_step2_artifact", False),

            "n_events": step2_res.get("n_events"),
            "n_episodes": step2_res.get("n_episodes"),
            "episode_feature_table_shape": (
                list(step2_res["episode_feature_table"].shape)
                if step2_res.get("episode_feature_table") is not None
                and hasattr(step2_res["episode_feature_table"], "shape")
                else None
            ),

            "baseline_silhouette": baseline_res.get("baseline_silhouette"),
            "baseline_db": baseline_res.get("baseline_db"),
            "baseline_source": baseline_res.get("baseline_source"),

            "vanilla25_silhouette": vanilla_res.get("vanilla25_silhouette"),
            "vanilla25_db": vanilla_res.get("vanilla25_db"),
            "vanilla25_mean_best_corr_to_baseline": vanilla_res.get("vanilla25_mean_best_corr_to_baseline"),
            "vanilla25_min_best_corr_to_baseline": vanilla_res.get("vanilla25_min_best_corr_to_baseline"),
            "vanilla25_max_best_corr_to_baseline": vanilla_res.get("vanilla25_max_best_corr_to_baseline"),

            "timeaware25_silhouette": timeaware_res.get("timeaware25_silhouette"),
            "timeaware25_db": timeaware_res.get("timeaware25_db"),
            "timeaware25_mean_best_corr_to_baseline": timeaware_res.get("timeaware25_mean_best_corr_to_baseline"),
            "timeaware25_min_best_corr_to_baseline": timeaware_res.get("timeaware25_min_best_corr_to_baseline"),
            "timeaware25_max_best_corr_to_baseline": timeaware_res.get("timeaware25_max_best_corr_to_baseline"),

            "vanilla25_better_match_than_timeaware25": (
                safe_float(vanilla_res.get("vanilla25_mean_best_corr_to_baseline")) is not None
                and safe_float(timeaware_res.get("timeaware25_mean_best_corr_to_baseline")) is not None
                and float(vanilla_res["vanilla25_mean_best_corr_to_baseline"])
                > float(timeaware_res["timeaware25_mean_best_corr_to_baseline"])
            ),
            "timeaware25_better_match_than_vanilla25": (
                safe_float(vanilla_res.get("vanilla25_mean_best_corr_to_baseline")) is not None
                and safe_float(timeaware_res.get("timeaware25_mean_best_corr_to_baseline")) is not None
                and float(timeaware_res["timeaware25_mean_best_corr_to_baseline"])
                > float(vanilla_res["vanilla25_mean_best_corr_to_baseline"])
            ),
        })

        save_json(summary, sample_out_dir / "step3_one_sample_summary.json")
        return summary

    except KeyboardInterrupt:
        return {
            "status": "failed",
            "trade_date": str(trade_date),
            "symbol": str(symbol).zfill(6),
            "sample_out_dir": str(sample_out_dir),
            "fail_stage": "step3_one_sample",
            "fail_reason": "keyboard_interrupt",
            "traceback": traceback.format_exc(),
        }

    except Exception as e:
        return {
            "status": "failed",
            "trade_date": str(trade_date),
            "symbol": str(symbol).zfill(6),
            "sample_out_dir": str(sample_out_dir),
            "fail_stage": "step3_one_sample",
            "fail_reason": str(e),
            "traceback": traceback.format_exc(),
        }


def detect_exchange_from_symbol(symbol: str) -> str:
    """
    轻量市场识别。
    这里只用于 universe 标注，不替代 Step0/Step1 的正式市场处理逻辑。
    """
    symbol = str(symbol).zfill(6)
    if symbol.startswith(("600", "601", "602", "603", "605", "609", "688", "689")):
        return "SH"
    if symbol.startswith(("000", "001", "002", "003", "300", "301")):
        return "SZ"
    return "UNKNOWN"


def _safe_file_size(path: Path) -> int:
    """
    安全读取文件大小，失败时返回 0。
    """
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0
    
def _sample_symbol_stems_from_folder(
    folder: Path,
    max_symbols: int,
    random_state: int = 42,
    oversample_factor: int = 8,
) -> list[str]:
    """
    快速近似抽样版：
    - 不再扫描整个目录
    - 只读取前一小段候选文件
    - 再从候选里随机抽 max_symbols 个

    这样会快很多，适合先构大 universe。
    """
    if not folder.exists():
        return []

    candidate_limit = max(max_symbols * oversample_factor, max_symbols)

    candidates: list[str] = []

    with os.scandir(folder) as it:
        for entry in it:
            if len(candidates) >= candidate_limit:
                break
            if not entry.is_file():
                continue
            name = entry.name
            if not name.lower().endswith(".ftr"):
                continue
            stem = name[:-4]
            candidates.append(stem)

    candidates = sorted(set(candidates))

    if len(candidates) <= max_symbols:
        return candidates

    rng = random.Random(random_state)
    sampled = rng.sample(candidates, k=max_symbols)
    return sorted(sampled)

def _sample_symbol_set_from_folder(
    folder: Path,
    max_symbols: int,
    random_state: int = 42,
    oversample_factor: int = 8,
) -> set[str]:
    sampled = _sample_symbol_stems_from_folder(
        folder=folder,
        max_symbols=max_symbols,
        random_state=random_state,
        oversample_factor=oversample_factor,
    )
    return set(sampled)

def _sample_trade_date_dirs(
    raw_root: Path,
    max_trade_dates: int | None,
    random_state: int = 42,
) -> list[Path]:
    trade_date_dirs = [
        p for p in raw_root.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    trade_date_dirs = sorted(trade_date_dirs, key=lambda p: p.name)

    if max_trade_dates is not None and len(trade_date_dirs) > max_trade_dates:
        rng = random.Random(random_state)
        trade_date_dirs = rng.sample(trade_date_dirs, k=max_trade_dates)
        trade_date_dirs = sorted(trade_date_dirs, key=lambda p: p.name)

    return trade_date_dirs


def build_valid_sample_universe_from_raw_data(
    raw_root: str | Path,
    out_csv: str | Path,
    max_trade_dates: int | None = None,
    random_state: int = 42,
    max_symbols_per_date: int | None = 200,
    anchor_table: str = "tick",
    overscan_factor: int = 3,
    compute_file_size: bool = True,
) -> pd.DataFrame:
    """
    更快版：
    - 每个交易日仅扫描 order/trade/tick 三个目录的文件名
    - 用 symbol 交集构造合法 universe
    - 不再对每个候选 symbol 做 3 次 exists()
    """
    raw_root = Path(raw_root)
    out_csv = Path(out_csv)

    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    trade_date_dirs = _sample_trade_date_dirs(
        raw_root=raw_root,
        max_trade_dates=max_trade_dates,
        random_state=random_state,
    )

    rows = []
    rng = random.Random(random_state)

    for td_dir in trade_date_dirs:
        trade_date = td_dir.name
        order_dir = td_dir / "order"
        trade_dir = td_dir / "trade"
        tick_dir = td_dir / "tick"

        if not (order_dir.exists() and trade_dir.exists() and tick_dir.exists()):
            continue

        # 只扫文件名，不逐个 exists
        scan_limit = None
        if max_symbols_per_date is not None:
            scan_limit = max_symbols_per_date * overscan_factor

        order_syms = set(_list_symbol_stems_from_folder_fast(order_dir, limit=scan_limit))
        trade_syms = set(_list_symbol_stems_from_folder_fast(trade_dir, limit=scan_limit))
        tick_syms  = set(_list_symbol_stems_from_folder_fast(tick_dir,  limit=scan_limit))

        valid_syms = sorted(order_syms & trade_syms & tick_syms)

        if len(valid_syms) == 0:
            continue

        # anchor 目录优先只是保留你的设计意图，但不再逐 symbol exists
        if anchor_table == "order":
            anchor_syms = order_syms
        elif anchor_table == "trade":
            anchor_syms = trade_syms
        else:
            anchor_syms = tick_syms

        valid_syms = [s for s in valid_syms if s in anchor_syms]

        if max_symbols_per_date is not None and len(valid_syms) > max_symbols_per_date:
            rng_local = random.Random(random_state + int(trade_date))
            valid_syms = sorted(rng_local.sample(valid_syms, k=max_symbols_per_date))

        for symbol in valid_syms:
            symbol = str(symbol).zfill(6)

            order_file = order_dir / f"{symbol}.ftr"
            trade_file = trade_dir / f"{symbol}.ftr"
            tick_file = tick_dir / f"{symbol}.ftr"

            if compute_file_size:
                order_size = _safe_file_size(order_file)
                trade_size = _safe_file_size(trade_file)
                tick_size = _safe_file_size(tick_file)
            else:
                order_size = 0
                trade_size = 0
                tick_size = 0

            size_proxy_mb = (order_size + trade_size + tick_size) / (1024 ** 2)

            rows.append({
                "trade_date": str(trade_date),
                "symbol": symbol,
                "exchange": detect_exchange_from_symbol(symbol),
                "order_file": str(order_file),
                "trade_file": str(trade_file),
                "tick_file": str(tick_file),
                "order_size_mb": round(order_size / (1024 ** 2), 6),
                "trade_size_mb": round(trade_size / (1024 ** 2), 6),
                "tick_size_mb": round(tick_size / (1024 ** 2), 6),
                "size_proxy_mb": round(size_proxy_mb, 6),
                "is_valid_step3_sample": True,
            })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise ValueError("No valid samples found from raw dataset.")

    # 如果没算文件大小，就给一个退化 proxy，避免 quantile 报错
    if compute_file_size and df["size_proxy_mb"].notna().any() and df["size_proxy_mb"].sum() > 0:
        q1 = df["size_proxy_mb"].quantile(1 / 3)
        q2 = df["size_proxy_mb"].quantile(2 / 3)

        def _bucket(x: float) -> str:
            if x <= q1:
                return "low"
            elif x <= q2:
                return "medium"
            return "high"

        df["liquidity_bucket"] = df["size_proxy_mb"].apply(_bucket)
    else:
        df["liquidity_bucket"] = "unknown"

    df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    return df
    
def sample_step3_pairs_from_universe(
    universe_df: pd.DataFrame,
    n_samples: int = 50,
    random_state: int = 42,
    use_liquidity_balanced_sampling: bool = True,
) -> pd.DataFrame:
    """
    从 valid sample universe 中抽样。

    默认做轻度 liquidity balance：
    - high / medium / low 尽量均衡
    """
    df = universe_df.copy()
    df = df[df["is_valid_step3_sample"] == True].copy()

    if len(df) == 0:
        raise ValueError("No valid samples available for sampling.")

    rng = np.random.default_rng(random_state)

    if not use_liquidity_balanced_sampling:
        n = min(n_samples, len(df))
        sampled_idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
        out = df.loc[sampled_idx].copy()
        return out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    # 轻量分层抽样
    buckets = ["high", "medium", "low"]
    bucket_dfs = {b: df[df["liquidity_bucket"] == b].copy() for b in buckets}

    base_n = n_samples // 3
    remainder = n_samples % 3
    target_counts = {
        "high": base_n + (1 if remainder > 0 else 0),
        "medium": base_n + (1 if remainder > 1 else 0),
        "low": base_n,
    }

    sampled_parts = []

    for b in buckets:
        bdf = bucket_dfs[b]
        if len(bdf) == 0:
            continue

        take_n = min(target_counts[b], len(bdf))
        sampled_idx = rng.choice(bdf.index.to_numpy(), size=take_n, replace=False)
        sampled_parts.append(bdf.loc[sampled_idx].copy())

    out = pd.concat(sampled_parts, axis=0, ignore_index=True)

    # 如果某个 bucket 不够，补齐到 n_samples
    if len(out) < min(n_samples, len(df)):
        already = set(out.index.tolist()) if len(out) > 0 else set()
        remaining = df.drop(index=out.index, errors="ignore").copy()
        if len(remaining) > 0:
            need = min(n_samples, len(df)) - len(out)
            add_n = min(need, len(remaining))
            sampled_idx = rng.choice(remaining.index.to_numpy(), size=add_n, replace=False)
            out = pd.concat([out, remaining.loc[sampled_idx].copy()], axis=0, ignore_index=True)

    out = out.head(min(n_samples, len(df))).copy()
    out = out.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    return out

def _list_symbol_stems_from_folder_fast(
    folder: Path,
    limit: int | None = None,
) -> list[str]:
    """
    只扫描目录项名字，不对每个文件做额外 stat。
    返回去掉 .ftr 后缀的 symbol 列表。
    """
    if not folder.exists():
        return []

    out = []
    with os.scandir(folder) as it:
        for entry in it:
            name = entry.name
            # DirEntry.is_file() 在某些文件系统上也可能触发额外 metadata；
            # 这里只按名字过滤，尽量避免慢调用
            if not name.lower().endswith(".ftr"):
                continue
            out.append(name[:-4])

            if limit is not None and len(out) >= limit:
                break

    return sorted(set(out))

def build_and_preview_step3_sample_list(
    raw_root: str | Path,
    universe_csv: str | Path,
    sample_list_csv: str | Path,
    n_samples: int = 50,
    max_trade_dates: int | None = None,
    random_state: int = 42,
    max_symbols_per_date: int | None = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) 先构造 valid universe
    2) 再随机抽样
    3) 保存 universe 和 sample list
    4) 打印预览
    """
    universe_df = build_valid_sample_universe_from_raw_data(
        raw_root=raw_root,
        out_csv=universe_csv,
        max_trade_dates=max_trade_dates,
        random_state=random_state,
        max_symbols_per_date=max_symbols_per_date,
        anchor_table="tick",
    )

    sample_df = sample_step3_pairs_from_universe(
        universe_df=universe_df,
        n_samples=n_samples,
        random_state=random_state,
        use_liquidity_balanced_sampling=True,
    )

    # Step3 先只生成基础 sample list，artifact 路径后面再补
    sample_df = sample_df.copy()
    sample_df["existing_step1_artifact_dir"] = None
    sample_df["existing_step2_artifact_dir"] = None

    sample_list_csv = Path(sample_list_csv)
    ensure_dir(sample_list_csv.parent)
    sample_df.to_csv(sample_list_csv, index=False)

    print("\n" + "=" * 120)
    print("STEP3 VALID SAMPLE UNIVERSE")
    print("=" * 120)
    print(f"[universe_csv] = {universe_csv}")
    print(f"[n_universe_rows] = {len(universe_df)}")
    print(f"[n_trade_dates] = {universe_df['trade_date'].nunique()}")
    print(f"[n_symbols] = {universe_df['symbol'].nunique()}")
    print("\n[universe liquidity bucket counts]")
    print(universe_df["liquidity_bucket"].value_counts(dropna=False))

    print("\n" + "=" * 120)
    print("STEP3 SAMPLED LIST PREVIEW")
    print("=" * 120)
    print(f"[sample_list_csv] = {sample_list_csv}")
    print(f"[n_samples] = {len(sample_df)}")
    print("\n[sample liquidity bucket counts]")
    print(sample_df["liquidity_bucket"].value_counts(dropna=False))
    print("\n[sample preview head(30)]")
    print(sample_df.head(30))

    return universe_df, sample_df


def _is_valid_step1_artifact_dir(path: str | Path | None) -> bool:
    if path is None:
        return False
    path = Path(path)
    return path.exists() and (path / "step1_event_stream_final.pkl").exists()


def _is_valid_step2_artifact_dir(path: str | Path | None) -> bool:
    if path is None:
        return False
    path = Path(path)
    return (
        path.exists()
        and (path / "es_minimal_for_sequence.pkl").exists()
        and (path / "episode_table.pkl").exists()
        and (path / "episode_summary.pkl").exists()
    )

def audit_artifact_availability_for_sample_list(
    sample_list_csv: str | Path,
    out_csv: str | Path,
    default_step1_lookup: dict[str, str] | None = None,
    default_step2_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    对 sample list 做 artifact availability audit。

    当前轻量版逻辑：
    - 按 symbol 查找默认 Step1 artifact
    - 按 symbol 或固定目录查找默认 Step2 artifact
    - 只做存在性检查，不在这里构建新 artifact

    参数：
    - default_step1_lookup: 例如 {"000001": "/path/to/step1_outputs_cached_000001"}
    - default_step2_lookup: 例如 {"000001": "/path/to/step2_artifacts_000001"}
    """
    df = pd.read_csv(sample_list_csv, dtype={"trade_date": str, "symbol": str})
    df["trade_date"] = df["trade_date"].astype(str)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)

    if "existing_step1_artifact_dir" not in df.columns:
        df["existing_step1_artifact_dir"] = None
    if "existing_step2_artifact_dir" not in df.columns:
        df["existing_step2_artifact_dir"] = None

    default_step1_lookup = default_step1_lookup or {}
    default_step2_lookup = default_step2_lookup or {}

    audited_rows = []

    for row in df.itertuples(index=False):
        rec = dict(row._asdict())

        symbol = str(rec["symbol"]).zfill(6)

        # -------------------------
        # Step1 artifact
        # -------------------------
        cur_step1 = rec.get("existing_step1_artifact_dir")
        if pd.isna(cur_step1) or not str(cur_step1).strip():
            cur_step1 = default_step1_lookup.get(symbol)

        step1_ok = _is_valid_step1_artifact_dir(cur_step1)

        # -------------------------
        # Step2 artifact
        # -------------------------
        cur_step2 = rec.get("existing_step2_artifact_dir")
        if pd.isna(cur_step2) or not str(cur_step2).strip():
            cur_step2 = default_step2_lookup.get(symbol)

        step2_ok = _is_valid_step2_artifact_dir(cur_step2)

        rec["existing_step1_artifact_dir"] = cur_step1 if step1_ok else None
        rec["existing_step2_artifact_dir"] = cur_step2 if step2_ok else None
        rec["step1_artifact_exists"] = bool(step1_ok)
        rec["step2_artifact_exists"] = bool(step2_ok)
        rec["ready_for_direct_step3"] = bool(step1_ok and step2_ok)

        audited_rows.append(rec)

    audited_df = pd.DataFrame(audited_rows)

    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)
    audited_df.to_csv(out_csv, index=False)

    return audited_df

def preview_artifact_audit_for_sample_list(
    sample_list_csv: str | Path,
    audited_out_csv: str | Path,
    default_step1_lookup: dict[str, str] | None = None,
    default_step2_lookup: dict[str, str] | None = None,
    preview_n: int = 30,
) -> pd.DataFrame:
    audited_df = audit_artifact_availability_for_sample_list(
        sample_list_csv=sample_list_csv,
        out_csv=audited_out_csv,
        default_step1_lookup=default_step1_lookup,
        default_step2_lookup=default_step2_lookup,
    )

    print("\n" + "=" * 120)
    print("STEP3 SAMPLE LIST ARTIFACT AUDIT")
    print("=" * 120)
    print(f"[audited_out_csv] = {audited_out_csv}")
    print(f"[n_rows] = {len(audited_df)}")

    print("\n[step1_artifact_exists counts]")
    print(audited_df["step1_artifact_exists"].value_counts(dropna=False))

    print("\n[step2_artifact_exists counts]")
    print(audited_df["step2_artifact_exists"].value_counts(dropna=False))

    print("\n[ready_for_direct_step3 counts]")
    print(audited_df["ready_for_direct_step3"].value_counts(dropna=False))

    print("\n[preview head]")
    print(audited_df.head(preview_n))

    return audited_df

def build_and_save_step2_artifacts_for_one_sample(
    trade_date: str | int,
    symbol: str,
    step1_artifact_dir: str | Path,
    sample_out_dir: Path,
) -> Dict[str, Any]:
    result = {
        "status": "success",
        "trade_date": str(trade_date),
        "symbol": str(symbol).zfill(6),
    }

    try:
        step2_artifact_dir = ensure_dir(sample_out_dir / "step2_artifact")

        print(f"[debug] symbol={str(symbol).zfill(6)}")
        print(f"[debug] step1_artifact_dir={Path(step1_artifact_dir)}")
        print(f"[debug] step1_artifact_exists={(Path(step1_artifact_dir) / 'step1_event_stream_final.pkl').exists()}")

        # 直接调用你 step2.py 里已经写好的正式 artifact builder
        es, episode_table, episode_summary = build_and_save_step2_artifacts(
            symbol=str(symbol).zfill(6),
            out_dir=str(step2_artifact_dir),
            keep_sessions=["continuous_am", "continuous_pm"],
            gap_hard_ms=3000,
            lag_hard_ms=5000,
            max_events_per_episode=64,
            max_duration_ms=10000,
            min_events_soft_cut=8,
            min_events_episode_flag=4,
            step1_artifact_dir=str(step1_artifact_dir),
        )

        result.update({
            "step2_artifact_dir": str(step2_artifact_dir),
            "n_events": int(len(es)) if es is not None else 0,
            "n_episodes": int(len(episode_table)) if episode_table is not None else 0,
            "episode_summary_shape": list(episode_summary.shape) if hasattr(episode_summary, "shape") else None,
        })

    except KeyboardInterrupt:
        result["status"] = "failed"
        result["fail_stage"] = "step2_artifact_build"
        result["fail_reason"] = "keyboard_interrupt"
        result["traceback"] = traceback.format_exc()
        return result

    except Exception as e:
        result["status"] = "failed"
        result["fail_stage"] = "step2_artifact_build"
        result["fail_reason"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result

def build_step1_step2_artifacts_for_sample_list(
    audited_sample_list_csv: str | Path,
    out_csv: str | Path,
    n_build: int = 5,
    rebuild_step1: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    从 audited sample list 中，给前 n_build 个样本批量构建 Step1 / Step2 artifact。

    输出：
    - existing_step1_artifact_dir
    - existing_step2_artifact_dir
    - step1_artifact_exists
    - step2_artifact_exists
    - artifact_build_status
    - artifact_build_fail_stage
    - artifact_build_fail_reason
    """
    df = pd.read_csv(audited_sample_list_csv, dtype={"trade_date": str, "symbol": str})
    df["trade_date"] = df["trade_date"].astype(str)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)

    out_rows = []
    build_df = df.head(n_build).copy()

    for i, row in enumerate(build_df.itertuples(index=False), start=1):
        trade_date = str(row.trade_date)
        symbol = str(row.symbol).zfill(6)

        if verbose:
            print("\n" + "=" * 120)
            print(f"BUILD STEP1/STEP2 ARTIFACTS | {i}/{len(build_df)} | trade_date={trade_date} symbol={symbol}")
            print("=" * 120)
        else:
            print(f"[{i}/{len(build_df)}] {trade_date} {symbol}")

        sample_out_dir = get_sample_output_dir(trade_date, symbol)

        rec = dict(row._asdict())
        rec["artifact_build_status"] = "success"
        rec["artifact_build_fail_stage"] = None
        rec["artifact_build_fail_reason"] = None

        # ------------------------------------------------------------------
        # Step1
        # ------------------------------------------------------------------
        if verbose:
            step1_res = run_step1_for_one_sample(
                trade_date=trade_date,
                symbol=symbol,
                sample_out_dir=sample_out_dir,
                rebuild_step1=rebuild_step1,
                existing_step1_artifact_dir=None,
            )
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                step1_res = run_step1_for_one_sample(
                    trade_date=trade_date,
                    symbol=symbol,
                    sample_out_dir=sample_out_dir,
                    rebuild_step1=rebuild_step1,
                    existing_step1_artifact_dir=None,
                )

        if step1_res["status"] != "success":
            rec["artifact_build_status"] = "failed"
            rec["artifact_build_fail_stage"] = step1_res.get("fail_stage")
            rec["artifact_build_fail_reason"] = step1_res.get("fail_reason")
            rec["existing_step1_artifact_dir"] = None
            rec["existing_step2_artifact_dir"] = None
            rec["step1_artifact_exists"] = False
            rec["step2_artifact_exists"] = False
            out_rows.append(rec)
            continue

        step1_artifact_dir = step1_res["step1_out_dir"]

        # ------------------------------------------------------------------
        # Step2
        # ------------------------------------------------------------------
        if verbose:
            step2_res = build_and_save_step2_artifacts_for_one_sample(
                trade_date=trade_date,
                symbol=symbol,
                step1_artifact_dir=step1_artifact_dir,
                sample_out_dir=sample_out_dir,
            )
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                step2_res = build_and_save_step2_artifacts_for_one_sample(
                    trade_date=trade_date,
                    symbol=symbol,
                    step1_artifact_dir=step1_artifact_dir,
                    sample_out_dir=sample_out_dir,
                )

        if step2_res["status"] != "success":
            rec["artifact_build_status"] = "failed"
            rec["artifact_build_fail_stage"] = step2_res.get("fail_stage")
            rec["artifact_build_fail_reason"] = step2_res.get("fail_reason")
            rec["existing_step1_artifact_dir"] = step1_artifact_dir
            rec["existing_step2_artifact_dir"] = None
            rec["step1_artifact_exists"] = _is_valid_step1_artifact_dir(step1_artifact_dir)
            rec["step2_artifact_exists"] = False
            out_rows.append(rec)
            continue

        step2_artifact_dir = step2_res["step2_artifact_dir"]

        rec["existing_step1_artifact_dir"] = step1_artifact_dir
        rec["existing_step2_artifact_dir"] = step2_artifact_dir
        rec["step1_artifact_exists"] = _is_valid_step1_artifact_dir(step1_artifact_dir)
        rec["step2_artifact_exists"] = _is_valid_step2_artifact_dir(step2_artifact_dir)
        rec["ready_for_direct_step3"] = bool(
            rec["step1_artifact_exists"] and rec["step2_artifact_exists"]
        )

        out_rows.append(rec)

        if not verbose:
            print(
                f"  -> status={rec.get('artifact_build_status')}, "
                f"step1={rec.get('step1_artifact_exists')}, "
                f"step2={rec.get('step2_artifact_exists')}, "
                f"fail_stage={rec.get('artifact_build_fail_stage')}, "
                f"fail_reason={rec.get('artifact_build_fail_reason')}"
            )

    out_df = pd.DataFrame(out_rows)

    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)
    out_df.to_csv(out_csv, index=False)

    return out_df

def preview_built_artifacts_for_sample_list(
    audited_sample_list_csv: str | Path,
    built_out_csv: str | Path,
    n_build: int = 1000,
    rebuild_step1: bool = False,
    preview_n: int = 20,
    verbose: bool = False,
) -> pd.DataFrame:
    built_df = build_step1_step2_artifacts_for_sample_list(
        audited_sample_list_csv=audited_sample_list_csv,
        out_csv=built_out_csv,
        n_build=n_build,
        rebuild_step1=rebuild_step1,
        verbose=verbose,
    )

    print("\n" + "=" * 120)
    print("STEP3 BUILT ARTIFACTS PREVIEW")
    print("=" * 120)
    print(f"[built_out_csv] = {built_out_csv}")
    print(f"[n_rows] = {len(built_df)}")

    if "artifact_build_status" in built_df.columns:
        print("\n[artifact_build_status counts]")
        print(built_df["artifact_build_status"].value_counts(dropna=False))

    if "step1_artifact_exists" in built_df.columns:
        print("\n[step1_artifact_exists counts]")
        print(built_df["step1_artifact_exists"].value_counts(dropna=False))

    if "step2_artifact_exists" in built_df.columns:
        print("\n[step2_artifact_exists counts]")
        print(built_df["step2_artifact_exists"].value_counts(dropna=False))

    show_cols = [
        "trade_date",
        "symbol",
        "liquidity_bucket",
        "artifact_build_status",
        "artifact_build_fail_stage",
        "artifact_build_fail_reason",
        "step1_artifact_exists",
        "step2_artifact_exists",
        "ready_for_direct_step3",
    ]
    show_cols = [c for c in show_cols if c in built_df.columns]

    print("\n[preview head]")
    print(built_df[show_cols].head(preview_n).to_string(index=False))
    print("\n[full fail reasons]")
    for i, row in built_df.iterrows():
        print(f"\nrow={i} trade_date={row['trade_date']} symbol={row['symbol']}")
        print("fail_stage:", row.get("artifact_build_fail_stage"))
        print("fail_reason:", row.get("artifact_build_fail_reason"))

    return built_df


def flatten_step3_one_sample_result(res: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": res.get("status"),
        "trade_date": res.get("trade_date"),
        "symbol": res.get("symbol"),
        "sample_out_dir": res.get("sample_out_dir"),

        "step1_out_dir": res.get("step1_out_dir"),
        "step2_artifact_dir": res.get("step2_artifact_dir"),

        "used_cached_step1": res.get("used_cached_step1"),
        "used_external_step1_artifact": res.get("used_external_step1_artifact"),
        "used_external_step2_artifact": res.get("used_external_step2_artifact"),

        "n_events": res.get("n_events"),
        "n_episodes": res.get("n_episodes"),

        "baseline_source": res.get("baseline_source"),
        "baseline_silhouette": res.get("baseline_silhouette"),
        "baseline_db": res.get("baseline_db"),

        "vanilla25_silhouette": res.get("vanilla25_silhouette"),
        "vanilla25_db": res.get("vanilla25_db"),
        "vanilla25_mean_best_corr_to_baseline": res.get("vanilla25_mean_best_corr_to_baseline"),
        "vanilla25_min_best_corr_to_baseline": res.get("vanilla25_min_best_corr_to_baseline"),
        "vanilla25_max_best_corr_to_baseline": res.get("vanilla25_max_best_corr_to_baseline"),

        "timeaware25_silhouette": res.get("timeaware25_silhouette"),
        "timeaware25_db": res.get("timeaware25_db"),
        "timeaware25_mean_best_corr_to_baseline": res.get("timeaware25_mean_best_corr_to_baseline"),
        "timeaware25_min_best_corr_to_baseline": res.get("timeaware25_min_best_corr_to_baseline"),
        "timeaware25_max_best_corr_to_baseline": res.get("timeaware25_max_best_corr_to_baseline"),

        "vanilla25_better_match_than_timeaware25": res.get("vanilla25_better_match_than_timeaware25"),
        "timeaware25_better_match_than_vanilla25": res.get("timeaware25_better_match_than_vanilla25"),

        "fail_stage": res.get("fail_stage"),
        "fail_reason": res.get("fail_reason"),
    }

def run_step3_batch_from_built_sample_list(
    built_sample_list_csv: str | Path,
    out_csv: str | Path,
    out_json: str | Path,
    limit: int | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    df = pd.read_csv(built_sample_list_csv, dtype={"trade_date": str, "symbol": str})
    df["trade_date"] = df["trade_date"].astype(str)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)

    if (
        "ready_for_direct_step3" in df.columns
        and df["ready_for_direct_step3"].fillna(False).any()
    ):
        df = df[df["ready_for_direct_step3"] == True].copy()
    else:
        df = df[
            (df["step1_artifact_exists"] == True) &
            (df["step2_artifact_exists"] == True)
        ].copy()

    if limit is not None:
        df = df.head(limit).copy()

    results = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        trade_date = str(row.trade_date)
        symbol = str(row.symbol).zfill(6)

        print("\n" + "=" * 120)
        res = run_step3_one_sample(
            trade_date=trade_date,
            symbol=symbol,
            rebuild_step1=False,
            existing_step1_artifact_dir=row.existing_step1_artifact_dir,
            existing_step2_artifact_dir=row.existing_step2_artifact_dir,
            device=device,
        )

        flat = flatten_step3_one_sample_result(res)
        results.append(flat)

        status = flat.get("status")
        vanilla_match = flat.get("vanilla25_mean_best_corr_to_baseline")
        timeaware_match = flat.get("timeaware25_mean_best_corr_to_baseline")

        winner = "tie"
        if vanilla_match is not None and timeaware_match is not None:
            if vanilla_match > timeaware_match:
                winner = "vanilla25"
            elif timeaware_match > vanilla_match:
                winner = "timeaware25"

        print(
            f"[{i}/{len(df)}] {trade_date}_{symbol} | "
            f"status={status} | "
            f"vanilla_match={vanilla_match} | "
            f"timeaware_match={timeaware_match} | "
            f"winner={winner}"
        )

        tmp_df = pd.DataFrame(results)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        tmp_df.to_csv(out_csv, index=False)

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        summary = {
            "n_total": 0,
            "n_success": 0,
            "n_failed": 0,
            "fail_reason": "no runnable samples after filtering",
        }
        out_json = Path(out_json)
        ensure_dir(out_json.parent)
        save_json(summary, out_json)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_csv, index=False)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return results_df

    summary = {
        "n_total": int(len(results_df)),
        "n_success": int((results_df["status"] == "success").sum()) if len(results_df) > 0 else 0,
        "n_failed": int((results_df["status"] != "success").sum()) if len(results_df) > 0 else 0,
    }

    success_df = results_df[results_df["status"] == "success"].copy()
    if len(success_df) > 0:
        match_diff = (
            success_df["vanilla25_mean_best_corr_to_baseline"]
            - success_df["timeaware25_mean_best_corr_to_baseline"]
        )

        summary.update({
            "mean_baseline_silhouette": safe_float(success_df["baseline_silhouette"].mean()),
            "mean_baseline_db": safe_float(success_df["baseline_db"].mean()),

            "mean_vanilla25_silhouette": safe_float(success_df["vanilla25_silhouette"].mean()),
            "mean_vanilla25_db": safe_float(success_df["vanilla25_db"].mean()),
            "mean_vanilla25_match": safe_float(success_df["vanilla25_mean_best_corr_to_baseline"].mean()),

            "mean_timeaware25_silhouette": safe_float(success_df["timeaware25_silhouette"].mean()),
            "mean_timeaware25_db": safe_float(success_df["timeaware25_db"].mean()),
            "mean_timeaware25_match": safe_float(success_df["timeaware25_mean_best_corr_to_baseline"].mean()),

            "pct_vanilla25_better_match": safe_float(
                success_df["vanilla25_better_match_than_timeaware25"].mean()
            ),
            "pct_timeaware25_better_match": safe_float(
                success_df["timeaware25_better_match_than_vanilla25"].mean()
            ),

            "mean_match_diff_vanilla_minus_timeaware": safe_float(match_diff.mean()),
            "median_match_diff_vanilla_minus_timeaware": safe_float(match_diff.median()),
        })

        if summary.get("mean_vanilla25_match") is not None and summary.get("mean_timeaware25_match") is not None:
            if summary["mean_vanilla25_match"] > summary["mean_timeaware25_match"]:
                summary["final_conclusion"] = (
                    "vanilla25 is preferred as the mainline encoder under current Step3 batch validation"
                )
            elif summary["mean_timeaware25_match"] > summary["mean_vanilla25_match"]:
                summary["final_conclusion"] = (
                    "timeaware25 is preferred as the mainline encoder under current Step3 batch validation"
                )
            else:
                summary["final_conclusion"] = (
                    "vanilla25 and timeaware25 are tied under current Step3 batch validation"
                )

    out_json = Path(out_json)
    ensure_dir(out_json.parent)
    save_json(summary, out_json)

    print("\n" + "=" * 120)
    print("STEP3 BATCH SUMMARY")
    print("=" * 120)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return results_df

def filter_trade_dates_to_recent_window(
    trade_date_list: list[str | int],
    recent_months: int = 12,
) -> list[str]:
    """
    输入一组 YYYYMMDD trade_date，保留最近 recent_months 的日期。
    口径：
    - 以样本中最新 trade_date 为窗口终点
    - 不是按自然年，而是 rolling recent window
    """
    s = pd.Series([str(x) for x in trade_date_list], dtype="string")
    s = s.dropna().drop_duplicates().sort_values()

    dt = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    valid = dt.notna()

    s = s.loc[valid].reset_index(drop=True)
    dt = dt.loc[valid].reset_index(drop=True)

    if len(dt) == 0:
        return []

    end_dt = dt.max()
    start_dt = end_dt - pd.DateOffset(months=recent_months)

    keep = (dt >= start_dt) & (dt <= end_dt)
    out = s.loc[keep].tolist()
    return sorted(out)

def list_all_trade_dates_from_raw_root(raw_root: str | Path) -> list[str]:
    """
    从 raw_root 下读取所有像 YYYYMMDD 的交易日目录名。
    只按目录名判断，不进入子目录。
    """
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    out = []
    for p in raw_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if len(name) == 8 and name.isdigit():
            out.append(name)

    return sorted(set(out))

def get_recent_trade_dates_from_raw_root(
    raw_root: str | Path,
    recent_months: int = 12,
) -> list[str]:
    """
    从 raw_root 读取全部交易日目录，再保留最近 recent_months 的窗口。
    """
    all_trade_dates = list_all_trade_dates_from_raw_root(raw_root)
    recent_trade_dates = filter_trade_dates_to_recent_window(
        all_trade_dates,
        recent_months=recent_months,
    )
    return recent_trade_dates

def select_trade_dates_evenly_from_recent_window(
    recent_trade_dates: list[str | int],
    max_trade_dates: int = 120,
) -> list[str]:
    """
    从 recent_trade_dates 中尽量均匀地抽取 max_trade_dates 个交易日。
    设计目标：
    - 不按年份分层
    - 仍然只在 recent window 内
    - 让选中的日期尽量铺满整个 12 个月窗口
    """
    dates = sorted({str(x) for x in recent_trade_dates if str(x).isdigit() and len(str(x)) == 8})

    n = len(dates)
    if n == 0:
        return []

    if n <= max_trade_dates:
        return dates

    # 在 [0, n-1] 上等距取点
    idx_float = np.linspace(0, n - 1, num=max_trade_dates)
    idx = np.round(idx_float).astype(int)

    # 去重保护；极少数情况下 round 后可能重复
    idx = sorted(set(idx.tolist()))

    # 如果去重后不足 max_trade_dates，则顺序补齐最近缺的点
    if len(idx) < max_trade_dates:
        need = max_trade_dates - len(idx)
        all_idx = list(range(n))
        remain = [i for i in all_idx if i not in set(idx)]
        idx.extend(remain[:need])
        idx = sorted(idx)

    selected = [dates[i] for i in idx[:max_trade_dates]]
    return selected

def list_symbol_stems_from_one_table_dir(table_dir: str | Path) -> list[str]:
    """
    从某一天某张表目录中读取所有 .ftr 文件名对应的 symbol。
    只按文件名判断，不做额外 stat。
    """
    table_dir = Path(table_dir)
    if not table_dir.exists():
        return []

    out = []
    for p in table_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if not name.endswith(".ftr"):
            continue
        stem = p.stem
        if len(stem) == 6 and stem.isdigit():
            out.append(stem)

    return sorted(set(out))

def build_daily_valid_symbol_list_from_raw(
    raw_root: str | Path,
    trade_date: str | int,
) -> list[str]:
    """
    对单个 trade_date：
    - 读取 order / trade / tick 三个目录下的 symbol
    - 取三者交集
    - 返回合法 symbol 列表
    """
    raw_root = Path(raw_root)
    trade_date = str(trade_date)

    order_dir = raw_root / trade_date / "order"
    trade_dir = raw_root / trade_date / "trade"
    tick_dir = raw_root / trade_date / "tick"

    order_syms = set(list_symbol_stems_from_one_table_dir(order_dir))
    trade_syms = set(list_symbol_stems_from_one_table_dir(trade_dir))
    tick_syms = set(list_symbol_stems_from_one_table_dir(tick_dir))

    valid_syms = sorted(order_syms & trade_syms & tick_syms)
    return valid_syms

def build_availability_table_from_file_index_chunked(
    file_index_csv: str | Path,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """
    分块读取 file index，构造 availability 表：
        trade_date, symbol, has_order, has_trade, has_tick, is_valid_step3_sample, exchange

    适合大 CSV，避免一次性整表字符串处理过慢。
    """
    from collections import defaultdict

    file_index_csv = Path(file_index_csv)
    if not file_index_csv.exists():
        raise FileNotFoundError(f"file_index_csv not found: {file_index_csv}")

    # key = (trade_date, symbol)
    # value = [has_order, has_trade, has_tick]
    agg = defaultdict(lambda: [0, 0, 0])

    reader = pd.read_csv(
        file_index_csv,
        header=None,
        names=["trade_date", "table_name", "symbol"],
        dtype=str,
        chunksize=chunksize,
    )

    for i, chunk in enumerate(reader, start=1):
        chunk = chunk.dropna(subset=["trade_date", "table_name", "symbol"]).copy()

        # 用 object 字符串处理，避免 pandas StringDtype 的高开销
        chunk["trade_date"] = chunk["trade_date"].astype(str).str.strip()
        chunk["table_name"] = chunk["table_name"].astype(str).str.strip().str.lower()
        chunk["symbol"] = chunk["symbol"].astype(str).str.strip().str.zfill(6)

        chunk = chunk[
            chunk["trade_date"].str.match(r"^\d{8}$")
            & chunk["symbol"].str.match(r"^\d{6}$")
            & chunk["table_name"].isin(["order", "trade", "tick"])
        ]

        for td, tb, sym in chunk.itertuples(index=False, name=None):
            key = (td, sym)
            if tb == "order":
                agg[key][0] = 1
            elif tb == "trade":
                agg[key][1] = 1
            elif tb == "tick":
                agg[key][2] = 1

        if i % 5 == 0:
            print(f"[chunk progress] processed_chunks={i}, current_keys={len(agg)}")

    rows = []
    for (td, sym), flags in agg.items():
        has_order, has_trade, has_tick = flags
        rows.append({
            "trade_date": td,
            "symbol": sym,
            "has_order": has_order,
            "has_trade": has_trade,
            "has_tick": has_tick,
        })

    avail = pd.DataFrame(rows)
    avail["is_valid_step3_sample"] = (
        (avail["has_order"] == 1)
        & (avail["has_trade"] == 1)
        & (avail["has_tick"] == 1)
    ).astype("int8")

    avail["exchange"] = avail["symbol"].map(detect_exchange_from_symbol)
    avail = avail.sort_values(["trade_date", "symbol"], kind="stable").reset_index(drop=True)
    return avail


def get_valid_trade_dates_from_availability(
    avail: pd.DataFrame,
    min_valid_symbols_per_day: int = 50,
) -> list[str]:
    """
    从 availability 表中，提取 valid sample 支持足够的 trade_date。
    只有当天 is_valid_step3_sample 数量 >= min_valid_symbols_per_day 的日期才保留。
    """
    required_cols = ["trade_date", "is_valid_step3_sample"]
    missing = [c for c in required_cols if c not in avail.columns]
    if missing:
        raise ValueError(f"availability table missing columns: {missing}")

    daily_valid = (
        avail.groupby("trade_date", as_index=False)["is_valid_step3_sample"]
        .sum()
        .rename(columns={"is_valid_step3_sample": "n_valid_symbols"})
    )

    daily_valid["trade_date"] = daily_valid["trade_date"].astype("string")
    daily_valid = daily_valid.sort_values("trade_date", kind="stable").reset_index(drop=True)

    keep = daily_valid["n_valid_symbols"] >= min_valid_symbols_per_day
    out = daily_valid.loc[keep, "trade_date"].astype(str).tolist()
    return out

def build_availability_table_from_file_index_streaming(
    file_index_csv: str | Path,
    progress_every: int = 1_000_000,
) -> pd.DataFrame:
    """
    用 Python csv 流式读取 file index，构造 availability 表。
    输入 CSV 每行格式：
        trade_date,table_name,symbol

    输出 DataFrame 列：
        trade_date, symbol, has_order, has_trade, has_tick,
        is_valid_step3_sample, exchange
    """
    import csv
    from collections import defaultdict

    file_index_csv = Path(file_index_csv)
    if not file_index_csv.exists():
        raise FileNotFoundError(f"file_index_csv not found: {file_index_csv}")

    # key=(trade_date, symbol), value=[has_order, has_trade, has_tick]
    agg = defaultdict(lambda: [0, 0, 0])

    line_count = 0
    kept_count = 0
    bad_count = 0

    with open(file_index_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            line_count += 1

            if len(row) != 3:
                bad_count += 1
                continue

            trade_date, table_name, symbol = row

            trade_date = trade_date.strip()
            table_name = table_name.strip().lower()
            symbol = symbol.strip().zfill(6)

            if len(trade_date) != 8 or not trade_date.isdigit():
                bad_count += 1
                continue

            if len(symbol) != 6 or not symbol.isdigit():
                bad_count += 1
                continue

            if table_name not in {"order", "trade", "tick"}:
                bad_count += 1
                continue

            key = (trade_date, symbol)
            if table_name == "order":
                agg[key][0] = 1
            elif table_name == "trade":
                agg[key][1] = 1
            elif table_name == "tick":
                agg[key][2] = 1

            kept_count += 1

            if line_count % progress_every == 0:
                print(
                    f"[stream progress] lines={line_count:,}, kept={kept_count:,}, "
                    f"bad={bad_count:,}, keys={len(agg):,}"
                )

    rows = []
    for (trade_date, symbol), flags in agg.items():
        has_order, has_trade, has_tick = flags
        rows.append({
            "trade_date": trade_date,
            "symbol": symbol,
            "has_order": has_order,
            "has_trade": has_trade,
            "has_tick": has_tick,
        })

    avail = pd.DataFrame(rows)

    if len(avail) == 0:
        # 返回空表也保留标准列，避免后面报错难看
        avail = pd.DataFrame(columns=[
            "trade_date", "symbol", "has_order", "has_trade", "has_tick",
            "is_valid_step3_sample", "exchange"
        ])
        return avail

    avail["has_order"] = pd.to_numeric(avail["has_order"], errors="coerce").fillna(0).astype("int8")
    avail["has_trade"] = pd.to_numeric(avail["has_trade"], errors="coerce").fillna(0).astype("int8")
    avail["has_tick"] = pd.to_numeric(avail["has_tick"], errors="coerce").fillna(0).astype("int8")

    avail["is_valid_step3_sample"] = (
        (avail["has_order"] == 1)
        & (avail["has_trade"] == 1)
        & (avail["has_tick"] == 1)
    ).astype("int8")

    avail["exchange"] = avail["symbol"].map(detect_exchange_from_symbol)

    avail = avail.sort_values(["trade_date", "symbol"], kind="stable").reset_index(drop=True)

    print(
        f"[stream done] total_lines={line_count:,}, kept={kept_count:,}, "
        f"bad={bad_count:,}, final_rows={len(avail):,}"
    )

    return avail

def build_universe_manifest_from_candidates(
    candidate_csv: str | Path,
    out_csv: str | Path,
    n_per_day: int = 80,
    n_sh: int = 40,
    n_sz: int = 40,
    random_state: int = 42,
) -> pd.DataFrame:
    candidate_csv = Path(candidate_csv)
    out_csv = Path(out_csv)

    df = pd.read_csv(candidate_csv, dtype={"trade_date": str, "symbol": str, "exchange": str})
    df["trade_date"] = df["trade_date"].astype(str)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["exchange"] = df["exchange"].astype(str)

    rng = np.random.default_rng(random_state)
    rows = []

    for td, sub in df.groupby("trade_date", sort=True):
        sh = sub.loc[sub["exchange"] == "SH", "symbol"].drop_duplicates().tolist()
        sz = sub.loc[sub["exchange"] == "SZ", "symbol"].drop_duplicates().tolist()

        sh = sorted(sh)
        sz = sorted(sz)

        take_sh = min(n_sh, len(sh))
        take_sz = min(n_sz, len(sz))

        pick_sh = list(rng.choice(sh, size=take_sh, replace=False)) if take_sh > 0 else []
        pick_sz = list(rng.choice(sz, size=take_sz, replace=False)) if take_sz > 0 else []

        picked = set(pick_sh + pick_sz)

        remain_needed = n_per_day - len(picked)
        if remain_needed > 0:
            remain_pool = sorted(set(sub["symbol"].tolist()) - picked)
            if len(remain_pool) > 0:
                extra_take = min(remain_needed, len(remain_pool))
                extra_pick = list(rng.choice(remain_pool, size=extra_take, replace=False))
                picked.update(extra_pick)

        picked = sorted(picked)

        for sym in picked:
            ex = "SH" if sym.startswith("6") else ("SZ" if sym.startswith(("0", "3")) else "UNKNOWN")
            rows.append({
                "trade_date": td,
                "symbol": sym,
                "exchange": ex,
            })

    out = pd.DataFrame(rows).sort_values(["trade_date", "symbol"], kind="stable").reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    return out

def build_step3_sample_list_from_universe(
    universe_csv: str | Path,
    out_csv: str | Path,
    n_samples: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    universe_csv = Path(universe_csv)
    out_csv = Path(out_csv)

    df = pd.read_csv(
        universe_csv,
        dtype={"trade_date": str, "symbol": str, "exchange": str},
    )

    df["trade_date"] = df["trade_date"].astype(str)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["exchange"] = df["exchange"].astype(str)

    if len(df) < n_samples:
        raise ValueError(
            f"Universe rows ({len(df)}) < requested n_samples ({n_samples})"
        )

    sample_df = (
        df.sample(n=n_samples, random_state=random_state, replace=False)
          .sort_values(["trade_date", "symbol"], kind="stable")
          .reset_index(drop=True)
    )

    sample_df.to_csv(out_csv, index=False)
    return sample_df

if __name__ == "__main__":
    set_global_seed()

    RAW_ROOT = "/data105/Level2_Data/ftr_data"
    AVAILABILITY_CSV = "/home/bu-yuting/新建文件夹/行为单元/step3_availability_recent1y_sorted.csv"
    SELECTED_TRADE_DATES_CSV = "/home/bu-yuting/新建文件夹/行为单元/step3_selected_trade_dates_recent1y.csv"
    VALID_CANDIDATES_CSV = "/home/bu-yuting/新建文件夹/行为单元/step3_valid_candidates_selected_dates.csv"
    UNIVERSE_MANIFEST_CSV = "/home/bu-yuting/新建文件夹/行为单元/step3_universe_manifest_recent1y.csv"

    print("\n" + "=" * 120)
    print("STEP3 CONFIG PREVIEW")
    print("=" * 120)
    print(f"RECENT_WINDOW_MONTHS = {RECENT_WINDOW_MONTHS}")
    print(f"UNIVERSE_TARGET_MAX_TRADE_DATES = {UNIVERSE_TARGET_MAX_TRADE_DATES}")
    print(f"UNIVERSE_TARGET_MAX_SYMBOLS_PER_DATE = {UNIVERSE_TARGET_MAX_SYMBOLS_PER_DATE}")
    print(f"UNIVERSE_TARGET_MAX_ROWS = {UNIVERSE_TARGET_MAX_ROWS}")
    print(f"STEP3_TARGET_N_SAMPLES = {STEP3_TARGET_N_SAMPLES}")

    # =============================================================================
    # 1) 基于 raw_root 看最近 1 年日期窗口（只是审计，不参与 availability 构造）
    # =============================================================================
    all_trade_dates = list_all_trade_dates_from_raw_root(RAW_ROOT)
    recent_trade_dates = get_recent_trade_dates_from_raw_root(
        raw_root=RAW_ROOT,
        recent_months=RECENT_WINDOW_MONTHS,
    )
    selected_trade_dates = select_trade_dates_evenly_from_recent_window(
        recent_trade_dates=recent_trade_dates,
        max_trade_dates=UNIVERSE_TARGET_MAX_TRADE_DATES,
    )

    print(f"\nall_trade_dates count = {len(all_trade_dates)}")
    if len(all_trade_dates) > 0:
        print(f"all_trade_dates min/max = {all_trade_dates[0]} / {all_trade_dates[-1]}")

    print(f"\nrecent_trade_dates count = {len(recent_trade_dates)}")
    if len(recent_trade_dates) > 0:
        print(f"recent_trade_dates min/max = {recent_trade_dates[0]} / {recent_trade_dates[-1]}")
        print(f"recent_trade_dates head = {recent_trade_dates[:5]}")
        print(f"recent_trade_dates tail = {recent_trade_dates[-5:]}")

    print(f"\nselected_trade_dates count = {len(selected_trade_dates)}")
    if len(selected_trade_dates) > 0:
        print(f"selected_trade_dates min/max = {selected_trade_dates[0]} / {selected_trade_dates[-1]}")
        print(f"selected_trade_dates head = {selected_trade_dates[:10]}")
        print(f"selected_trade_dates tail = {selected_trade_dates[-10:]}")

    # 先把 raw-root 口径选出的 120 个日期落盘，便于后续核对
    selected_trade_dates_df = pd.DataFrame({"trade_date": selected_trade_dates})
    selected_trade_dates_df.to_csv(SELECTED_TRADE_DATES_CSV, index=False)
    print("\n[selected trade dates saved]")
    print(SELECTED_TRADE_DATES_CSV)

    # =============================================================================
    # 2) 直接读取已经预先构造好的 availability manifest
    # =============================================================================
    print("\n" + "=" * 120)
    print("LOAD AVAILABILITY TABLE (PREBUILT)")
    print("=" * 120)

    avail = pd.read_csv(
        AVAILABILITY_CSV,
        dtype={
            "trade_date": str,
            "symbol": str,
            "has_order": int,
            "has_trade": int,
            "has_tick": int,
        }
    )

    avail["trade_date"] = avail["trade_date"].astype(str)
    avail["symbol"] = avail["symbol"].astype(str).str.zfill(6)

    avail["is_valid_step3_sample"] = (
        (avail["has_order"] == 1)
        & (avail["has_trade"] == 1)
        & (avail["has_tick"] == 1)
    ).astype("int8")

    avail["exchange"] = avail["symbol"].map(detect_exchange_from_symbol)

    print(f"availability shape = {avail.shape}")
    print(f"valid sample count = {int(avail['is_valid_step3_sample'].sum())}")

    print("\n[head]")
    print(avail.head(10))

    print("\n[exchange counts]")
    print(avail["exchange"].value_counts(dropna=False))

    # =============================================================================
    # 3) 基于 availability 再做一次正式的 trade_date 筛选
    #    口径更严谨：只保留 valid sample 支持足够的日期
    # =============================================================================
    valid_trade_dates_from_avail = get_valid_trade_dates_from_availability(
        avail,
        min_valid_symbols_per_day=50,
    )

    selected_trade_dates_from_avail = select_trade_dates_evenly_from_recent_window(
        recent_trade_dates=valid_trade_dates_from_avail,
        max_trade_dates=UNIVERSE_TARGET_MAX_TRADE_DATES,
    )

    print("\n" + "=" * 120)
    print("TRADE DATE SELECTION FROM AVAILABILITY")
    print("=" * 120)

    print(f"valid_trade_dates_from_avail count = {len(valid_trade_dates_from_avail)}")
    if len(valid_trade_dates_from_avail) > 0:
        print(
            f"valid_trade_dates_from_avail min/max = "
            f"{valid_trade_dates_from_avail[0]} / {valid_trade_dates_from_avail[-1]}"
        )
        print(f"valid_trade_dates_from_avail head = {valid_trade_dates_from_avail[:5]}")
        print(f"valid_trade_dates_from_avail tail = {valid_trade_dates_from_avail[-5:]}")

    print(f"\nselected_trade_dates_from_avail count = {len(selected_trade_dates_from_avail)}")
    if len(selected_trade_dates_from_avail) > 0:
        print(
            f"selected_trade_dates_from_avail min/max = "
            f"{selected_trade_dates_from_avail[0]} / {selected_trade_dates_from_avail[-1]}"
        )
        print(f"selected_trade_dates_from_avail head = {selected_trade_dates_from_avail[:10]}")
        print(f"selected_trade_dates_from_avail tail = {selected_trade_dates_from_avail[-10:]}")

    # 用 availability 口径的正式日期覆盖保存
    pd.DataFrame({"trade_date": selected_trade_dates_from_avail}).to_csv(
        SELECTED_TRADE_DATES_CSV,
        index=False
    )
    print("\n[selected trade dates overwritten by availability-based selection]")
    print(SELECTED_TRADE_DATES_CSV)

    # =============================================================================
    # 4) 从 availability 中提取：选中日期 + valid sample 的候选池
    # =============================================================================
    candidate_df = avail[
        avail["trade_date"].isin(selected_trade_dates_from_avail)
        & (avail["is_valid_step3_sample"] == 1)
    ][["trade_date", "symbol", "exchange"]].copy()

    candidate_df = candidate_df.sort_values(["trade_date", "symbol"], kind="stable").reset_index(drop=True)
    candidate_df.to_csv(VALID_CANDIDATES_CSV, index=False)

    print("\n" + "=" * 120)
    print("VALID CANDIDATES ON SELECTED DATES")
    print("=" * 120)
    print(f"candidate_df shape = {candidate_df.shape}")
    print(candidate_df.head(10))

    candidate_daily_counts = candidate_df.groupby("trade_date").size()
    print("\n[candidate daily count summary]")
    print(candidate_daily_counts.describe())

    print("\n[candidate exchange counts]")
    print(candidate_df["exchange"].value_counts(dropna=False))

    # =============================================================================
    # 5) 生成正式 9600 universe manifest
    #    当前主方案：每个 trade_date 取 80 个，尽量 SH 40 + SZ 40
    # =============================================================================
    universe_manifest = build_universe_manifest_from_candidates(
        candidate_csv=VALID_CANDIDATES_CSV,
        out_csv=UNIVERSE_MANIFEST_CSV,
        n_per_day=80,
        n_sh=40,
        n_sz=40,
        random_state=42,
    )

    print("\n" + "=" * 120)
    print("UNIVERSE MANIFEST")
    print("=" * 120)
    print(f"universe_manifest shape = {universe_manifest.shape}")
    print(universe_manifest.head(10))

    daily_counts = universe_manifest.groupby("trade_date").size()
    print("\n[daily count summary]")
    print(daily_counts.describe())

    print("\n[exchange counts]")
    print(universe_manifest["exchange"].value_counts(dropna=False))

    print("\n[universe manifest saved]")
    print(UNIVERSE_MANIFEST_CSV)

    STEP3_SAMPLE_LIST_CSV = "/home/bu-yuting/新建文件夹/行为单元/step3_sample_list_recent1y_1000.csv"

    step3_sample_list = build_step3_sample_list_from_universe(
        universe_csv=UNIVERSE_MANIFEST_CSV,
        out_csv=STEP3_SAMPLE_LIST_CSV,
        n_samples=STEP3_TARGET_N_SAMPLES,
        random_state=42,
    )

    print("\n" + "=" * 120)
    print("STEP3 SAMPLE LIST")
    print("=" * 120)
    print(f"step3_sample_list shape = {step3_sample_list.shape}")
    print(step3_sample_list.head(10))

    sample_daily_counts = step3_sample_list.groupby("trade_date").size()
    print("\n[sample daily count summary]")
    print(sample_daily_counts.describe())

    print("\n[sample exchange counts]")
    print(step3_sample_list["exchange"].value_counts(dropna=False))

    print("\n[step3 sample list saved]")
    print(STEP3_SAMPLE_LIST_CSV)

    audited_full_df = preview_artifact_audit_for_sample_list(
        sample_list_csv=STEP3_FULL_SAMPLE_LIST_CSV,
        audited_out_csv=STEP3_AUDITED_FULL_CSV,
        default_step1_lookup=KNOWN_STEP1_ARTIFACT_LOOKUP,
        default_step2_lookup=KNOWN_STEP2_ARTIFACT_LOOKUP,
        preview_n=30,
    )

    built_full_df = build_step1_step2_artifacts_for_sample_list(
        audited_sample_list_csv=STEP3_AUDITED_FULL_CSV,
        out_csv=STEP3_BUILT_FULL_CSV,
        n_build=1000,
        rebuild_step1=True,
        verbose=False,
    )

    STEP3_BATCH_OUT_DIR = "/home/bu-yuting/新建文件夹/step3_cross_sample_validation/recent1y_1000samples"

    print("\n" + "=" * 120)
    print("RUN STEP3 BATCH FROM SAMPLE LIST")
    print("=" * 120)

    batch_res = run_step3_batch_from_built_sample_list(
        built_sample_list_csv=STEP3_BUILT_FULL_CSV,
        out_csv=STEP3_BATCH_OUT_CSV,
        out_json=STEP3_BATCH_OUT_JSON,
        limit=None,
        device="cpu",
    )

    print("\n" + "=" * 120)
    print("STEP3 BATCH RESULT")
    print("=" * 120)

    if isinstance(batch_res, dict):
        print("[batch result keys]")
        print(list(batch_res.keys()))

        summary_path = Path(STEP3_BATCH_OUT_DIR) / "batch_summary.json"
        print("\n[expected summary path]")
        print(summary_path)

        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_obj = json.load(f)

            print("\n[batch summary]")
            for k, v in summary_obj.items():
                print(f"{k}: {v}")