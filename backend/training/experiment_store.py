"""實驗層與逐筆結果的資料庫存取（hermesnote DB，asyncpg）。API 與 worker 共用。

表：model_experiments、model_experiment_arms、model_sample_frames(_points)、model_prediction_records、
model_evaluations、model_holdout_usage、model_upstream_folds（migrations/2026-11-16_m2_m7_*.sql）。
大量資料（框架時間點、逐筆結果）一律用 COPY 串流寫入。
"""

import json
import uuid
from datetime import date, datetime

import numpy as np
import pandas as pd

from services.hermesnote_db import get_conn
from training import fence, json_safe
from training.provenance import PIPELINE_VERSION

COPY_CHUNK = 20000


def _u(v):
    return None if v is None else (v if isinstance(v, uuid.UUID) else uuid.UUID(str(v)))


def _j(v):
    return json.loads(v) if isinstance(v, str) else v


def _dt(v):
    if v is None:
        return None
    if isinstance(v, np.datetime64):
        return None if np.isnat(v) else pd.Timestamp(v).to_pydatetime()
    if isinstance(v, str):
        return pd.Timestamp(v).to_pydatetime()
    return v


def _row(r) -> dict:
    out = {}
    for k, v in dict(r).items():
        if isinstance(v, uuid.UUID):
            v = str(v)
        elif isinstance(v, (datetime, date)):
            v = v.isoformat()
        elif k in ("settings", "spec", "model_seeds", "definition", "counts", "evaluations", "excluded_reasons",
                   "detail", "proba", "branches", "upstream", "graph_spec", "gate_check", "result", "warnings"):
            v = _j(v)
        out[k] = v
    return out


# ── 實驗與組別 ───────────────────────────────────────────────────────────
async def create_experiment(name: str, description: str | None, settings: dict, created_by: str | None = None) -> str:
    conn = await get_conn()
    try:
        row = await conn.fetchrow(
            "INSERT INTO model_experiments (name, description, settings, pipeline_version, created_by) "
            "VALUES ($1,$2,$3,$4,$5) RETURNING id",
            name, description, json_safe.dumps(settings), PIPELINE_VERSION, created_by)
        return str(row["id"])
    finally:
        await conn.close()


async def list_experiments() -> list[dict]:
    conn = await get_conn()
    try:
        rows = await conn.fetch(
            """SELECT e.id, e.name, e.description, e.settings, e.pipeline_version, e.created_at, e.updated_at,
                      (SELECT count(*) FROM model_experiment_arms a WHERE a.experiment_id = e.id) AS n_arms,
                      (SELECT count(*) FROM model_training_jobs j WHERE j.experiment_id = e.id) AS n_jobs
               FROM model_experiments e ORDER BY e.created_at DESC""")
        return [_row(r) for r in rows]
    finally:
        await conn.close()


async def get_experiment(exp_id: str) -> dict | None:
    conn = await get_conn()
    try:
        row = await conn.fetchrow("SELECT * FROM model_experiments WHERE id = $1", _u(exp_id))
        if row is None:
            return None
        exp = _row(row)
        exp["arms"] = [_row(r) for r in await conn.fetch(
            "SELECT * FROM model_experiment_arms WHERE experiment_id = $1 ORDER BY created_at", _u(exp_id))]
        exp["frames"] = [_row(r) for r in await conn.fetch(
            "SELECT id, phase, version, status, definition, hash, counts, n_points, created_at, ready_at "
            "FROM model_sample_frames WHERE experiment_id = $1 ORDER BY phase, version", _u(exp_id))]
        exp["jobs"] = [_row(r) for r in await conn.fetch(
            """SELECT id, status, phase, parent_job_id, arm_id, frame_id, model_seed, split_seed, job_type,
                      pipeline_version, warnings, gate_check, requested_by, error, created_at, finished_at
               FROM model_training_jobs WHERE experiment_id = $1 ORDER BY created_at""", _u(exp_id))]
        return exp
    finally:
        await conn.close()


async def update_settings(exp_id: str, settings: dict) -> None:
    conn = await get_conn()
    try:
        await conn.execute("UPDATE model_experiments SET settings = $2, updated_at = now() WHERE id = $1",
                           _u(exp_id), json_safe.dumps(settings))
    finally:
        await conn.close()


async def create_arm(exp_id: str, name: str, spec: dict, model_seeds: list) -> str:
    conn = await get_conn()
    try:
        row = await conn.fetchrow(
            "INSERT INTO model_experiment_arms (experiment_id, name, spec, model_seeds) VALUES ($1,$2,$3,$4) RETURNING id",
            _u(exp_id), name, json.dumps(spec), json.dumps(model_seeds))
        return str(row["id"])
    finally:
        await conn.close()


async def get_arm(arm_id: str) -> dict | None:
    conn = await get_conn()
    try:
        row = await conn.fetchrow("SELECT * FROM model_experiment_arms WHERE id = $1", _u(arm_id))
        return _row(row) if row else None
    finally:
        await conn.close()


# ── 樣本框架 ─────────────────────────────────────────────────────────────
async def latest_frame(exp_id: str, phase: int, create: bool = True) -> dict | None:
    conn = await get_conn()
    try:
        row = await conn.fetchrow(
            "SELECT * FROM model_sample_frames WHERE experiment_id = $1 AND phase = $2 ORDER BY version DESC LIMIT 1",
            _u(exp_id), phase)
        if row is None and create:
            row = await conn.fetchrow(
                "INSERT INTO model_sample_frames (experiment_id, phase, version) VALUES ($1,$2,1) RETURNING *",
                _u(exp_id), phase)
        return _row(row) if row else None
    finally:
        await conn.close()


async def new_frame_version(exp_id: str, phase: int, note: str | None = None) -> dict:
    conn = await get_conn()
    try:
        v = await conn.fetchval("SELECT COALESCE(max(version), 0) + 1 FROM model_sample_frames "
                                "WHERE experiment_id = $1 AND phase = $2", _u(exp_id), phase)
        row = await conn.fetchrow(
            "INSERT INTO model_sample_frames (experiment_id, phase, version, definition) VALUES ($1,$2,$3,$4) RETURNING *",
            _u(exp_id), phase, v, json.dumps({"note": note}) if note else None)
        return _row(row)
    finally:
        await conn.close()


async def get_frame(frame_id: str) -> dict | None:
    conn = await get_conn()
    try:
        row = await conn.fetchrow("SELECT * FROM model_sample_frames WHERE id = $1", _u(frame_id))
        return _row(row) if row else None
    finally:
        await conn.close()


async def save_frame(frame_id: str, frame) -> None:
    """worker：第一次用到這個框架時寫入時間點並標記 ready（之後的任務只核對雜湊）。"""
    conn = await get_conn()
    try:
        async with conn.transaction():
            await fence.check_async(conn)  # 目前任務的所有權（見 training/fence.py）
            await conn.execute("DELETE FROM model_sample_frame_points WHERE frame_id = $1", _u(frame_id))
            fid = _u(frame_id)

            def gen():
                for i in range(len(frame)):
                    yield (fid, _dt(frame.ts[i]), _dt(frame.tau[i]), _dt(frame.target_ts[i]), _dt(frame.target_avail[i]),
                           str(frame.status[i]), frame.reason[i])
            await conn.copy_records_to_table(
                "model_sample_frame_points", records=gen(),
                columns=["frame_id", "ts", "decision_available_ts", "target_ts", "target_available_ts", "status", "reason"])
            await conn.execute(
                "UPDATE model_sample_frames SET status = 'ready', definition = $2, hash = $3, counts = $4, n_points = $5, "
                "ready_at = now() WHERE id = $1",
                fid, json_safe.dumps(frame.definition), frame.hash, json_safe.dumps(frame.counts()), len(frame))
    finally:
        await conn.close()


async def frame_points(frame_id: str, status: str | None = None, cursor: str | None = None, limit: int = 1000) -> list:
    conn = await get_conn()
    try:
        rows = await conn.fetch(
            "SELECT ts, decision_available_ts, target_ts, target_available_ts, status, reason FROM model_sample_frame_points "
            "WHERE frame_id = $1 AND ($2::text IS NULL OR status = $2) AND ($3::timestamp IS NULL OR ts > $3) "
            "ORDER BY ts LIMIT $4", _u(frame_id), status, _dt(cursor), limit)
        return [_row(r) for r in rows]
    finally:
        await conn.close()


# ── 逐筆結果、評估、使用紀錄、串接折次 ──────────────────────────────────────
async def write_prediction_records(rows_iter) -> int:
    from training.v2_records import COLUMNS

    conn = await get_conn()
    n = 0
    try:
        async with conn.transaction():  # 整批一個交易（中途被拒絕不會留下部分逐筆結果）
            await fence.check_async(conn)
            chunk = []
            for r in rows_iter:
                chunk.append(r)
                if len(chunk) >= COPY_CHUNK:
                    await conn.copy_records_to_table("model_prediction_records", records=chunk, columns=COLUMNS)
                    n += len(chunk)
                    chunk = []
            if chunk:
                await conn.copy_records_to_table("model_prediction_records", records=chunk, columns=COLUMNS)
                n += len(chunk)
        return n
    finally:
        await conn.close()


async def write_evaluation(job_id, node_id, phase, dataset, weights, evaluations, stats: dict) -> None:
    conn = await get_conn()
    try:
        async with conn.transaction():
            await fence.check_async(conn, job_id)
            await conn.execute(
                "INSERT INTO model_evaluations (job_id, node_id, phase, dataset, weights, evaluations, n_included, n_excluded, "
                "excluded_reasons, cutoff_ts) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)",
                _u(job_id), node_id, phase, dataset, weights, json_safe.dumps(evaluations), stats.get("n_included"),
                stats.get("n_unmatured", 0) + stats.get("n_skipped", 0), json_safe.dumps(stats), _dt(stats.get("cutoff_ts")))
    finally:
        await conn.close()


async def write_holdout_usage(kind: str, *, job_id=None, parent_job_id=None, experiment_id=None, arm_id=None,
                              start=None, end=None, detail: dict | None = None) -> None:
    conn = await get_conn()
    try:
        async with conn.transaction():
            await fence.check_async(conn, job_id)
            await conn.execute(
                "INSERT INTO model_holdout_usage (experiment_id, arm_id, job_id, parent_job_id, kind, period_start, period_end, detail) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                _u(experiment_id), _u(arm_id), _u(job_id), _u(parent_job_id), kind,
                date.fromisoformat(start) if isinstance(start, str) else start,
                date.fromisoformat(end) if isinstance(end, str) else end, json_safe.dumps(detail or {}))
    finally:
        await conn.close()


async def write_upstream_folds(job_id, node_id: str, info: dict, frame_hash: str) -> None:
    conn = await get_conn()
    try:
        async with conn.transaction():
            await fence.check_async(conn, job_id)
            for f in info["folds"]:
                await conn.execute(
                    """INSERT INTO model_upstream_folds (job_id, node_id, fold, strategy, frame_hash, n_train, train_start,
                           train_end, train_max_label_end, before_max_le, after_min_ws, purge_start, purge_end, predict_start,
                           predict_end, n_pred, fingerprint, detail)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)""",
                    _u(job_id), node_id, int(f["fold"]), f["strategy"], frame_hash, f.get("n_train"),
                    _dt(f.get("train_start")), _dt(f.get("train_end")), _dt(f.get("train_max_label_end")),
                    _dt(f.get("before_max_le")), _dt(f.get("after_min_ws")), _dt(f.get("purge_start")),
                    _dt(f.get("purge_end")), _dt(f.get("predict_start")), _dt(f.get("predict_end")), f.get("n_pred"),
                    f.get("fingerprint"),
                    json_safe.dumps({"consumers": info["consumers"], "policy": info["policy"], "purge_sec": info["purge_sec"],
                                     "skipped": f.get("skipped"), "final": f.get("final")}))
    finally:
        await conn.close()


async def list_records(job_id: str, node_id: str | None = None, dataset: str | None = None, weights: str | None = None,
                       cursor: int | None = None, limit: int = 1000) -> list:
    conn = await get_conn()
    try:
        rows = await conn.fetch(
            """SELECT id, node_id, dataset, weights, model_seed, sample_ts, decision_available_ts, window_start_ts, target_ts,
                      target_available_ts, y_true, y_pred, proba, included, exclusion_reason, branches, upstream, frame_hash, phase
               FROM model_prediction_records
               WHERE job_id = $1 AND ($2::text IS NULL OR node_id = $2) AND ($3::text IS NULL OR dataset = $3)
                 AND ($4::text IS NULL OR weights = $4) AND ($5::bigint IS NULL OR id > $5)
               ORDER BY id LIMIT $6""", _u(job_id), node_id, dataset, weights, cursor, limit)
        return [_row(r) for r in rows]
    finally:
        await conn.close()


async def records_for_compare(job_id: str, dataset: str, weights: str) -> list:
    conn = await get_conn()
    try:
        return [dict(r) for r in await conn.fetch(
            "SELECT sample_ts, y_true, y_pred, proba, frame_hash FROM model_prediction_records "
            "WHERE job_id = $1 AND dataset = $2 AND weights = $3 AND included ORDER BY sample_ts",
            _u(job_id), dataset, weights)]
    finally:
        await conn.close()


async def list_evaluations(job_id: str) -> list:
    conn = await get_conn()
    try:
        return [_row(r) for r in await conn.fetch(
            "SELECT * FROM model_evaluations WHERE job_id = $1 ORDER BY created_at, weights", _u(job_id))]
    finally:
        await conn.close()


async def holdout_usage(exp_id: str | None = None, job_id: str | None = None) -> list:
    conn = await get_conn()
    try:
        return [_row(r) for r in await conn.fetch(
            "SELECT * FROM model_holdout_usage WHERE ($1::uuid IS NULL OR experiment_id = $1) "
            "AND ($2::uuid IS NULL OR job_id = $2 OR parent_job_id = $2) ORDER BY created_at DESC LIMIT 500",
            _u(exp_id), _u(job_id))]
    finally:
        await conn.close()


async def upstream_folds(job_id: str) -> list:
    conn = await get_conn()
    try:
        return [_row(r) for r in await conn.fetch(
            "SELECT * FROM model_upstream_folds WHERE job_id = $1 ORDER BY node_id, fold", _u(job_id))]
    finally:
        await conn.close()


async def audit_inputs(job_id: str) -> dict:
    """串接核對用的原始紀錄（只讀資料庫，不用訓練程式的判定）。"""
    conn = await get_conn()
    try:
        recs = [dict(r) for r in await conn.fetch(
            "SELECT node_id, dataset, weights, sample_ts, decision_available_ts, window_start_ts, target_available_ts, "
            "upstream, frame_hash FROM model_prediction_records WHERE job_id = $1 AND upstream IS NOT NULL", _u(job_id))]
        folds = [dict(r) for r in await conn.fetch("SELECT * FROM model_upstream_folds WHERE job_id = $1", _u(job_id))]
        return {"records": recs, "folds": folds}
    finally:
        await conn.close()
