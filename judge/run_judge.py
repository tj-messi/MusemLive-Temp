#!/usr/bin/env python3
"""Portable lenient-v3 point-based Judge for MuseumLive-Bench-450."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


SLOT_SLACK = 3.0
JUDGE_PROFILE = "lenient_v3"
W_ACTION, W_POINTS, PEN_FORBID, PEN_CONTRA = 0.4, 0.6, 0.75, 0.25
CREDIT = {"hit": 1.0, "partial": 0.8, "miss": 0.0, "contradict": 0.0}
ACT = {"correct": 1.0, "partial": 0.7, "wrong": 0.0}
W_CORR, W_CC_ACTION, W_CC_POINTS, PEN_REINFORCE = 0.4, 0.25, 0.35, 0.5
CORR_CREDIT = {
    "must_correct": {"corrected": 1.0, "partial": 0.6, "not_corrected": 0.0, "reinforced": 0.0},
    "must_follow": {"corrected": 1.0, "partial": 0.5, "not_corrected": 0.0, "reinforced": 0.0},
}
SPENT = {"usd": 0.0, "calls": 0}
LOCK = threading.Lock()

MATCH_SYSTEM = """你是流式视频交互 benchmark 的得分点匹配 Judge。只做匹配，不打总分。
输入：题型、期望动作（SPEAK=直接回答/提醒，CLARIFY=先反问澄清，WAIT=不该回答）、参考答案、得分点列表、禁止说法列表、
本轮用户语音的文字、模型本轮回答，以及此前对话（仅供理解上下文，评的只是"模型本轮回答"）。
trigger=visual_event 时用户此刻没有说话：这是画面/声音触发的时机，"模型本轮回答"是模型在该时间窗内主动说的话（空 = 保持沉默），
期望 SPEAK 表示模型应主动提醒/告知，期望 WAIT 表示模型应保持安静；此前对话里有用户先前的要求。

判断：
1. action：只看模型本轮回应的"类型"是否符合期望动作，不看内容对错（内容对错由 points 体现）。
   - SPEAK：针对用户这句话（或画面触发时针对用户先前的要求）给出了回答/提醒（哪怕答错了、对象搞错了）= correct；回答了但同时夹杂拒答或大段跑题 = partial；
     只说"好的/我会留意"之类确认、拒答（"视频里没有…"）、复述用户问题、或完全没回应用户这句话（答非所问、自顾自描述别的东西）= wrong。
   - CLARIFY：提出了针对性的澄清问句 = correct；有澄清意味但没形成问句或没点出要区分的对象 = partial；直接给出答案/猜测 = wrong。
   - WAIT：沉默或只做极简确认 = correct；说了实质内容 = wrong。
   observed 填模型实际动作：answer / clarify / ack_only / refuse / silent / other。
2. points：逐条判断得分点：
   - hit：回答表达出了该点的核心意思即可（同义、概括、换种说法、细节不全但关键对象/事实对上都算 hit，不要求逐字或面面俱到）；
   - partial：沾边但核心不完整（例如方向对但对象不具体、只说出一半关键信息、用了更笼统的说法但没有说错）；
   - miss：没有提到；
   - contradict：明确说了与该点相反或错误的内容（说错颜色/对象/数字、张冠李戴）；只是笼统或没提到不算 contradict。
   拿不准 hit 还是 partial 时判 hit；拿不准 partial 还是 miss 时判 partial。
   得分点里的 [音频]/[视觉] 只是来源标签，"对应转写"是证据，不要求模型逐字复述。
   人称不同（他/她/您/你/用户）不算错；只有把说话人或对象搞错、导致事实错误时才算 contradict。
   evidence 摘录模型回答中对应的原话（没有就留空）。
3. forbidden：逐条判断模型是否说出了该禁止说法（意思相同即算）。
只输出 JSON：{"action": "correct|partial|wrong", "observed": "...", "action_reason": "...",
"points": [{"id": "P1", "status": "hit|partial|miss|contradict", "evidence": "...", "why": "..."}],
"forbidden": [{"id": "F1", "violated": true|false, "evidence": "..."}]}"""

CORR_ADDENDUM = """

本题型是上下文纠错（context_correction），输入里有 correction_check。另外判断 correction，这是本题最重要的评分维度：
- mode=must_correct：用户此前说过一个错误理解（user_claim），之后没有自己更正。参考答案会明确指出用户之前的理解不对并给出正确事实。
  判断模型本轮回答是否像参考答案一样做了纠正：
  corrected = 明确指出用户之前的说法/判断不对（如"其实不是X，而是Y""您之前说的有误"），并给出与参考答案一致的正确事实；
  partial = 纠正不完整：只是在回答里用了正确说法却没点明用户之前错了，或点明了不对但没给出正确事实、给出的事实与参考答案不符；
  not_corrected = 没有触及用户的错误理解（只回答了本轮问题）；
  reinforced = 附和、默认或沿用了用户的错误理解。
- mode=must_follow：用户已经自己更正（user_correction）。判断模型是否以更正后的理解为准：
  corrected = 回答明确以更正后的事实为准（确认了用户的更正，或回答里用的是更正后的说法），与参考答案一致；
  partial = 回答没有体现更正内容，但也没有与更正后的事实冲突；
  reinforced = 回到或沿用了用户更正前的错误理解。
JSON 里增加："correction": {"status": "corrected|partial|not_corrected|reinforced", "evidence": "模型回答原话", "why": "..."}"""


def silent(response) -> bool:
    return response is None or not str(response).strip() or "SILENCE" in str(response).upper()


def strip_ref(point: str) -> str:
    return re.sub(r"\s*[（(]对应转写[:：].*?[)）]\s*$", "", point).strip()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def correction_ctx(result: dict, trigger_event_id: str | None):
    events = sorted(result.get("user_events") or [], key=lambda row: row["time_s"])
    trigger_time = next(
        (row["time_s"] for row in events if row["event_id"] == trigger_event_id),
        float("inf"),
    )
    previous = [row for row in events if row["time_s"] < trigger_time]
    claims = [row for row in previous if (row.get("state_op") or {}).get("op") == "claim"]
    corrections = [row for row in previous if (row.get("state_op") or {}).get("op") == "correct"]
    if not claims:
        return None
    context = {
        "mode": "must_follow" if corrections else "must_correct",
        "user_claim": {
            "utterance": claims[-1]["text"],
            "meaning": claims[-1]["state_op"].get("target", ""),
        },
    }
    if corrections:
        context["user_correction"] = {
            "utterance": corrections[-1]["text"],
            "meaning": corrections[-1]["state_op"].get("target", ""),
        }
    return context


def duplex_response(result: dict, slot: dict):
    low = float(slot["window"]["start_s"])
    high = float(slot["window"]["end_s"]) + SLOT_SLACK
    turns = [
        row
        for row in result.get("duplex_turns") or []
        if row["start_s"] <= high and row["end_s"] >= low
    ]
    history = [
        {"t": row["time_s"], "user": row.get("utterance", ""), "model": row.get("response", "")}
        for row in result["timeline"]
        if row.get("type") == "user_event" and float(row["time_s"]) < low
    ][-4:]
    return (
        " / ".join(row["text"] for row in turns),
        [{"start_s": row["start_s"], "end_s": row["end_s"]} for row in turns],
        history,
    )


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        payload = json.loads(path.read_text())
        for result in payload["results"]:
            user_events = {
                row.get("event_id"): row
                for row in result["timeline"]
                if row.get("type") == "user_event"
            }
            for wrapped in result["slots"]:
                slot = wrapped["slot"]
                trigger = slot["trigger"]
                row = {
                    "judge_id": f"{result['branch_key']}::{slot['slot_id']}",
                    "task_id": result["task_id"],
                    "branch_id": result["branch_id"],
                    "capability": result["capability"],
                    "slot_id": slot["slot_id"],
                    "expected_action": slot["expected_action"],
                    "trigger_kind": trigger["kind"],
                    "target_text": slot.get("target_text") or "",
                    "key_attributes": slot.get("key_attributes") or [],
                    "forbidden_claims": slot.get("forbidden_claims") or [],
                    "guidance_direction": slot.get("guidance_direction"),
                    "scorable": True,
                    "protocol": "duplex",
                }
                if result["capability"] == "context_correction":
                    row["correction_check"] = correction_ctx(result, trigger.get("event_id"))
                if trigger["kind"] != "user_query":
                    row["response"], row["duplex_turns"], row["history"] = duplex_response(result, slot)
                    row["utterance"] = ""
                else:
                    event = user_events.get(trigger.get("event_id"))
                    row["utterance"] = (event or {}).get("utterance", "")
                    row["response"] = (event or {}).get("response")
                    row["history"] = [
                        {"t": item["time_s"], "user": item.get("utterance", ""), "model": item.get("response", "")}
                        for item in wrapped.get("timeline_prefix") or []
                        if item.get("type") == "user_event"
                        and item.get("event_id") != trigger.get("event_id")
                    ][-4:]
                rows.append(row)
    return rows


def rubric_key(row: dict) -> str:
    value = [row["expected_action"], row["target_text"]]
    return hashlib.md5(json.dumps(value, ensure_ascii=False).encode()).hexdigest()[:16]


def core_key(row: dict) -> str:
    value = [row["utterance"], row["expected_action"], row["target_text"], row["key_attributes"]]
    return hashlib.md5(json.dumps(value, ensure_ascii=False).encode()).hexdigest()[:16]


def points_of(row: dict, rubric: dict) -> list[str]:
    if row["key_attributes"]:
        return list(row["key_attributes"])
    if row["expected_action"] in ("SPEAK", "CLARIFY") and row["target_text"]:
        return rubric.get(rubric_key(row)) or [row["target_text"]]
    return []


def core_of(row: dict, rubric: dict, points: list[str]) -> list[bool]:
    if row["key_attributes"] and len(row["key_attributes"]) >= 2:
        labels = rubric.get("core_" + core_key(row))
        if labels and len(labels) == len(points):
            return [label == "core" for label in labels]
    return [True] * len(points)


class Matcher:
    def __init__(self, model: str, extra: dict):
        self.model = model
        self.extra = extra
        self.url = os.environ.get("JUDGE_API_URL", "https://openrouter.ai/api/v1/chat/completions")
        self.key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("JUDGE_API_KEY")
        self.trust_env = os.environ.get("JUDGE_TRUST_ENV", "0") == "1"

    def call(self, messages: list[dict], max_tokens: int = 1500) -> dict:
        if not self.key:
            raise RuntimeError("non-silent outputs require OPENROUTER_API_KEY or JUDGE_API_KEY")
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        } | self.extra
        last_error = None
        for attempt in range(4):
            try:
                session = requests.Session()
                session.trust_env = self.trust_env
                response = session.post(
                    self.url,
                    headers={"Authorization": "Bearer " + self.key},
                    json=body,
                    timeout=(20, 240),
                )
                response.raise_for_status()
                payload = response.json()
                usage = payload.get("usage") or {}
                with LOCK:
                    SPENT["calls"] += 1
                    SPENT["usd"] += float(usage.get("cost") or 0)
                text = payload["choices"][0]["message"]["content"]
                text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
                return json.loads(text)
            except Exception as error:
                last_error = error
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"judge call failed: {str(last_error)[:300]}")


def match(row: dict, points: list[str], matcher: Matcher) -> dict:
    payload = {
        "capability": row["capability"],
        "slot_id": row["slot_id"],
        "expected_action": row["expected_action"],
        "trigger": row["trigger_kind"],
        "reference_answer": row["target_text"],
        "points": [{"id": f"P{index + 1}", "text": point} for index, point in enumerate(points)],
        "forbidden": [
            {"id": f"F{index + 1}", "text": point}
            for index, point in enumerate(row["forbidden_claims"])
        ],
        "previous_turns": row["history"],
        "user_utterance_this_turn": row["utterance"],
        "model_response_this_turn": row["response"],
    }
    if row.get("guidance_direction"):
        payload["guidance_direction"] = row["guidance_direction"]
    system = MATCH_SYSTEM
    if row.get("correction_check"):
        payload["correction_check"] = row["correction_check"]
        system += CORR_ADDENDUM
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    for _ in range(3):
        output = matcher.call(messages)
        if (
            isinstance(output, dict)
            and all(isinstance(item, dict) for item in (output.get("points") or []) + (output.get("forbidden") or []))
            and (output.get("correction") is None or isinstance(output.get("correction"), dict))
        ):
            return output
    raise RuntimeError("judge returned malformed structure 3 times")


def score(row: dict, points: list[str], output: dict, core: list[bool]):
    expected = row["expected_action"]
    if expected != "WAIT" and silent(row["response"]):
        return 0.0, {
            "action": "wrong",
            "observed": "silent",
            "coverage": 0.0,
            "n_points": len(points),
            "statuses": ["miss"] * len(points),
            "core": core,
            "forbidden_hits": 0,
            "contradict": 0,
            "note": "silent",
        }
    action_credit = ACT.get((output.get("action") or "").lower(), 0.0)
    labels = {item.get("id"): (item.get("status") or "miss").lower() for item in output.get("points") or []}
    statuses = [labels.get(f"P{index + 1}", "miss") for index in range(len(points))]
    core_values = [CREDIT.get(status, 0.0) for status, is_core in zip(statuses, core) if is_core]
    support_values = [CREDIT.get(status, 0.0) for status, is_core in zip(statuses, core) if not is_core]
    if not statuses:
        coverage = None
    elif support_values:
        coverage = 0.8 * (sum(core_values) / len(core_values)) + 0.2 * (
            sum(support_values) / len(support_values)
        )
    else:
        coverage = sum(core_values) / len(core_values)
    contradictions = sum(status == "contradict" for status in statuses)
    forbidden_hits = sum(bool(item.get("violated")) for item in output.get("forbidden") or [])
    correction_check = row.get("correction_check")
    correction_credit = None
    reinforced = 0
    if correction_check and expected != "WAIT":
        correction_status = ((output.get("correction") or {}).get("status") or "not_corrected").lower()
        correction_credit = CORR_CREDIT[correction_check["mode"]].get(correction_status, 0.0)
        reinforced = int(correction_status == "reinforced")
    if correction_credit is not None:
        base = 5.0 * (
            W_CORR * correction_credit
            + W_CC_ACTION * action_credit
            + W_CC_POINTS * (coverage if coverage is not None else action_credit)
        ) - PEN_REINFORCE * reinforced
    elif expected == "WAIT" or coverage is None:
        base = 5.0 * action_credit
    else:
        base = 5.0 * (W_ACTION * action_credit + W_POINTS * coverage)
    value = max(0.0, min(5.0, base - PEN_FORBID * forbidden_hits - PEN_CONTRA * contradictions))
    detail = {
        "action": output.get("action"),
        "observed": output.get("observed"),
        "coverage": None if coverage is None else round(coverage, 3),
        "n_points": len(points),
        "statuses": statuses,
        "core": core,
        "forbidden_hits": forbidden_hits,
        "contradict": contradictions,
    }
    if correction_credit is not None:
        detail.update(
            {
                "correction_mode": correction_check["mode"],
                "correction": (output.get("correction") or {}).get("status"),
                "correction_credit": correction_credit,
            }
        )
    return round(value, 1), detail


def slim(row: dict) -> dict:
    return {
        key: row[key]
        for key in (
            "judge_id",
            "task_id",
            "branch_id",
            "capability",
            "slot_id",
            "expected_action",
            "trigger_kind",
        )
    } | {
        "response": row.get("response"),
        "utterance": row.get("utterance"),
        "protocol": row.get("protocol", "duplex"),
    }


def judge_row(row: dict, rubric: dict, raw_dir: Path, matcher: Matcher) -> dict:
    points = points_of(row, rubric)
    core = core_of(row, rubric, points)
    input_signature = hashlib.sha256(
        json.dumps(
            {
                "judge_id": row["judge_id"],
                "response": row.get("response"),
                "history": row.get("history"),
                "utterance": row.get("utterance"),
                "expected_action": row["expected_action"],
                "target_text": row["target_text"],
                "points": points,
                "forbidden_claims": row["forbidden_claims"],
                "correction_check": row.get("correction_check"),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    raw_path = raw_dir / (hashlib.md5(row["judge_id"].encode()).hexdigest() + ".json")
    if row["expected_action"] != "WAIT" and silent(row["response"]):
        output = {"action": "wrong", "observed": "silent", "points": [], "forbidden": []}
    elif row["expected_action"] == "WAIT" and silent(row["response"]):
        output = {"action": "correct", "observed": "silent", "points": [], "forbidden": []}
    elif raw_path.exists():
        cached = json.loads(raw_path.read_text())
        if cached.get("input_sha256") == input_signature:
            output = cached["match"]
        else:
            output = match(row, points, matcher)
            raw_path.write_text(
                json.dumps(
                    {
                        "judge_id": row["judge_id"],
                        "input_sha256": input_signature,
                        "points": points,
                        "match": output,
                    },
                    ensure_ascii=False,
                    indent=1,
                )
                + "\n"
            )
    else:
        output = match(row, points, matcher)
        raw_path.write_text(
            json.dumps(
                {
                    "judge_id": row["judge_id"],
                    "input_sha256": input_signature,
                    "points": points,
                    "match": output,
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n"
        )
    value, detail = score(row, points, output, core)
    return {
        **slim(row),
        "scorable": True,
        "status": "ok",
        "score": value,
        **detail,
        "points": [strip_ref(point) for point in points],
        "match": output,
    }


def report(results: list[dict]) -> dict:
    ok = [row for row in results if row.get("status") == "ok"]
    histogram = Counter(min(int(row["score"]), 4) for row in ok)
    per_capability = defaultdict(list)
    for row in ok:
        per_capability[row["capability"]].append(row)
    return {
        "slots": len(results),
        "scorable_ok": len(ok),
        "not_scorable": 0,
        "api_failed": sum(row.get("status") != "ok" for row in results),
        "mean": round(sum(row["score"] for row in ok) / len(ok), 2) if ok else None,
        "hist": {f"[{index},{index + 1}{')' if index < 4 else ']'}": histogram[index] for index in range(5)},
        "exact_0": sum(row["score"] == 0 for row in ok),
        "exact_5": sum(row["score"] == 5 for row in ok),
        "per_cap": {
            capability: {
                "n": len(rows),
                "mean": round(sum(row["score"] for row in rows) / len(rows), 2),
                "point_hit_rate": round(
                    sum(status == "hit" for row in rows for status in row["statuses"])
                    / max(1, sum(len(row["statuses"]) for row in rows)),
                    3,
                ),
                "core_hit_rate": round(
                    sum(
                        status == "hit"
                        for row in rows
                        for status, is_core in zip(row["statuses"], row.get("core") or [])
                        if is_core
                    )
                    / max(1, sum(sum(row.get("core") or []) for row in rows)),
                    3,
                ),
                "action_correct": round(sum(row["action"] == "correct" for row in rows) / len(rows), 3),
                "not_scorable": 0,
            }
            for capability, rows in sorted(per_capability.items())
        },
        "spent_usd": round(SPENT["usd"], 4),
        "calls": SPENT["calls"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="append", required=True, help="file or glob; repeatable")
    parser.add_argument("--rubric", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "qwen/qwen3.8-omni-flash"))
    args = parser.parse_args()

    paths = []
    for pattern in args.eval:
        matches = [Path(item) for item in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = load_rows(sorted(set(paths)))
    judge_ids = [row["judge_id"] for row in rows]
    if len(judge_ids) != len(set(judge_ids)):
        duplicates = [item for item, count in Counter(judge_ids).items() if count > 1]
        raise ValueError(f"duplicate judge ids from --eval inputs: {duplicates[:10]}")
    if args.sample:
        random.seed(20261017)
        by_capability = defaultdict(list)
        for row in rows:
            by_capability[row["capability"]].append(row)
        per_group = max(1, args.sample // len(by_capability))
        rows = [
            row
            for group in by_capability.values()
            for row in random.sample(group, min(per_group, len(group)))
        ]
    rubric = json.loads(Path(args.rubric).read_text())
    out = Path(args.out)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extra = json.loads(os.environ.get("JUDGE_EXTRA", "{}"))
    matcher = Matcher(args.judge_model, extra)
    configuration = {
        "protocol": "museumlive_point_judge_lenient_v3",
        "judge_profile": JUDGE_PROFILE,
        "judge_model": args.judge_model,
        "judge_api_url": matcher.url,
        "matcher_prompt_sha256": hashlib.sha256(MATCH_SYSTEM.encode()).hexdigest(),
        "correction_addendum_sha256": hashlib.sha256(CORR_ADDENDUM.encode()).hexdigest(),
        "rubric_sha256": file_sha256(Path(args.rubric)),
        "eval_inputs": [
            {"path": str(path), "sha256": file_sha256(path)} for path in sorted(set(paths))
        ],
        "formula": {
            "action_weight": W_ACTION,
            "points_weight": W_POINTS,
            "forbidden_penalty": PEN_FORBID,
            "contradiction_penalty": PEN_CONTRA,
            "action_partial_credit": ACT["partial"],
            "point_partial_credit": CREDIT["partial"],
            "context_correction_weights": [W_CORR, W_CC_ACTION, W_CC_POINTS],
            "reinforced_penalty": PEN_REINFORCE,
        },
        "workers": args.workers,
        "sample": args.sample,
        "judge_extra": extra,
    }
    (out / "configuration.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"scoring {len(rows)} slots with {args.workers} workers", flush=True)
    results = []
    with ThreadPoolExecutor(args.workers) as executor:
        futures = [executor.submit(judge_row, row, rubric, raw_dir, matcher) for row in rows]
        for index, future in enumerate(futures, 1):
            try:
                results.append(future.result())
            except Exception as error:
                row = rows[index - 1]
                results.append({**slim(row), "scorable": True, "status": "api_failed", "error": str(error)[:500]})
            if index % 200 == 0:
                print(f"{index}/{len(rows)} ${SPENT['usd']:.3f}", flush=True)
    with (out / "results.jsonl").open("w") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = report(results)
    summary["judge_profile"] = JUDGE_PROFILE
    summary["judge_model"] = args.judge_model
    summary["matcher_prompt_sha256"] = configuration["matcher_prompt_sha256"]
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["api_failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
