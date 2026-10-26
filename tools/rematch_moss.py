#!/usr/bin/env python3
"""Rebuild delivery-aware MOSS response matches from complete chunk logs.

MOSS receives user-question text only after TTS completion. This tool maps each
question to the recorded text-delivery tick, splits model output at direct-user
delivery boundaries, and retains replies that start after delivery, within the
six-second reply window, and before the next user event. The emitted candidates
must pass through the frozen Judge before their scores are used. Native-audio
models do not use this text-only eligibility rule. Raw runtime inputs and
outputs should remain in the external controlled artifact store.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


USER_PREFIX = "【用户对你说】"


def delivery_events(branch):
    """Join delivered user text to events without guessing from speech onset."""
    if branch.get("omni", {}).get("audio_input") is not False:
        raise ValueError("Only explicitly text-input MOSS logs are eligible")
    if "text_inputs" not in branch or "duplex_chunks" not in branch:
        raise ValueError("Complete text_inputs and duplex_chunks are required")
    spans = {s["event_id"]: s for s in branch["omni"]["tts_spans"]}
    t0 = float(branch["omni"]["window_s"][0])
    inputs = branch["text_inputs"]
    events = []
    for event in branch["user_events"]:
        span = spans[event["event_id"]]
        # The published adapter labels each feed/output tick by its end.
        scheduled_tick = t0 + max(0, math.ceil(float(span["tts_end"]) - t0) - 1) + 1
        matches = [x for x in inputs
                   if x.get("text") == USER_PREFIX + event["text"]
                   and (x.get("event_id") == event["event_id"]
                        or ("event_id" not in x and math.isclose(float(x["t"]), scheduled_tick, abs_tol=1e-6)))]
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous delivery for {event['event_id']}")
        record = matches[0]
        # New logs may record consumption explicitly. Old logs only record
        # the adapter's feed tick; that distinction is retained in the output.
        tick = float(record.get("consumed_at_s", record["t"]))
        events.append({"event_id": event["event_id"], "delivery_s": tick,
                       "delivery_evidence": "consumption_trace" if "consumed_at_s" in record else "adapter_feed_tick",
                       "tts_end_s": float(span["tts_end"])})
    events.sort(key=lambda x: x["delivery_s"])
    if len({e["delivery_s"] for e in events}) != len(events):
        raise ValueError("Multiple user events in one tick need token-order traces")
    return events


def segment_chunks(chunks, boundaries):
    """Reconstruct output segments, never concatenating across a user input."""
    segments = []
    current = None
    previous_t = None
    previous_speaking = False
    boundaries = sorted(boundaries)

    def finish():
        nonlocal current
        if current and current["text"].strip():
            current["text"] = current["text"].strip()
            segments.append(current)
        current = None

    for chunk in chunks:
        tick = float(chunk["t"])
        if not math.isfinite(tick) or (previous_t is not None and tick <= previous_t):
            raise ValueError("Chunk timestamps must be finite and strictly increasing")
        cut = previous_t is not None and any(previous_t < b <= tick for b in boundaries)
        if cut:
            finish()
        speaking = not chunk["listen"]
        if speaking:
            if current is None:
                current = {"start_s": tick, "end_s": tick, "text": "",
                           "split_continuation": bool(cut and previous_speaking)}
            current["text"] += chunk.get("text") or ""
            current["end_s"] = tick
            if chunk.get("eot"):
                finish()
        else:
            finish()
        previous_t = tick
        previous_speaking = speaking and not chunk.get("eot", False)
    finish()
    return segments


def rematch_branch(branch):
    events = delivery_events(branch)
    segments = segment_chunks(branch["duplex_chunks"], [e["delivery_s"] for e in events])
    original = {x["event_id"]: x for x in branch.get("timeline", []) if x.get("type") == "user_event"}
    reply_wait = float(branch["omni"].get("reply_wait_s", 6))
    if reply_wait < 0:
        raise ValueError("reply_wait_s must be nonnegative")
    queries = []
    for index, event in enumerate(events):
        stop = events[index + 1]["delivery_s"] if index + 1 < len(events) else float("inf")
        candidates = [s for s in segments if event["delivery_s"] <= s["start_s"] <= event["tts_end_s"] + reply_wait
                      and s["start_s"] < stop]
        selected = candidates[0] if candidates else None
        old = original.get(event["event_id"], {})
        changed = ((old.get("response") or "").strip() != (selected["text"] if selected else "")
                   or old.get("response_time_s") != (selected["start_s"] if selected else None))
        queries.append({**event, "selected_candidate": selected, "all_eligible_candidates": candidates,
                        "old_response_time_s": old.get("response_time_s"),
                        "changed": changed, "requires_rejudge": changed})
    # Judge inputs include preceding dialogue. A changed response can therefore
    # invalidate later judging context, even when the later response is unchanged.
    changed_times = [q["delivery_s"] for q in queries if q["changed"]]
    affected_slots = []
    if changed_times:
        changed_ids = {q["event_id"] for q in queries if q["changed"]}
        for slot_record in branch.get("slots", []):
            slot = slot_record.get("slot", slot_record)
            if (slot.get("trigger", {}).get("event_id") in changed_ids
                    or float(slot["window"]["end_s"]) + 1e-6 >= min(changed_times)):
                affected_slots.append(slot["slot_id"])
    return {"branch_key": branch["branch_key"], "queries": queries,
            "slots_requiring_response_or_context_rejudging": affected_slots,
            "resegmented_turns": segments}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", required=True, type=Path, help="complete evaluation_results.json")
    parser.add_argument("--out", required=True, type=Path, help="external delivery-aware candidate JSON; does not overwrite archived scores")
    args = parser.parse_args()
    if args.eval.resolve() == args.out.resolve():
        parser.error("Output must differ from the input artifact")
    source = args.eval.read_bytes()
    data = json.loads(source)
    if "moss" not in data.get("model", "").lower():
        parser.error("Input model must explicitly identify MOSS")
    branches = [rematch_branch(b) for b in data["results"]]
    output = {"status": "delivery_aware_candidates_require_frozen_judge", "source_sha256": hashlib.sha256(source).hexdigest(),
              "clock": "recorded input-feed/output ticks; prefer actual prompt-consumption traces when available",
              "limitations": ["Same-tick feed does not establish token-level prompt consumption.",
                              "A split continuation can still answer an obsolete request; semantic judging is required.",
                              "Run main, ablation, context and FPS conditions separately, then regenerate all dependent analyses.",
                              "No native-audio incremental-response rule is changed."],
              "branches": branches}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "branches": len(branches),
                      "changed_queries": sum(q["changed"] for b in branches for q in b["queries"])}))


if __name__ == "__main__":
    main()
