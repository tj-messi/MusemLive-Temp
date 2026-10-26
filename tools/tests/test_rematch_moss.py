"""Behavioral regression cases for delivery-aware MOSS response matching."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("rematch_moss", Path(__file__).parents[1] / "rematch_moss.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    return {"branch_key": "task::branch", "omni": {"audio_input": False, "window_s": [0, 14],
             "reply_wait_s": 6, "tts_spans": [{"event_id": "q1", "tts_start": 2, "tts_end": 4.2}]},
            "user_events": [{"event_id": "q1", "time_s": 2, "text": "Which object?"}],
            "text_inputs": [{"t": 5, "text": "【用户对你说】Which object?"}],
            "duplex_chunks": [{"t": 3, "listen": False, "text": "old", "eot": True},
                              {"t": 5, "listen": True, "text": "", "eot": False},
                              {"t": 6, "listen": False, "text": "valid answer", "eot": True}],
            "timeline": [{"type": "user_event", "event_id": "q1", "response": "old", "response_time_s": 3}],
            "slots": [{"slot": {"slot_id": "answer", "window": {"start_s": 2, "end_s": 10}}}]}


class RematchTests(unittest.TestCase):
    def test_early_answer_replaced_by_later_eligible_answer(self):
        result = module.rematch_branch(fixture())
        self.assertEqual(result["queries"][0]["selected_candidate"]["text"], "valid answer")
        self.assertTrue(result["queries"][0]["requires_rejudge"])
        self.assertEqual(result["slots_requiring_response_or_context_rejudging"], ["answer"])

    def test_continuous_turn_is_split_without_old_prefix(self):
        row = fixture()
        row["duplex_chunks"] = [{"t": 4, "listen": False, "text": "old prefix ", "eot": False},
                                {"t": 5, "listen": False, "text": "after delivery", "eot": True}]
        candidate = module.rematch_branch(row)["queries"][0]["selected_candidate"]
        self.assertEqual(candidate["text"], "after delivery")
        self.assertTrue(candidate["split_continuation"])

    def test_later_candidates_preserved_not_silently_discarded(self):
        row = fixture()
        row["duplex_chunks"].append({"t": 7, "listen": False, "text": "another", "eot": True})
        self.assertEqual(len(module.rematch_branch(row)["queries"][0]["all_eligible_candidates"]), 2)

    def test_next_text_delivery_bounds_prior_query(self):
        row = fixture()
        row["user_events"].append({"event_id": "q2", "time_s": 5, "text": "New question"})
        row["omni"]["tts_spans"].append({"event_id": "q2", "tts_start": 5, "tts_end": 5.9})
        row["text_inputs"].append({"t": 6, "text": "【用户对你说】New question"})
        result = module.rematch_branch(row)
        self.assertIsNone(result["queries"][0]["selected_candidate"])
        self.assertEqual(result["queries"][1]["selected_candidate"]["text"], "valid answer")

    def test_native_audio_is_rejected(self):
        row = fixture()
        row["omni"]["audio_input"] = True
        with self.assertRaisesRegex(ValueError, "text-input"):
            module.rematch_branch(row)

    def test_missing_delivery_is_rejected(self):
        row = fixture()
        row["text_inputs"] = []
        with self.assertRaisesRegex(ValueError, "Missing/ambiguous"):
            module.rematch_branch(row)

    def test_missing_chunk_log_is_rejected(self):
        row = fixture()
        del row["duplex_chunks"]
        with self.assertRaisesRegex(ValueError, "Complete"):
            module.rematch_branch(row)

    def test_out_of_order_chunks_are_rejected(self):
        row = fixture()
        row["duplex_chunks"].reverse()
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            module.rematch_branch(row)

    def test_unchanged_response_does_not_invalidate_labels(self):
        row = fixture()
        row["timeline"][0].update(response="valid answer", response_time_s=6)
        result = module.rematch_branch(row)
        self.assertFalse(result["queries"][0]["requires_rejudge"])
        self.assertEqual(result["slots_requiring_response_or_context_rejudging"], [])

    def test_input_is_not_mutated(self):
        row = fixture()
        original = copy.deepcopy(row)
        module.rematch_branch(row)
        self.assertEqual(row, original)

    def test_expired_response_not_assigned(self):
        row = fixture()
        row["duplex_chunks"][-1]["t"] = 11
        self.assertIsNone(module.rematch_branch(row)["queries"][0]["selected_candidate"])

    def test_query_slot_rejudged_when_delivery_after_nominal_slot_end(self):
        row = fixture()
        row["slots"][0]["slot"].update(trigger={"event_id": "q1"}, window={"start_s": 2, "end_s": 4})
        self.assertEqual(module.rematch_branch(row)["slots_requiring_response_or_context_rejudging"], ["answer"])

    def test_repeated_utterance_resolved_by_delivery_tick(self):
        row = fixture()
        row["user_events"].append({"event_id": "q2", "time_s": 8, "text": "Which object?"})
        row["omni"]["tts_spans"].append({"event_id": "q2", "tts_start": 8, "tts_end": 9.2})
        row["text_inputs"].append({"t": 10, "text": "【用户对你说】Which object?"})
        self.assertEqual([x["delivery_s"] for x in module.delivery_events(row)], [5, 10])


if __name__ == "__main__":
    unittest.main()
