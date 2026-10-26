"""Scoring invariants that affect reported research results; no API requests."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import run_holistic_benchmark as run


def judged(tid, score, pred='yes'):
    return {'id': tid, 'status': 'ok', 'human_review': None,
            'judgment': {'pred': pred, 'score': score, 'reason': '示例',
                         'needs_qa_review': False, 'qa_review_reason': ''}}


class ScoreInvariants(unittest.TestCase):
    def test_zero_is_scored_but_pending_and_failures_are_not(self):
        rows = [judged('a', 0, 'no'), judged('b', 5),
                {'id': 'c', 'status': 'ok', 'human_review': None, 'judgment': {
                    'pred': None, 'score': None, 'reason': '题意不清',
                    'needs_qa_review': True, 'qa_review_reason': '指代有歧义'}},
                {'id': 'd', 'status': 'api_failed', 'judgment': None, 'human_review': None,
                 'error': 'transport failure'}]
        for row in rows:
            row['effective'] = run.effective_result(row, None)
        for layer in ('judge', 'reviewed'):
            s = run.metrics(rows, layer)
            self.assertEqual((s['scored_items'], s['score_sum'], s['mean_score']), (2, 5, 2.5))
            self.assertEqual((s['pending_qa_review'], s['api_failed']), (1, 1))
            self.assertEqual(s['score_bands'], {'0–1': 1, '2–3': 0, '4–5': 1})

    def test_human_overlay_does_not_overwrite_automatic_score(self):
        row = judged('a', 4)
        human = {'score': 5, 'status': 'reviewed', 'rating': 'too_strict', 'note': '只问配色'}
        row['effective'] = run.effective_result(row, human)
        self.assertEqual(row['judgment']['score'], 4)
        self.assertEqual(run.metrics([row], 'reviewed')['mean_score'], 5)
        self.assertEqual(run.metrics([row], 'judge')['mean_score'], 4)

    def test_invalid_numeric_types_and_pending_scores_rejected(self):
        j = judged('a', 0)['judgment']
        for value in (True, False, '5', 2.5, None, -1, 6):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run.validate_judgment(dict(j, score=value))
        with self.assertRaises(ValueError):
            run.validate_judgment(dict(j, needs_qa_review=True, qa_review_reason='ambiguous'))

    def test_context_and_human_labels_cannot_leak_into_payload(self):
        item = {'question': '什么颜色？', 'reference_answer': '蓝白', 'candidate_answer': '蓝白相间'}
        before = run.payload_for(item, 'fixed system prompt')
        expanded = dict(item, method='private method', rubric='old rubric', human_score=5,
                        evidence=['private image'], prior_dialogue_for_human_only=['old history'])
        self.assertEqual(before, run.payload_for(expanded, 'fixed system prompt'))

    def test_multiturn_requires_all_scored_yes_answers(self):
        rows = [judged('a', 5), judged('b', 0, 'no'), judged('c', 5),
                {'id': 'd', 'status': 'api_failed', 'judgment': None, 'human_review': None, 'error': 'transport'}]
        items = [{'id': row['id'], 'point_id': 'first' if n < 2 else 'second', 'method': 'full', 'kind': 'test'}
                 for n, row in enumerate(rows)]
        for row in rows:
            row['effective'] = run.effective_result(row, None)
        s = run.summarize(items, rows)['multi_turn_raw_judge']
        self.assertEqual((s['groups'], s['eligible_groups'], s['unscored_groups']), (2, 1, 1))
        self.assertEqual(s['all_yes_groups'], 0)


if __name__ == '__main__':
    unittest.main()
