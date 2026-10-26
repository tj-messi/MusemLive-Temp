"""Regressions for previews that remove or do not execute JavaScript."""
import base64
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from build_holistic_visualization import render_document


class ContentProbe(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.cases, self.images, self.scripts = [], [], [], 0
        self.texts = {}
        self.controls = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'script':
            self.scripts += 1
        if tag == 'article':
            self.cases.append(attrs)
        if tag == 'img':
            self.images.append(attrs)
        if attrs.get('id') == 'start-presentation' or attrs.get('class') == 'filters':
            self.controls[attrs.get('id', 'filters')] = attrs
        if tag not in ('meta', 'br', 'img', 'input', 'link', 'hr'):
            self.stack.append((tag, attrs))

    def handle_endtag(self, tag):
        while self.stack:
            current, _ = self.stack.pop()
            if current == tag:
                break

    def handle_data(self, data):
        if any(tag in ('script', 'style') for tag, _ in self.stack):
            return
        for _, attrs in self.stack:
            key = attrs.get('id')
            if key:
                self.texts[key] = self.texts.get(key, '') + data


class StaticShowcase(unittest.TestCase):
    def make_document(self, question='是什么？'):
        image_bytes = b'fixture-image'
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        image_uri = 'data:image/jpeg;base64,' + base64.b64encode(image_bytes).decode()
        cases = [{'id': f'case_{n}', 'title': f'Case {n}', 'clip_id': 'clip_1', 'kind': 'current_view',
                  'question': question, 'candidate_answer': '__MEAN__ <b>原始文本</b>', 'reference_answer': '参考答案',
                  'previous': [], 'question_time_s': 12.3, 'human_review': None,
                  'effective': {'score': 0 if n % 2 else 5}, 'judgment': {'reason': '评分理由'},
                  'evidence': [{'sha256': image_hash, 'clip_s': 10.0}]} for n in range(8)]
        stats = {'scored_items': 8, 'pending_qa_review': 0, 'mean_score': 2.5, 'normalized_score': .5,
                 'score_distribution': {str(n): 4 if n in (0, 5) else 0 for n in range(6)},
                 'score_bands': {'0–1': 4, '2–3': 0, '4–5': 4}}
        template = (Path(__file__).resolve().parents[1] / 'code/holistic_showcase_template.html').read_text(encoding='utf-8')
        return render_document(template, {'summary': {'with_human_review': stats}, 'cases': cases,
                              'images': {image_hash: image_uri}, 'metadata': {'results_sha256': 'verified'}}), image_uri

    def test_all_content_survives_removal_of_every_script(self):
        html, image_uri = self.make_document()
        probe = ContentProbe()
        probe.feed(re.sub(r'<script\b[^>]*>.*?</script\s*>', '', html, flags=re.S | re.I))
        self.assertEqual(probe.scripts, 0)
        self.assertEqual(probe.texts['mean'], '2.500')
        self.assertEqual(probe.texts['scored'], '8')
        self.assertIn('0–1 分 4 条', probe.texts['legend'])
        self.assertEqual(len(probe.cases), 8)
        self.assertEqual([c['data-score'] for c in probe.cases].count('0'), 4)
        self.assertEqual([c['data-score'] for c in probe.cases].count('5'), 4)
        self.assertEqual(sum(i.get('src') == image_uri for i in probe.images), 8)
        self.assertTrue(all('hidden' not in c for c in probe.cases))
        self.assertTrue(all('hidden' in attrs for attrs in probe.controls.values()))

    def test_untrusted_qa_is_text_and_not_executable_markup(self):
        payload = '<img src=x onerror=alert(1)><script>bad()</script>'
        html, _ = self.make_document(question=payload)
        probe = ContentProbe()
        probe.feed(html)
        self.assertEqual(probe.scripts, 1)  # Only the optional, fixed enhancement script.
        self.assertFalse(any('onerror' in image for image in probe.images))
        self.assertIn(payload, probe.texts['gallery'])
        self.assertIn('__MEAN__ <b>原始文本</b>', probe.texts['gallery'])


if __name__ == '__main__':
    unittest.main()
