"""Real-browser regression for normal, script-blocked and script-stripped previews."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import urllib.request

import websocket


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--html', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--chrome', type=Path)
    parser.add_argument('--pdf', type=Path, help='Optional static presentation PDF exported after verification')
    args = parser.parse_args()
    out, page, work = args.run_dir.resolve(), args.html.resolve(), args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    chrome = args.chrome or shutil.which('google-chrome') or shutil.which('chrome') or Path(
        r'C:\Program Files\Google\Chrome\Application\chrome.exe')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    process = subprocess.Popen([str(chrome), '--headless=new', '--disable-gpu', '--no-first-run',
        '--no-default-browser-check', '--disable-background-networking',
        f'--remote-allow-origins=http://localhost:{port}', '--remote-debugging-address=127.0.0.1',
        f'--remote-debugging-port={port}', '--user-data-dir=' + str(work/'chrome-profile'), 'about:blank'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    connection = None
    sequence = 0
    errors, network = [], []

    def call(method, params=None):
        nonlocal sequence
        sequence += 1
        connection.send(json.dumps({'id': sequence, 'method': method, 'params': params or {}}))
        while True:
            message = json.loads(connection.recv())
            if message.get('method') == 'Runtime.exceptionThrown':
                errors.append(message['params'])
            if message.get('method') == 'Network.requestWillBeSent':
                url = message['params']['request']['url']
                if url.startswith(('http:', 'https:')):
                    network.append(url)
            if message.get('id') == sequence:
                assert 'error' not in message, message
                return message.get('result', {})

    def evaluate(expression):
        result = call('Runtime.evaluate', {'expression': expression, 'returnByValue': True, 'awaitPromise': True})
        assert not result.get('exceptionDetails'), result
        return result.get('result', {}).get('value')

    def navigate(path):
        call('Page.navigate', {'url': path.as_uri()})
        for _ in range(100):
            if evaluate("document.readyState==='complete'"):
                return
            time.sleep(.1)
        raise AssertionError('Page did not finish loading')

    def viewport(width, height, mobile=False):
        call('Emulation.setDeviceMetricsOverride', {'width': width, 'height': height,
            'deviceScaleFactor': 1, 'mobile': mobile})

    def screenshot(name):
        data = call('Page.captureScreenshot', {'format': 'png', 'captureBeyondViewport': False})
        (work/name).write_bytes(base64.b64decode(data['data']))

    source = {s['id']: s for s in json.loads((out/'sample.json').read_text(encoding='utf-8'))}
    result_map = {s['id']: s for s in json.loads((out/'results.json').read_text(encoding='utf-8'))}
    selection = json.loads((out/'showcase_selection.json').read_text(encoding='utf-8'))['cases']
    stats = json.loads((out/'summary.json').read_text(encoding='utf-8'))['with_human_review']

    def check_content():
        observed = evaluate("""(()=>({
          mean:document.getElementById('mean').textContent,
          scored:document.getElementById('scored').textContent,
          pending:document.getElementById('pending').textContent,
          distribution:[...document.querySelectorAll('#exact-dist strong')].map(x=>Number(x.textContent)),
          cards:[...document.querySelectorAll('#gallery .case')].map(c=>({
            id:c.dataset.id,score:Number(c.dataset.score),hidden:c.hidden,
            question:c.querySelector('.question').textContent,
            answer:c.querySelector('.answer-text').textContent,
            reference:c.querySelector('.reference').textContent,
            reason:c.querySelector('.reason').textContent})),
          images:[...document.querySelectorAll('#gallery [data-image]')].map(f=>({
            hash:f.dataset.image,src:f.querySelector('img').src,loaded:f.querySelector('img').naturalWidth>0})),
          overflow:document.documentElement.scrollWidth>innerWidth,
          controlsHidden:document.getElementById('start-presentation').hidden&&document.querySelector('.filters').hidden
        }))()""")
        assert observed['mean'] == f"{stats['mean_score']:.3f}"
        assert int(observed['scored']) == stats['scored_items']
        assert int(observed['pending']) == stats['pending_qa_review']
        assert observed['distribution'] == [stats['score_distribution'][str(n)] for n in range(6)]
        assert [c['id'] for c in observed['cards']] == [c['id'] for c in selection]
        for card in observed['cards']:
            original, result = source[card['id']], result_map[card['id']]
            assert card['score'] == result['effective']['score']
            assert card['question'].endswith(original['question'])
            assert card['answer'] == original['candidate_answer']
            assert card['reference'].endswith(original['reference_answer'])
            assert card['reason'].endswith(result['judgment']['reason'])
            assert not card['hidden']
        expected_hashes = [e['sha256'] for c in selection for e in c['evidence']]
        assert [i['hash'] for i in observed['images']] == expected_hashes
        for image in observed['images']:
            assert image['loaded']
            assert hashlib.sha256(base64.b64decode(image['src'].split(',', 1)[1])).hexdigest() == image['hash']
        assert not observed['overflow']
        return {'mean': observed['mean'], 'cases': len(observed['cards']), 'images': len(observed['images']),
                'all_content_matches_original': True, 'controls_hidden': observed['controlsHidden']}

    try:
        target = None
        for _ in range(100):
            try:
                targets = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json', timeout=1))
                target = next(t for t in targets if t['type'] == 'page')
                break
            except (OSError, StopIteration):
                time.sleep(.1)
        assert target is not None
        connection = websocket.create_connection(target['webSocketDebuggerUrl'], timeout=35,
                                                   origin=f'http://localhost:{port}')
        for method in ('Runtime.enable', 'Page.enable', 'Network.enable'):
            call(method)
        viewport(1360, 1100)
        navigate(page)
        assert evaluate("document.body.getAttribute('data-ready')==='true'")
        normal = check_content()
        assert not normal['controls_hidden']
        screenshot('normal_overview.png')
        evaluate("document.getElementById('examples').scrollIntoView({behavior:'instant'})")
        screenshot('normal_cases.png')
        for score in (0, 5):
            visible = evaluate(f"document.querySelector('[data-filter=\"{score}\"]').click();[...document.querySelectorAll('#gallery .case:not([hidden])')].map(c=>Number(c.dataset.score))")
            assert visible == [score]*4
        evaluate("document.getElementById('start-presentation').click()")
        assert evaluate("document.getElementById('presentation').open")
        evaluate("document.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight'}))")
        assert evaluate("document.getElementById('presentation-count').textContent") == '2 / 4'
        evaluate("document.getElementById('prev-case').click()")
        evaluate("document.querySelector('#presentation-case [data-image]').click()")
        assert evaluate("document.getElementById('lightbox').open && document.getElementById('large-image').src===document.querySelector('#presentation-case img').src")
        evaluate("document.getElementById('close-image').click();document.getElementById('close-presentation').click();document.querySelector('[data-filter=all]').click()")
        viewport(390, 844, True)
        check_content()
        evaluate("window.scrollTo({top:0,behavior:'instant'})")
        screenshot('normal_mobile.png')
        no_dialog_script = call('Page.addScriptToEvaluateOnNewDocument', {
            'source': 'HTMLDialogElement.prototype.showModal=undefined;'})['identifier']
        navigate(page)
        check_content()
        assert evaluate("document.getElementById('start-presentation').hidden")
        call('Page.removeScriptToEvaluateOnNewDocument', {'identifier': no_dialog_script})
        call('Emulation.setScriptExecutionDisabled', {'value': True})
        navigate(page)
        disabled = check_content()
        assert disabled['controls_hidden']
        screenshot('scripts_disabled_mobile.png')
        evaluate("document.getElementById('examples').scrollIntoView({behavior:'instant'})")
        screenshot('scripts_disabled_cases_mobile.png')
        viewport(360, 800, True)
        check_content()
        stripped = work/'scripts_removed.html'
        stripped.write_text(re.sub(r'<script\b[^>]*>.*?</script\s*>', '', page.read_text(encoding='utf-8'),
                                  flags=re.S | re.I), encoding='utf-8')
        call('Emulation.setScriptExecutionDisabled', {'value': False})
        viewport(390, 844, True)
        navigate(stripped)
        removed = check_content()
        assert removed['controls_hidden']
        screenshot('scripts_removed_mobile.png')
        pdf_hash = None
        if args.pdf:
            viewport(1360, 1100)
            call('Emulation.setScriptExecutionDisabled', {'value': True})
            navigate(page)
            check_content()
            pdf = call('Page.printToPDF', {'printBackground': True, 'preferCSSPageSize': True,
                'landscape': True, 'displayHeaderFooter': True, 'headerTemplate': '<span></span>',
                'footerTemplate': '<div style="font:9px Arial;width:100%;text-align:center;color:#687870">MuseumLive | <span class="pageNumber"></span> / <span class="totalPages"></span></div>'})
            args.pdf.parent.mkdir(parents=True, exist_ok=True)
            args.pdf.write_bytes(base64.b64decode(pdf['data']))
            pdf_hash = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
        assert not errors and not network, {'errors': errors, 'network': network}
        report = {'status': 'passed', 'html_sha256': hashlib.sha256(page.read_bytes()).hexdigest(),
            'normal_browser': normal, 'scripts_disabled': disabled, 'scripts_removed': removed,
            'missing_dialog_support_hides_dialog_controls': True,
            'score_filters_and_keyboard_presentation': True, 'image_lightbox_preserves_original': True,
            'viewport_widths_checked': [1360, 390, 360], 'javascript_errors': errors,
            'external_requests': network, 'pdf_sha256': pdf_hash,
            'scope': 'Chrome desktop, simulated mobile viewports and script restrictions; actual recipient app not directly tested'}
        (out/'showcase_browser_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False))
    finally:
        if connection:
            connection.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == '__main__':
    main()
