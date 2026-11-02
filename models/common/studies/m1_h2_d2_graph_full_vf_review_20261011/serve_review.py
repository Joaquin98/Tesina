"""Read-only localhost reviewer, including HTTP byte ranges for original video."""
import sys as _delivery_sys
from pathlib import Path as _DeliveryPath
_delivery_sys.dont_write_bytecode = True
_delivery_root = next(parent for parent in _DeliveryPath(__file__).resolve().parents if (parent / 'models/source_layout.py').is_file())
if str(_delivery_root) not in _delivery_sys.path:
    _delivery_sys.path.insert(0, str(_delivery_root))
from models.source_layout import experiment_directory
from models.model_catalog import resolve_artifact as _resolve_model_artifact, artifact_glob as _model_artifact_glob, temporary_settings as _temporary_model_settings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import re
import argparse
HERE = experiment_directory(__file__)
VIDEO = HERE.parents[1] / 'data/videos/partidos_evaluacion/VF_S01_F4_30_09_2023_Jockey_Club_VT_vs_Alma_Jr_PRIMERA_JneZwNgtDYI.mp4'

class Handler(BaseHTTPRequestHandler):

    def do_HEAD(self):
        self.respond(False)

    def do_GET(self):
        self.respond(True)

    def respond(self, body):
        route = self.path.split('?')[0]
        paths = {'/': (HERE / 'review.html', 'text/html; charset=utf-8'), '/video': (VIDEO, 'video/mp4'), '/result.json': (HERE / 'results/result.json', 'application/json'), '/timing.json': (HERE / 'results/timing.json', 'application/json'), '/comparison.json': (HERE / 'results/historical_comparison.json', 'application/json'), '/recommended.json': (HERE / 'results/recommended_threshold_events.json', 'application/json')}
        if route == '/progress.json':
            path = HERE / 'logs/progress.jsonl'
            try:
                row = json.loads(_resolve_model_artifact(path).read_text(encoding='utf-8').strip().splitlines()[-1])
            except (FileNotFoundError, IndexError, json.JSONDecodeError):
                row = {'stage': 'loading', 'message': 'Cargando modelos y compilando las funciones', 'elapsed_seconds': 0}
            data = json.dumps(row).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            if body:
                self.wfile.write(data)
            return
        if route not in paths:
            self.send_error(404)
            return
        path, mime = paths[route]
        if not _resolve_model_artifact(path).exists():
            self.send_error(404, 'Result not ready')
            return
        size = path.stat().st_size
        start, end, status = (0, size - 1, 200)
        header = self.headers.get('Range') if route == '/video' else None
        if header:
            match = re.fullmatch('bytes=(\\d*)-(\\d*)', header.strip())
            if not match or not any(match.groups()):
                self.send_error(416)
                return
            a, b = match.groups()
            if a:
                start = int(a)
                end = min(int(b), end) if b else end
            else:
                start = max(0, size - int(b))
            if start >= size or end < start:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Cache-Control', 'no-store')
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        if body:
            try:
                with _resolve_model_artifact(path).open('rb') as file:
                    file.seek(start)
                    left = end - start + 1
                    while left:
                        data = file.read(min(left, 256 * 1024))
                        if not data:
                            break
                        self.wfile.write(data)
                        left -= len(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Revisor: http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()
