"""Run the original DebugBench LeetCodeTester locally; no neural inference or exported cookies."""
import argparse
import csv
import hashlib
import http.server
import json
import os
from pathlib import Path
import sys
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--token', type=Path, required=True)
    parser.add_argument('--keys', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.outputs.mkdir(mode=0o700, parents=True, exist_ok=True)
    token = args.token.read_text().strip()
    keys = json.loads(args.keys.read_text())[0]
    if not keys.get('leetcode_session') or not keys.get('csrf_token'):
        raise ValueError('Original LeetCode credentials are missing')
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / 'evaluations/benchmarks/debugbench/evaluation'))
    from leetcode_oj import LeetCodeTester
    rows = csv.DictReader((repo / 'scripts/pacebench/selections/abs_fit/selections_C100_text.csv').open())
    calls = {r['instance_id'] for r in rows if r['benchmark'] == 'debugbench'}
    allowed = {(call.split('_', 1)[0], call.rsplit('_', 1)[1]) for call in calls}
    tester = LeetCodeTester(leetcode_session=keys['leetcode_session'], csrf_token=keys['csrf_token'], cooldown=25)
    lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            if self.path != '/debugbench' or self.headers.get('Authorization') != 'Bearer ' + token:
                self.send_error(401)
                return
            try:
                size = int(self.headers['Content-Length'])
                if not 0 < size <= 2 << 20:
                    raise ValueError('Original fixed-code request exceeds allowed size')
                request = json.loads(self.rfile.read(size))
                if set(request) != {'code', 'task_id', 'language'} or (request['language'], request['task_id']) not in allowed:
                    raise ValueError('Request is not an original selected DebugBench call')
                encoded = json.dumps(request, sort_keys=True).encode()
                path = args.outputs / (hashlib.sha256(encoded).hexdigest() + '.json')
                with lock:
                    if path.exists():
                        record = json.loads(path.read_text())
                        if record['request'] != request:
                            raise RuntimeError('Committed grader request differs')
                    else:
                        reward, result = tester.test(**request)
                        record = {'request': request, 'test_result_bool': reward, 'test_result_dict': result,
                                  'grader': 'original_debugbench_leetcode_oj.LeetCodeTester'}
                        with path.open('x') as file:
                            json.dump(record, file, allow_nan=False)
                            file.flush()
                            os.fsync(file.fileno())
                    body = json.dumps(record, allow_nan=False).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                print('ORIGINAL_DEBUGBENCH_GRADED', request['task_id'], record['test_result_bool'], flush=True)
            except Exception as error:
                print('ORIGINAL_DEBUGBENCH_GRADER_FAILED', type(error).__name__, flush=True)
                self.send_error(500, type(error).__name__)

    print('ORIGINAL_DEBUGBENCH_LOCAL_CPU_READY', args.port, flush=True)
    http.server.ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
