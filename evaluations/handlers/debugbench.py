"""DebugBench handler for ProxyBench."""

import json
import os
import sys

_HANDLERS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVALUATIONS_DIR = os.path.dirname(_HANDLERS_DIR)
BENCHMARKS_DIR = os.path.join(_EVALUATIONS_DIR, "benchmarks")

DEBUGBENCH_DIR = os.path.join(BENCHMARKS_DIR, "debugbench")
DEBUGBENCH_BENCHMARK_DIR = os.path.join(DEBUGBENCH_DIR, "benchmark")
DEBUGBENCH_EVAL_DIR = os.path.join(DEBUGBENCH_DIR, "evaluation")
DEBUGBENCH_KEYS_FILE = os.path.join(DEBUGBENCH_EVAL_DIR, "keys.json")
_LOCAL_GRADER_URL = 'http://127.0.0.1:8771/debugbench'
_LOCAL_GRADER_TOKEN = '/root/eugeny-pace/debugbench-grader.token'


def _run_debugbench(
    model_name: str,
    base_url: str,
    api_key: str,
    instance_id: str,
) -> list:
    """Run a single DebugBench instance (LLM debug + LeetCode OJ test).

    instance_id format: "{file_stem}_{slug}"
    e.g. "python3_condition error_the-kth-factor-of-n"
    """
    file_stems = [
        os.path.splitext(f)[0]
        for f in os.listdir(DEBUGBENCH_BENCHMARK_DIR)
        if f.endswith(".json")
    ]

    matched_stem = None
    slug = None
    for stem in file_stems:
        prefix = stem + "_"
        if instance_id.startswith(prefix):
            matched_stem = stem
            slug = instance_id[len(prefix):]
            break

    if matched_stem is None:
        raise ValueError(
            f"Cannot parse instance_id '{instance_id}'. "
            f"Expected format: '{{file_stem}}_{{slug}}' where file_stem is one of: "
            f"{sorted(file_stems)[:5]}..."
        )

    data_file = os.path.join(DEBUGBENCH_BENCHMARK_DIR, matched_stem + ".json")
    with open(data_file) as f:
        cases = json.load(f)

    case = next((c for c in cases if c.get("slug") == slug), None)
    if case is None:
        raise ValueError(
            f"Slug '{slug}' not found in {matched_stem}.json. "
            f"Available slugs: {[c['slug'] for c in cases[:5]]}"
        )

    os.environ["LITELLM_API_KEY"] = api_key
    os.environ["LITELLM_BASE_URL"] = base_url

    if DEBUGBENCH_EVAL_DIR not in sys.path:
        sys.path.insert(0, DEBUGBENCH_EVAL_DIR)

    from debugger import LiteLLMResponser, IODebugger

    # Prefix with "openai/" so litellm routes through LITELLM_BASE_URL instead
    # of treating "azure_ai/..." as a native Azure AI Foundry target.
    litellm_model = f"openai/{model_name}" if not model_name.startswith("openai/") else model_name
    responser = LiteLLMResponser(model=litellm_model)
    debugger = IODebugger(responser)

    lang = matched_stem.split("_")[0]
    fixed_code, fixing_exp = debugger.debug(lang=lang, code=case["buggy_code"])

    result = dict(case)
    result["fixed_code"] = fixed_code
    result["fixing_exp"] = fixing_exp
    result["test_result_bool"] = None
    result["test_result_dict"] = None

    grader_url = os.environ.get('PACE_DEBUGBENCH_GRADER_URL')
    if grader_url is None and os.path.isfile(_LOCAL_GRADER_TOKEN):
        grader_url = _LOCAL_GRADER_URL
    if grader_url:
        import pathlib
        import urllib.request
        if grader_url != _LOCAL_GRADER_URL:
            raise ValueError('Original local grader must be SSH-bound loopback')
        token_path = os.environ.get('PACE_DEBUGBENCH_GRADER_TOKEN') or _LOCAL_GRADER_TOKEN
        token = pathlib.Path(token_path).read_text().strip()
        request = {'code': fixed_code, 'language': lang, 'task_id': slug}
        query = urllib.request.Request(grader_url, data=json.dumps(request).encode(),
                                       headers={'Content-Type': 'application/json',
                                                'Authorization': 'Bearer ' + token})
        with urllib.request.urlopen(query, timeout=180) as response:
            graded = json.load(response)
        if graded['request'] != request or graded['grader'] != 'original_debugbench_leetcode_oj.LeetCodeTester':
            raise RuntimeError('Original DebugBench grader request/owner differs')
        from evaluations.generation import retain_transport
        retain_transport({'original_debugbench_grader': graded})
        result['test_result_bool'] = graded['test_result_bool']
        result['test_result_dict'] = graded['test_result_dict']
    else:
        from leetcode_oj import LeetCodeTester
        with open(DEBUGBENCH_KEYS_FILE) as f:
            keys = json.load(f)
        first_key = keys[0] if keys else {}
        session = first_key.get('leetcode_session', '')
        csrf = first_key.get('csrf_token', '')
        if not session or session.startswith('<') or not csrf or csrf.startswith('<'):
            raise RuntimeError('Original DebugBench LeetCode credentials are missing')
        tester = LeetCodeTester(leetcode_session=session, csrf_token=csrf, cooldown=25)
        result['test_result_bool'], result['test_result_dict'] = tester.test(
            code=fixed_code, language=lang, task_id=slug)

    return [result]
