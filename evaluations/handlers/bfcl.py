"""BFCL (Berkeley Function-Calling Leaderboard) handler for ProxyBench."""

import os
from pathlib import Path
import sys

_HANDLERS_DIR = os.path.dirname(os.path.abspath(__file__))
_EVALUATIONS_DIR = os.path.dirname(_HANDLERS_DIR)
BENCHMARKS_DIR = os.path.join(_EVALUATIONS_DIR, "benchmarks")

BFCL_DIR = os.path.join(
    BENCHMARKS_DIR, "BFCL", "gorilla", "berkeley-function-call-leaderboard"
)

if BFCL_DIR not in sys.path:
    sys.path.insert(0, BFCL_DIR)

BFCL_SUBTASK_TO_TEST_CATEGORY = {
    # non_live
    "non_live_simple_python":        "simple_python",
    "non_live_simple_java":          "simple_java",
    "non_live_simple_javascript":    "simple_javascript",
    "non_live_multiple":             "multiple",
    "non_live_parallel":             "parallel",
    "non_live_parallel_multiple":    "parallel_multiple",
    "non_live_irrelevance":          "irrelevance",
    # live
    "live_simple":                   "live_simple",
    "live_multiple":                 "live_multiple",
    "live_parallel":                 "live_parallel",
    "live_parallel_multiple":        "live_parallel_multiple",
    "live_irrelevance":              "live_irrelevance",
    "live_relevance":                "live_relevance",
    # multi_turn
    "multi_turn_base":               "multi_turn_base",
    "multi_turn_long_context":       "multi_turn_long_context",
    "multi_turn_miss_func":          "multi_turn_miss_func",
    "multi_turn_miss_param":         "multi_turn_miss_param",
    # agentic
    "agentic_memory_kv":             "memory_kv",
    "agentic_memory_vector":         "memory_vector",
    "agentic_memory_rec_sum":        "memory_rec_sum",
    "agentic_web_search_base":       "web_search_base",
    "agentic_web_search_no_snippet": "web_search_no_snippet",
}

BFCL_DEFAULT_SUBTASK = "non_live_simple_python"


def _ensure_model_registered(model_name: str) -> None:
    """Register `model_name` in BFCL's MODEL_CONFIG_MAPPING if it is missing.

    The AST eval path (`convert_func_name`) looks up the model by its
    underscore->slash form, e.g. "azure_ai/gpt-5.2" -> "azure/ai/gpt-5.2".
    Custom models routed through an OpenAI-compatible proxy are not in BFCL's
    built-in registry, so we insert a minimal OpenAI-style config. Only
    `underscore_to_dot` is consulted during evaluation; we set it True because
    OpenAI-style endpoints reject '.' in function names.
    """
    from bfcl_eval.constants import model_config as _mc
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )

    key = model_name.replace("_", "/")
    if key in _mc.MODEL_CONFIG_MAPPING:
        return
    _mc.MODEL_CONFIG_MAPPING[key] = _mc.ModelConfig(
        model_name=model_name,
        display_name=model_name,
        url="",
        org="custom",
        license="proprietary",
        model_handler=OpenAICompletionsHandler,
        is_fc_model=True,
        underscore_to_dot=True,
    )


def _run_bfcl(
    model_name: str,
    base_url: str,
    api_key: str,
    subtask: str,
    instance_id: str,
) -> list:
    """Run a single BFCL instance (inference + evaluation)."""
    if subtask not in BFCL_SUBTASK_TO_TEST_CATEGORY:
        raise ValueError(
            f"Subtask '{subtask}' is not supported for bfcl. "
            f"Supported subtasks: {sorted(BFCL_SUBTASK_TO_TEST_CATEGORY.keys())}"
        )
    test_category = BFCL_SUBTASK_TO_TEST_CATEGORY[subtask]

    from bfcl_eval.utils import (
        load_dataset_entry,
        load_ground_truth_entry,
        is_relevance_or_irrelevance,
        is_multi_turn,
        is_agentic,
        is_java,
        is_js,
    )
    from bfcl_eval.constants.enums import Language, ReturnFormat
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    from bfcl_eval.eval_checker.eval_runner import (
        _evaluate_single_ast_entry,
        _evaluate_single_relevance_entry,
        _evaluate_single_multi_turn_entry,
        _evaluate_single_agentic_entry,
    )

    os.environ["OPENAI_API_KEY"] = api_key
    api_root = base_url.rstrip("/")
    if not api_root.endswith("/v1"):
        api_root += "/v1"
    os.environ["OPENAI_BASE_URL"] = api_root

    # Custom/proxied models aren't in BFCL's built-in registry; register so the
    # AST eval checker can resolve the model during scoring.
    _ensure_model_registered(model_name)

    handler = OpenAICompletionsHandler(
        model_name=model_name,
        temperature=0.001,
        registry_name="custom-FC",
        is_fc_model=True,
    )
    # BFCL's stock OpenAI client uses its SDK default timeout, which is shorter
    # than legitimate long multi-turn DSV4 generations. Preserve the handler and
    # request payload, but raise only the transport deadline.
    handler.client = handler.client.with_options(timeout=7200.0)

    # Some models reject `temperature` (e.g. reasoning models 400). The handler
    # always puts temperature in the payload, so wrap its API call to drop
    # temperature and retry once when the model rejects it.
    _orig_generate = handler.generate_with_backoff

    def _generate_no_temp_fallback(**kwargs):
        try:
            return _orig_generate(**kwargs)
        except Exception as exc:
            if "temperature" in kwargs and "temperature" in str(exc).lower():
                kwargs.pop("temperature", None)
                return _orig_generate(**kwargs)
            raise

    handler.generate_with_backoff = _generate_no_temp_fallback

    all_entries = load_dataset_entry(
        test_category, include_prereq=False, include_language_specific_hint=False
    )
    entry = next((e for e in all_entries if str(e["id"]) == str(instance_id)), None)
    if entry is None:
        # PACE selections / standardized CSVs reference BFCL instances by a SHORT
        # positional id ("<subtask>_<N>" = row N in the flattened data file),
        # but BFCL's live_* data files use LONG ids (e.g. "live_multiple_0-0-0").
        # Fall back to a positional lookup so the shipped Pace-Bench selection
        # scores correctly. Non-live files already use short ids (direct match).
        short = str(instance_id).rsplit("_", 1)[-1]
        if short.isdigit():
            idx = int(short)
            if 0 <= idx < len(all_entries):
                entry = all_entries[idx]
    if entry is None:
        raise ValueError(
            f"Instance id '{instance_id}' not found in test_category '{test_category}'. "
            f"Sample ids: {[e['id'] for e in all_entries[:5]]}"
        )

    # The upstream batch runner injects runtime state before agentic inference.
    # The single-instance PACE bridge must reproduce those write-path steps.
    if is_agentic(test_category):
        from bfcl_eval.utils import (
            populate_initial_settings_for_memory_test_cases,
            populate_initial_settings_for_web_search_test_cases,
        )
        selected = [entry]
        selected = populate_initial_settings_for_memory_test_cases(
            selected, Path("/tmp/bfcl-pace-results")
        )
        entry = populate_initial_settings_for_web_search_test_cases(selected)[0]

    actual_entry_id = str(entry["id"])

    def _ground_truth():
        all_ground_truth = load_ground_truth_entry(test_category)
        gt_entry = next(
            (g for g in all_ground_truth
             if str(g["id"]) in {str(instance_id), actual_entry_id}),
            None,
        )
        return gt_entry["ground_truth"] if gt_entry else []

    model_responses, metadata = handler.inference(
        entry, include_input_log=False, exclude_state_log=True
    )

    if is_relevance_or_irrelevance(test_category):
        result = _evaluate_single_relevance_entry(
            handler, instance_id, model_responses, entry, model_name, test_category
        )
    elif is_multi_turn(test_category):
        ground_truth = _ground_truth()
        result = _evaluate_single_multi_turn_entry(
            handler, instance_id, model_responses, ground_truth, entry, model_name, test_category
        )
    elif is_agentic(test_category):
        possible_answer = _ground_truth()
        result = _evaluate_single_agentic_entry(
            handler, instance_id, model_responses, possible_answer, entry, model_name, test_category
        )
    else:
        possible_answer = _ground_truth()

        if is_java(test_category):
            language, return_format = Language.JAVA, ReturnFormat.JAVA
        elif is_js(test_category):
            language, return_format = Language.JAVASCRIPT, ReturnFormat.JAVASCRIPT
        else:
            language, return_format = Language.PYTHON, ReturnFormat.PYTHON

        result = _evaluate_single_ast_entry(
            handler,
            instance_id,
            model_responses,
            possible_answer,
            entry,
            model_name,
            test_category,
            language=language,
            return_format=return_format,
            has_tool_call_tag=False,
        )

    result.setdefault("model_result_raw", model_responses)
    result["metadata"] = metadata
    return [result]
