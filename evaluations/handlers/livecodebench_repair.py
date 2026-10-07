"""Original LCB one-program plus one-feedback repair, with retained real outputs."""
import json
import os
import subprocess
from pathlib import Path


def _digest(path):
    return subprocess.check_output(['b3sum', '--num-threads', '4', str(path)], text=True).split()[0]


def run_selfrepair(problem, idx, model_name, client, original_messages):
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.prompts.self_repair import format_prompt_self_repair
    from lcb_runner.utils.extraction_utils import extract_code
    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
    from lcb_runner.evaluation.pass_k_utils import extract_instance_results
    root = Path(os.environ['PACE_LCB_RESULTS_ROOT'])
    binding = Path(os.environ['PACE_NATIVE_BINDING'])
    if not root.is_absolute() or not binding.is_absolute() or not binding.is_file():
        raise RuntimeError('Original LCB repair needs owned artifacts and exact native input binding')
    root = root / model_name / ('selfrepair_' + str(idx))
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lineage = {'model': model_name, 'instance_id': idx, 'question_id': problem.question_id,
               'native_binding_blake3': _digest(binding),
               'handler_blake3': _digest(Path(__file__)),
               'initial_messages': original_messages, 'sample': problem.get_evaluation_sample()}
    identity = root / 'lineage.json'
    if identity.exists():
        if json.loads(identity.read_text()) != lineage:
            raise RuntimeError('Original LCB repair lineage changed')
    else:
        with identity.open('x') as file:
            json.dump(lineage, file)
    def stage(name, messages):
        saved = root / (name + '.response.json')
        if saved.exists():
            response = json.loads(saved.read_text())
            if response['messages'] != messages:
                raise RuntimeError('Original LCB retained generation prompt differs')
        else:
            result = client.chat.completions.create(model=model_name, messages=messages,
                                                   temperature=0, max_tokens=4096, timeout=7200)
            response = {'messages': messages, 'response': result.model_dump(mode='json')}
            with saved.open('x') as file:
                json.dump(response, file)
                file.flush(); os.fsync(file.fileno())
        content = response['response']['choices'][0]['message']['content']
        if not isinstance(content, str):
            raise RuntimeError('Original LCB generation has invalid content')
        code = extract_code(content, LMStyle.OpenAIChat)
        grade_file = root / (name + '.grade.json')
        if grade_file.exists():
            grade = json.loads(grade_file.read_text())
            if grade['response_blake3'] != _digest(saved):
                raise RuntimeError('Original LCB retained grade binds another response')
        else:
            raw, metadata = check_correctness(lineage['sample'], code, timeout=6)
            if not isinstance(raw, list) or not raw:
                raise RuntimeError('Original LCB returned no test cases')
            grade = {'passed': extract_instance_results({0: [raw]})[0][0],
                     'test_case_results': raw, 'grader_metadata': metadata,
                     'response_blake3': _digest(saved)}
            with grade_file.open('x') as file:
                json.dump(grade, file)
                file.flush(); os.fsync(file.fileno())
        return content, code, grade
    original_output, original_code, first = stage('original', original_messages)
    repair_prompt = format_prompt_self_repair(problem.question_content, LMStyle.OpenAIChat,
                                             original_code, first['passed'],
                                             json.dumps(first['grader_metadata']))
    if repair_prompt == '':
        output, code, grade = original_output, original_code, first
    else:
        output, code, grade = stage('repair', repair_prompt)
    result = problem.insert_output_evaluation([output], [code], [grade['passed']],
                                              test_case_results=grade['test_case_results'],
                                              grader_metadata=grade['grader_metadata'],
                                              original_code_list=[original_code], original_grade=first,
                                              repair_exercised=repair_prompt != '')
    result['instance_id'], result['subtask'] = idx, 'selfrepair'
    return [result]
