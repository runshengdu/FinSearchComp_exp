import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI
from utils import atomic_write_jsonl, load_jsonl_records, progress_line


def is_reward_info_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, dict):
        return len(v) == 0
    return False


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    candidates: List[str] = []
    m = _JSON_BLOCK_RE.search(text)
    if m:
        candidates.append(m.group(1).strip())
    candidates.append(text)
    for c in candidates:
        if not c:
            continue
        start = c.find("{")
        end = c.rfind("}")
        if start < 0 or end < 0 or end <= start:
            continue
        snippet = c[start : end + 1]
        try:
            obj = json.loads(snippet)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _extract_rationale(text: str) -> str:
    m = re.search(r"【评分依据】[:：]\s*(.+)", text)
    if not m:
        return ""
    return (m.group(1) or "").strip()


def eval_single_record(
    call_llm_with_retries: Callable[..., Dict[str, Any]],
    client: OpenAI,
    evaluator_cfg: Dict[str, Any],
    judge_system_prompt: str,
    judge_prompt: str,
) -> Optional[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": judge_system_prompt},
        {"role": "user", "content": judge_prompt},
    ]
    resp = call_llm_with_retries(
        client=client,
        model=evaluator_cfg["name"],
        messages=messages,
        tools=[],
        temperature=evaluator_cfg.get("temperature"),
        max_tokens=evaluator_cfg.get("max_tokens"),
        extra_body=evaluator_cfg.get("extra_body"),
        tool_choice="none",
    )
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if not isinstance(content, str) or not content.strip():
        return None

    obj = _extract_first_json_object(content)
    if not obj:
        return None
    score = obj.get("answer_score")
    if not isinstance(score, int) or score not in (0, 1):
        return None

    return {"answer_score": int(score), "rationale": _extract_rationale(content)}


def run_evaluation(
    call_llm_with_retries: Callable[..., Dict[str, Any]],
    evaluator_client: OpenAI,
    evaluator_cfg: Dict[str, Any],
    response_path: Path,
    dataset_by_id: Dict[str, Dict[str, Any]],
    max_workers: int = 8,
) -> Tuple[int, int]:
    records = load_jsonl_records(response_path)

    init_updated = False
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            continue
        if "reward_info" not in r:
            records[i] = dict(r)
            records[i]["reward_info"] = None
            init_updated = True
    if init_updated:
        atomic_write_jsonl(response_path, records)

    jobs: List[Tuple[int, str, str, str]] = []
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            continue
        if not is_reward_info_empty(r.get("reward_info")):
            continue
        prompt_id = r.get("prompt_id")
        if not isinstance(prompt_id, str):
            continue
        ds = dataset_by_id.get(prompt_id)
        if not ds:
            continue

        prompt = ds.get("prompt")
        response_reference = ds.get("response_reference")
        judge_prompt_template = ds.get("judge_prompt_template")
        judge_system_prompt = ds.get("judge_system_prompt")
        response = r.get("llm_response")

        if not isinstance(prompt, str) or not isinstance(response_reference, str):
            continue
        if not isinstance(judge_prompt_template, str) or not isinstance(judge_system_prompt, str):
            continue
        if not isinstance(response, str):
            continue

        try:
            judge_prompt = judge_prompt_template.format(
                prompt=prompt,
                response_reference=response_reference,
                response=response,
            )
        except Exception:
            continue

        jobs.append((i, judge_system_prompt, judge_prompt, prompt_id))

    if jobs:
        write_lock = threading.Lock()
        workers = max(1, min(int(max_workers), len(jobs)))
        start_ts = time.monotonic()
        last_len = 0
        last_print_ts = 0.0
        done = 0
        scored = 0

        line = progress_line("eval", 0, len(jobs), start_ts, extra=f"scored={scored}")
        last_len = len(line)
        sys.stdout.write(line + " " * 2)
        sys.stdout.flush()

        def _work(job: Tuple[int, str, str, str]) -> Tuple[int, Optional[Dict[str, Any]]]:
            idx, judge_system_prompt, judge_prompt, _prompt_id = job
            client = OpenAI(
                api_key=evaluator_cfg.get("api_key"),
                base_url=evaluator_cfg.get("base_url"),
            )
            reward_info = eval_single_record(
                call_llm_with_retries=call_llm_with_retries,
                client=client,
                evaluator_cfg=evaluator_cfg,
                judge_system_prompt=judge_system_prompt,
                judge_prompt=judge_prompt,
            )
            return idx, reward_info

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_work, job) for job in jobs]
            for fut in as_completed(futures):
                idx, reward_info = fut.result()
                done += 1
                if reward_info is None:
                    pass
                else:
                    with write_lock:
                        r = records[idx]
                        if isinstance(r, dict) and is_reward_info_empty(r.get("reward_info")):
                            records[idx] = dict(r)
                            records[idx]["reward_info"] = reward_info
                            atomic_write_jsonl(response_path, records)
                            scored += 1

                now = time.monotonic()
                if done == len(jobs) or (now - last_print_ts) >= 0.1:
                    last_print_ts = now
                    line = progress_line("eval", done, len(jobs), start_ts, extra=f"scored={scored}")
                    pad = " " * max(0, last_len - len(line))
                    last_len = len(line)
                    sys.stdout.write("\r" + line + pad)
                    sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()

    correct = 0
    total = 0
    for r in records:
        if not isinstance(r, dict):
            continue
        ri = r.get("reward_info")
        if not isinstance(ri, dict):
            continue
        score = ri.get("answer_score")
        if isinstance(score, int) and score in (0, 1):
            total += 1
            correct += int(score)
    return correct, total
