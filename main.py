import argparse
import copy
import datetime as _dt
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

_SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import yaml
from openai import OpenAI

from evaluator import run_evaluation
from prompts import build_default_system_prompt
from tools import TOOL_FN_MAP, tool_specs
from utils import json_dumps, json_dumps_pretty, load_jsonl_records, progress_line


def _expand_history_for_save(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    new_history = []
    for msg in history:
        new_msg = msg.copy()
        if msg.get("role") == "tool":
            content = msg.get("content")
            if isinstance(content, str):
                try:
                    new_msg["content"] = json.loads(content)
                except Exception:
                    pass
        new_history.append(new_msg)
    return new_history


_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env_refs(value: Any) -> Any:
    if isinstance(value, str):
        def _repl(m: re.Match[str]) -> str:
            var = m.group(1)
            if var not in os.environ:
                raise RuntimeError(f"环境变量未设置: {var}")
            return os.environ[var]

        return _ENV_REF_RE.sub(_repl, value)
    if isinstance(value, list):
        return [_resolve_env_refs(v) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_env_refs(v) for k, v in value.items()}
    return value


def _load_model_config(models_yaml_path: Path, model_id: str) -> Dict[str, Any]:
    raw = yaml.safe_load(models_yaml_path.read_text(encoding="utf-8"))
    models = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(models, list):
        raise RuntimeError("models.yaml 格式错误：缺少 models 列表")

    for m in models:
        if isinstance(m, dict) and m.get("name") == model_id:
            resolved = _resolve_env_refs(m)
            return resolved

    raise RuntimeError(f"没有找到model-id={model_id}的配置")


def _model_config_for_save(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(model_cfg)
    out.pop("api_key", None)
    return out


def _add_prompt_caching(messages: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    if not ("minimax" in model_name.lower() or "claude" in model_name.lower()):
        return messages

    cached_messages = copy.deepcopy(messages)

    for n in range(len(cached_messages)):
        if n < len(cached_messages) - 3:
            continue
        msg = cached_messages[n]
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")

        if isinstance(content, str):
            msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            continue

        if isinstance(content, list):
            for content_item in content:
                if isinstance(content_item, dict) and "type" in content_item:
                    content_item["cache_control"] = {"type": "ephemeral"}

    return cached_messages


def _call_llm_with_retries(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    temperature: Optional[float],
    max_tokens: Optional[int],
    extra_body: Optional[Dict[str, Any]],
    tool_choice: Optional[Any],
) -> Dict[str, Any]:
    last_err: Optional[BaseException] = None
    for attempt in range(1, 4):
        try:
            cached_messages = _add_prompt_caching(messages, model)
            resp = client.chat.completions.create(
                model=model,
                messages=cached_messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            return resp.model_dump()
        except BaseException as e:
            last_err = e
            if attempt >= 3:
                raise
            time.sleep(0.8 * (2 ** (attempt - 1)) + random.random() * 0.2)
    raise last_err or RuntimeError("LLM 调用失败")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _default_output_path(model_id: str, timestamp: str) -> Path:
    return Path("output") / model_id / f"{timestamp}.jsonl"


def _parse_sub_tasks(values: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if not values:
        return None
    if isinstance(values, str):
        values = [values]
    out: List[str] = []
    for v in values:
        parts = [p.strip() for p in v.split(",")] if "," in v else [v.strip()]
        for p in parts:
            if p:
                out.append(p)
    return out or None


def _load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError("dataset json 格式错误：期望顶层为 list")
    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "prompt_id" in item and "prompt" in item and "label" in item:
            out.append(item)
    return out


def _existing_prompt_ids(save_to: Path, dataset_ids: Iterable[str]) -> set:
    if not save_to.exists():
        return set()
    dataset_ids_set = set(dataset_ids)
    seen = set()
    for obj in load_jsonl_records(save_to):
        pid = obj.get("prompt_id")
        if isinstance(pid, str) and pid in dataset_ids_set:
            seen.add(pid)
    return seen


def _tool_names() -> List[str]:
    names: List[str] = []
    for spec in tool_specs():
        fn = spec.get("function") if isinstance(spec, dict) else None
        fn = fn if isinstance(fn, dict) else {}
        name = fn.get("name")
        if isinstance(name, str) and name not in names:
            names.append(name)
    return names


def _tool_call_count(history: List[Dict[str, Any]]) -> Dict[str, int]:
    names = _tool_names()
    counts: Dict[str, int] = {n: 0 for n in names}
    seen_ids_by_tool: Dict[str, set] = {n: set() for n in names}
    for msg in history:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            fn_name = fn.get("name")
            if not isinstance(tc_id, str) or not isinstance(fn_name, str):
                continue
            if fn_name not in counts:
                continue
            if tc_id in seen_ids_by_tool[fn_name]:
                continue
            seen_ids_by_tool[fn_name].add(tc_id)
            counts[fn_name] += 1
    return counts


def _assistant_message_count(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    assistant_msgs: List[Dict[str, Any]] = [
        m for m in history if isinstance(m, dict) and m.get("role") == "assistant"
    ]
    last = assistant_msgs[-1] if assistant_msgs else {}
    final_token = last.get("usage")
    return {"count": len(assistant_msgs), "final_token": final_token}



def _run_single_prompt(
    client: OpenAI,
    model_cfg: Dict[str, Any],
    prompt: str,
    max_steps: int,
    tool_executor: Optional[ThreadPoolExecutor],
    language: str = "en",
) -> Tuple[List[Dict[str, Any]], str]:
    system_prompt = build_default_system_prompt(prompt, language=language)

    history: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    tools = tool_specs()
    temperature = model_cfg.get("temperature")
    max_tokens = model_cfg.get("max_tokens")
    extra_body = model_cfg.get("extra_body")

    final_answer = ""
    usage_updates: List[Tuple[Dict[str, Any], Dict[str, int]]] = []

    for step in range(max_steps):
        tool_choice: Any = "auto"
        if step == max_steps - 1:
            history.append(
                {
                    "role": "user",
                    "content": "已达到最大工具调用步数。请停止调用任何工具，直接给出最终答案。",
                }
            )
            tool_choice = "none"

        resp = _call_llm_with_retries(
            client=client,
            model=model_cfg["name"],
            messages=history,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            tool_choice=tool_choice,
        )

        choice = (resp.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message") or {}

        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls") is not None:
            assistant_msg["tool_calls"] = message.get("tool_calls")
        if message.get("reasoning_details") is not None:
            assistant_msg["reasoning_details"] = message.get("reasoning_details")
        elif message.get("reasoning_content") is not None:
            assistant_msg["reasoning_content"] = message.get("reasoning_content")
        usage = resp.get("usage") if isinstance(resp, dict) else None
        if isinstance(usage, dict):
            usage_updates.append((assistant_msg, usage))
        history.append(assistant_msg)

        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if tool_calls and isinstance(tool_calls, list) and tool_choice != "none":
            def _invoke_tool(fn_name: Any, args: Dict[str, Any]) -> Any:
                if fn_name in TOOL_FN_MAP:
                    try:
                        return TOOL_FN_MAP[fn_name](**args)
                    except Exception as e:
                        return {"error": str(e), "tool": fn_name, "args": args}
                return {"error": f"unknown tool: {fn_name}", "args": args}

            results_by_id: Dict[Any, Any] = {}
            futures = []
            if tool_executor is not None and len(tool_calls) > 1:
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tc_id = tc.get("id")
                    fn = (tc.get("function") or {}) if isinstance(tc.get("function"), dict) else {}
                    fn_name = fn.get("name")
                    raw_args = fn.get("arguments")

                    args: Dict[str, Any] = {}
                    if isinstance(raw_args, str) and raw_args.strip():
                        try:
                            args = json.loads(raw_args)
                        except Exception:
                            args = {"_raw": raw_args}

                    futures.append((tc_id, tool_executor.submit(_invoke_tool, fn_name, args)))

                for tc_id, fut in futures:
                    try:
                        results_by_id[tc_id] = fut.result()
                    except Exception as e:
                        results_by_id[tc_id] = {"error": str(e), "tool_call_id": tc_id}
            else:
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tc_id = tc.get("id")
                    fn = (tc.get("function") or {}) if isinstance(tc.get("function"), dict) else {}
                    fn_name = fn.get("name")
                    raw_args = fn.get("arguments")

                    args: Dict[str, Any] = {}
                    if isinstance(raw_args, str) and raw_args.strip():
                        try:
                            args = json.loads(raw_args)
                        except Exception:
                            args = {"_raw": raw_args}
                    results_by_id[tc_id] = _invoke_tool(fn_name, args)

            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json_dumps(results_by_id.get(tc_id)),
                    }
                )
            continue

        content = message.get("content")
        if isinstance(content, str):
            final_answer = content.strip()

        if finish_reason == "stop":
            break

        if final_answer:
            break

    for msg, u in usage_updates:
        msg["usage"] = u

    return history, final_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=False, default="deepseek-reasoner")
    parser.add_argument("--sub-tasks", type=str, default="Complex_Historical_Investigation(Greater China)")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--save-to", default=None)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--llm-workers", type=int, default=30)
    parser.add_argument("--evaluator", required=False, default="kimi-k2.5")
    parser.add_argument("--response-file", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    dataset_path = root / "dataset" / "finsearchcomp_data.json"
    models_yaml_path = root / "models.yaml"
    evaluators_yaml_path = root / "evaluators.yaml"

    dataset = _load_dataset(dataset_path)
    dataset_by_id: Dict[str, Dict[str, Any]] = {}
    for d in dataset:
        pid = d.get("prompt_id")
        if isinstance(pid, str):
            dataset_by_id[pid] = d

    eval_mode = bool(args.evaluator) or bool(args.response_file)
    if eval_mode:
        if not args.evaluator or not args.response_file:
            raise RuntimeError("评估模式需要同时提供 --evaluator 和 --response-file")
        response_path = Path(args.response_file)
        if not response_path.exists():
            raise RuntimeError(f"response-file 不存在: {response_path}")

        evaluator_cfg = _load_model_config(evaluators_yaml_path, str(args.evaluator))
        evaluator_client = OpenAI(
            api_key=evaluator_cfg.get("api_key"),
            base_url=evaluator_cfg.get("base_url"),
        )
        correct, total = run_evaluation(
            call_llm_with_retries=_call_llm_with_retries,
            evaluator_client=evaluator_client,
            evaluator_cfg=evaluator_cfg,
            response_path=response_path,
            dataset_by_id=dataset_by_id,
        )
        acc = (correct / total) if total else 0.0
        print(f"accuracy={acc:.4f} ({correct}/{total})")
        return

    if not args.model_id:
        raise RuntimeError("生成模式需要提供 --model-id")

    model_cfg = _load_model_config(models_yaml_path, args.model_id)

    sub_tasks = _parse_sub_tasks(args.sub_tasks)
    if sub_tasks:
        dataset = [d for d in dataset if str(d.get("label", "")).strip() in set(sub_tasks)]

    dataset_ids = [str(d["prompt_id"]) for d in dataset if isinstance(d.get("prompt_id"), str)]

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_to = None
    if args.save_to:
        p = Path(args.save_to)
        if p.exists():
            save_to = p
    if save_to is None:
        save_to = (root / _default_output_path(args.model_id, timestamp)).resolve()
    _ensure_parent_dir(save_to)

    seen = _existing_prompt_ids(save_to, dataset_ids)
    remaining = [d for d in dataset if d.get("prompt_id") not in seen]
    if args.num_tasks is not None:
        remaining = remaining[: max(0, int(args.num_tasks))]

    task_config = {"max_steps": int(args.max_steps), "sub_tasks": sub_tasks}
    model_config_for_save = {"model_id": args.model_id, **_model_config_for_save(model_cfg)}

    target_language = "en"
    if "Greater China" in args.sub_tasks:
        target_language = "zh"

    print(f"save_to={save_to}")
    print(f"dataset_total={len(dataset)} skipped_existing={len(seen)} remaining={len(remaining)}")
    if sub_tasks:
        print(f"sub_tasks={json_dumps(sub_tasks)}")

    if not remaining:
        if sub_tasks and len(dataset) == 0:
            print("没有匹配到任何数据：请检查 --sub-tasks 是否与 dataset 的 label 一致")
        else:
            print("没有待运行任务：可能全部已在 save_to 中存在，或 --num-tasks=0")
        return

    llm_workers = int(args.llm_workers)
    if llm_workers < 1:
        raise RuntimeError("--llm-workers 必须 >= 1")
    tool_workers = min(128, max(16, llm_workers * 2))
    write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=tool_workers) as tool_ex:
        with ThreadPoolExecutor(max_workers=llm_workers) as llm_ex:
            with save_to.open("a", encoding="utf-8") as out_f:
                futures = []
                start_ts = time.monotonic()
                done = 0
                succeeded = 0
                failed = 0
                last_len = 0
                last_print_ts = 0.0

                def _work(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                    prompt_id = item.get("prompt_id")
                    prompt = item.get("prompt")
                    if not isinstance(prompt_id, str) or not isinstance(prompt, str):
                        return None
                    client = OpenAI(
                        api_key=model_cfg.get("api_key"),
                        base_url=model_cfg.get("base_url"),
                    )
                    history, final_answer = _run_single_prompt(
                        client=client,
                        model_cfg=model_cfg,
                        prompt=prompt,
                        max_steps=int(args.max_steps),
                        tool_executor=tool_ex,
                        language=target_language,
                    )
                    if not final_answer:
                        return None
                    tool_call_count = _tool_call_count(history)
                    assistant_message_count = _assistant_message_count(history)
                    return {
                        "task_config": task_config,
                        "model_config": model_config_for_save,
                        "prompt_id": prompt_id,
                        "llm_response": final_answer,
                        "tool_call_count": tool_call_count,
                        "assistant_message_count": assistant_message_count,
                        "reward_info": None,
                        "dialogue": _expand_history_for_save(history)
                    }

                for item in remaining:
                    futures.append(llm_ex.submit(_work, item))

                total = len(futures)
                if total:
                    line = progress_line("run", 0, total, start_ts, extra=f"success={succeeded} fail={failed}")
                    last_len = len(line)
                    sys.stdout.write(line + " " * 2)
                    sys.stdout.flush()

                for fut in as_completed(futures):
                    try:
                        record = fut.result()
                    except Exception as e:
                        record = None
                        failed += 1
                        sys.stderr.write(f"\nworker_error={type(e).__name__}: {e}\n")
                        sys.stderr.flush()
                    done += 1
                    if record:
                        with write_lock:
                            out_f.write(json_dumps_pretty(record) + "\n")
                            out_f.flush()
                        succeeded += 1

                    now = time.monotonic()
                    if done == total or (now - last_print_ts) >= 0.1:
                        last_print_ts = now
                        line = progress_line("run", done, total, start_ts, extra=f"success={succeeded} fail={failed}")
                        pad = " " * max(0, last_len - len(line))
                        last_len = len(line)
                        sys.stdout.write("\r" + line + pad)
                        sys.stdout.flush()

                if total:
                    sys.stdout.write("\n")
                    sys.stdout.flush()


if __name__ == "__main__":
    main()
