# finsearchcomp_exp

Run FinSearchComp tasks end-to-end with an LLM + lightweight web tools, save results to JSONL, and optionally score them with an evaluator model.

## Repository Layout

- [main.py](main.py): CLI entrypoint for generation and evaluation.
- [dataset/finsearchcomp_data.json](dataset/finsearchcomp_data.json): Task dataset (each item has `prompt_id`, `label`, `prompt`, and judge templates).
- [src/tools.py](src/tools.py): Tool implementations and tool specs (`web_search_chinese`, `web_search_global`, `web_content`).
- [src/llm.py](src/llm.py): OpenAI-compatible chat call wrapper + retries.
- [src/memory_compress.py](src/memory_compress.py): 3-layer memory compression for tool-heavy runs.
- [src/web_summary.py](src/web_summary.py): LLM-based web text summarizer used by compression.
- [src/evaluator.py](src/evaluator.py): Scoring pipeline for saved responses.
- [models.yaml](models.yaml): Generation model configs (API key via env vars).
- [evaluators.yaml](evaluators.yaml): Evaluator model configs.
- [src/llm_context_window.json](src/llm_context_window.json): Context window per model ID.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install tiktoken
```

Notes:
- `tiktoken` is required because memory compression uses token counting.

## Environment Variables

### Generation / Evaluation model keys

Pick the variables you need based on the model(s) you run. `models.yaml` / `evaluators.yaml` use `${VAR_NAME}`.

- `OPENROUTER_API_KEY` (for OpenRouter models like `openai/gpt-5.2`, `anthropic/...`, `google/...`)
- `DEEPSEEK_API_KEY`
- `GLM_API_KEY`
- `MINIMAX_API_KEY`
- `MOONSHOT_API_KEY`
- `ARK_API_KEY`
- `DASHSCOPE_API_KEY` (also used by `src/web_summary.py`)

### Tool keys (web tools)

- `GLM_API_KEY` (used by `web_search_chinese`)
- `PARALLEL_API_KEY` (used by `web_search_global`)
- `EXA_API_KEY` (used by `web_content`)

## Run: Generation Mode

Generate answers for tasks and append JSON records to an output JSONL file.

```bash
python main.py --model-id deepseek-reasoner
```

### Task selection

`main.py` filters tasks by:
1. `--sub-tasks` (matches dataset item `label`)
2. `--task-ids` (matches dataset item `prompt_id`)

You can use either one or both; when both are provided, the final set is the intersection.

Examples:

```bash
python main.py --model-id deepseek-reasoner --sub-tasks "Simple_Historical_Lookup(Greater China)"
```

```bash
python main.py --model-id deepseek-reasoner --task-ids "(T2)Simple_Historical_Lookup_001,(T2)Simple_Historical_Lookup_002"
```

```bash
python main.py --model-id deepseek-reasoner --sub-tasks "Simple_Historical_Lookup(Greater China)" --task-ids "(T2)Simple_Historical_Lookup_001"
```

### Output

By default, results are appended to:

`output/<model-id>/<timestamp>.jsonl`

Each JSONL record contains:

- `prompt_id`
- `llm_response`
- `tool_call_count` and `assistant_message_count`
- `dialogue` (expanded tool outputs are parsed back from JSON strings)
- `reward_info` (reserved for evaluation; initially `null`)

If `--save-to <path>` points to an existing file, the run will:
- load existing `prompt_id`s from that file
- skip tasks already present (append only new ones)

### Key CLI flags

- `--model-id`: model name to load from `models.yaml`
- `--sub-tasks`: comma-separated labels (must match `dataset[].label`)
- `--task-ids`: comma-separated prompt IDs (must match `dataset[].prompt_id`)
- `--save-to`: append to an existing JSONL file, or default output path if omitted
- `--max-steps`: max tool-call loop steps per task (default 50)
- `--llm-workers`: number of parallel tasks (default 30)

## Run: Evaluation Mode

Evaluation mode is enabled when `--response-file` is provided. It scores existing records in that JSONL file using an evaluator model and writes back `reward_info`.

```bash
python main.py --response-file output\deepseek-reasoner\20260101_010101.jsonl --evaluator deepseek-chat
```

Notes:
- If `--evaluator` is omitted, it defaults to `deepseek-chat`.
- Evaluator configs are loaded from `evaluators.yaml`.

## Tools

Tools are available to the generation model via OpenAI tool calls:

- `web_search_chinese(query)`: web search (Chinese queries)
- `web_search_global(query)`: web search (non-Chinese queries)
- `web_content(url)`: fetch a page’s extracted text (and optional summary, depending on provider)

See [src/tools.py](src/tools.py) for provider endpoints and response shapes.

## Memory Compression (3 layers)

Tool-heavy runs can exceed the model context window. This project uses three layers of compression:

1. **Before each LLM call**: replace older tool result messages with a short placeholder  
   - `remove_tool_call_results_from_messages` in [src/memory_compress.py](src/memory_compress.py)
2. **Before each LLM call**: summarize large `web_content` tool results kept in the outgoing messages  
   - `compress_web_content` in [src/memory_compress.py](src/memory_compress.py)
3. **After `web_content` returns (before saving into history)**: optionally summarize the tool result based on context window pressure  
   - `apply_web_summary` in [src/memory_compress.py](src/memory_compress.py)

The summarizer is implemented in [src/web_summary.py](src/web_summary.py) and uses the `DASHSCOPE_API_KEY`.

