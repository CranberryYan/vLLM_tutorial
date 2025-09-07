import asyncio
import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

MODEL_PATH = "/home/workspace/vLLM_tutorial/modelzoo/DeepSeek-R1-Distill-Qwen-1.5B"

NUM_REQ          = 128          # 总请求数
SHORT_TOKENS     = 10           # 短 prompt（decode 重）
LONG_TOKENS      = 1024         # 长 prompt（prefill 重）
GEN_TOKENS       = 256          # 生成长度
ARRIVAL_MODE     = "poisson"    # "burst" or "poisson"
POISSON_RATE     = 64           # 泊松到达强度 λ (req/秒)，越大表示更“拥挤”
SEED             = 1234
MIX_RATIO_LONG   = 0.5          # 长/短比例：长占 50%

random.seed(SEED)

def synthetic_prompt(n_tokens: int) -> str:
  base = "一句话解释vLLM的优势"
  return base * math.ceil(n_tokens / 4)

@dataclass
class Record:
  req_id: str
  cls: str                 # "short" or "long"
  t_submit: float
  t_first_token: float
  t_finish: float
  in_tokens: int
  out_tokens: int

def _pct(a, p):
  a_sorted = sorted(a)
  if not a_sorted:
    return 0.0
  k = int(len(a_sorted) * p)
  k = min(max(k, 0), len(a_sorted)-1)
  return a_sorted[k]

def _summ_one(records: List[Record], title: str):
  if not records:
    print(f"\n== {title} == (no records)")
    return
  dur_total = max(r.t_finish for r in records) - min(r.t_submit for r in records)
  total_in = sum(r.in_tokens for r in records)
  total_out = sum(r.out_tokens for r in records)
  ips = total_in / dur_total if dur_total > 0 else 0.0
  ops = total_out / dur_total if dur_total > 0 else 0.0
  latencies = [r.t_finish - r.t_submit for r in records]
  ttfts = [r.t_first_token - r.t_submit for r in records]

  print(f"\n== {title} ==")
  print(f"Requests: {len(records)} | Window: {dur_total:.2f}s")
  print(f"Tokens In:  {total_in} ({ips:.1f} tok/s)")
  print(f"Tokens Out: {total_out} ({ops:.1f} tok/s)")
  print(f"Latency (P50/P90/P99): {statistics.median(latencies):.2f}s / {_pct(latencies,0.9):.2f}s / {_pct(latencies,0.99):.2f}s")
  print(f"TTFT   (P50/P90/P99): {statistics.median(ttfts):.2f}s / {_pct(ttfts,0.9):.2f}s / {_pct(ttfts,0.99):.2f}s")

def summarize_all_and_groups(records: List[Record], title: str):
  _summ_one(records, title)
  # 分组统计
  groups: Dict[str, List[Record]] = {"short": [], "long": []}
  for r in records:
    groups.setdefault(r.cls, []).append(r)
  for cls_name, recs in groups.items():
    _summ_one(recs, f"{title} — [{cls_name}]")

def make_arrivals(mode: str) -> Tuple[str, List[float]]:
  if mode == "burst":
    return "Burst (all at t=0)", [0.0 for _ in range(NUM_REQ)]
  elif mode == "poisson":
    times = []
    t = 0.0
    for _ in range(NUM_REQ):
      gap = random.expovariate(POISSON_RATE)  # 平均 1/λ 秒一个
      t += gap
      times.append(t)
    t0 = times[0]
    times = [x - t0 for x in times]
    return f"Poisson (lambda={POISSON_RATE}/s)", times
  else:
    raise ValueError("ARRIVAL_MODE must be 'burst' or 'poisson'")

async def run_scenario(arrival_times: List[float]):
  # 1) 准备引擎
  ae = AsyncEngineArgs(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    max_num_seqs=NUM_REQ,
  )
  engine = AsyncLLMEngine.from_engine_args(ae)

  sampling = SamplingParams(max_tokens=GEN_TOKENS, temperature=0.0, top_p=1.0)

  # 2) 为每个请求分配类别与对应 prompt
  num_long = int(NUM_REQ * MIX_RATIO_LONG)
  num_short = NUM_REQ - num_long
  # 固定前 num_long 个为长，剩下为短；也可打乱顺序
  classes = (["long"] * num_long) + (["short"] * num_short)
  random.shuffle(classes)

  prompts_by_cls = {
    "short": synthetic_prompt(SHORT_TOKENS),
    "long": synthetic_prompt(LONG_TOKENS),
  }

  records: List[Record] = []
  t0 = time.perf_counter()

  async def submit_and_collect(idx: int, t_rel: float, cls: str):
    await asyncio.sleep(max(0.0, t_rel))
    req_id = f"req-{idx}"
    t_submit = time.perf_counter()
    t_first = None
    in_tok = 0
    out_tok = 0
    prompt = prompts_by_cls[cls]

    async for result in engine.generate(
      prompt=prompt,
      sampling_params=sampling,
      request_id=req_id,
    ):
      if getattr(result, "prompt_token_ids", None) is not None:
          in_tok = len(result.prompt_token_ids)
      if getattr(result, "outputs", None):
          out_tok = len(result.outputs[0].token_ids)
          if t_first is None and out_tok > 0:
            t_first = time.perf_counter()
      if getattr(result, "finished", False):
          t_done = time.perf_counter()
          records.append(Record(
            req_id=req_id,
            cls=cls,
            t_submit=t_submit - t0,
            t_first_token=(t_first if t_first is not None else t_done) - t0,
            t_finish=t_done - t0,
            in_tokens=in_tok,
            out_tokens=out_tok,
        ))
          break

  tasks = [
    asyncio.create_task(submit_and_collect(i, at, classes[i]))
    for i, at in enumerate(arrival_times)
  ]
  await asyncio.gather(*tasks)
  return records

async def main():
  title_pois, arrivals_pois = make_arrivals("poisson")

  print("Running Mixed-Length Poisson scenario...")
  rec_pois = await run_scenario(arrivals_pois)
  summarize_all_and_groups(rec_pois, f"{title_pois} (mixed {int(MIX_RATIO_LONG*100)}% long)")

if __name__ == "__main__":
  asyncio.run(main())
