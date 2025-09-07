import asyncio
import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import List, Tuple

from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

MODEL_PATH = \
    "/home/workspace/vLLM_tutorial/modelzoo/DeepSeek-R1-Distill-Qwen-1.5B"
NUM_REQ       = 128                 # 请求总数
PROMPT_TOKENS = 1024                  # 短上下文 -> decode 重
GEN_TOKENS    = 256                 # 生成长度
ARRIVAL_MODE  = "poisson"           # "burst" or "poisson"
POISSON_RATE  = 64                  # 泊松到达强度 λ (req/秒)，越大表示更“拥挤”
SEED          = 1234

random.seed(SEED)

def synthetic_prompt(n_tokens: int) -> str:
  base = "一句话解释vLLM的优势"
  return base * math.ceil(n_tokens / 4)

@dataclass # 相当于 Cpp 的 struct,
           # python 自动生成 __init__ 构造函数, __repr__ 打印等
class Record:
  # 成员变量
  req_id: str
  t_submit: float
  t_first_token: float
  t_finish: float
  in_tokens: int
  out_tokens: int

# List[Record]: std::vector<Record> records
def summarize(records: List[Record], title: str):
  dur_total = \
    max(r.t_finish for r in records) - min(r.t_submit for r in records)
  total_in = sum(r.in_tokens for r in records)
  total_out = sum(r.out_tokens for r in records)
  ips = total_in / dur_total if dur_total > 0 else 0.0
  ops = total_out / dur_total if dur_total > 0 else 0.0
  latencies = [r.t_finish - r.t_submit for r in records]
  ttfts = [r.t_first_token - r.t_submit for r in records]

  # 求 a 的 p 分位点
  def pct(a, p):
    a_sorted = sorted(a)
    k = int(len(a_sorted) * p)
    k = min(max(k, 0), len(a_sorted)-1)
    return a_sorted[k]

  print(f"\n== {title} ==")
  print(f"Requests: {len(records)} | Window: {dur_total:.2f}s")
  print(f"Tokens In:  {total_in} ({ips:.1f} tok/s)")
  print(f"Tokens Out: {total_out} ({ops:.1f} tok/s)")
  print(f"Latency (P50/P90/P99): {statistics.median(latencies):.2f}s / {pct(latencies,0.9):.2f}s / {pct(latencies,0.99):.2f}s")
  print(f"TTFT   (P50/P90/P99): {statistics.median(ttfts):.2f}s / {pct(ttfts,0.9):.2f}s / {pct(ttfts,0.99):.2f}s")

# async def: 异步函数
# 在 Python 里, 异步函数必须用 await 来调用
async def run_scenario(arrival_times: List[float]):
  # 1) v1: 用 AsyncEngineArgs 构造 AsyncLLM（同步创建，不要 await）
  ae = AsyncEngineArgs(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    max_num_seqs=NUM_REQ,
  )
  engine = AsyncLLMEngine.from_engine_args(ae)

  # v1: 采样参数仍用 SamplingParams，但 add_request/generate 的参数名叫 params
  sampling = SamplingParams(max_tokens=GEN_TOKENS, temperature=0.0, top_p=1.0)
  prompt = synthetic_prompt(PROMPT_TOKENS)

  records: List[Record] = []
  t0 = time.perf_counter()

  async def submit_and_collect(idx: int, t_rel: float):
    # 等到相对到达时间
    await asyncio.sleep(max(0.0, t_rel))
    req_id = f"req-{idx}"
    t_submit = time.perf_counter()
    t_first = None
    in_tok = 0
    out_tok = 0

    # v1 正确用法：对 generate(...) 做 async for（逐步产出该 request 的输出）
    async for result in engine.generate(
      prompt=prompt,
      sampling_params=sampling,
      request_id=req_id,
    ):
      # 输入 token 数
      if getattr(result, "prompt_token_ids", None) is not None:
        in_tok = len(result.prompt_token_ids)
      # 输出 token 数（取第一条候选）
      if getattr(result, "outputs", None):
        out_tok = len(result.outputs[0].token_ids)
        if t_first is None and out_tok > 0:
          t_first = time.perf_counter()
      # 该请求完成
      if getattr(result, "finished", False):
        t_done = time.perf_counter()
        records.append(Record(
          req_id=req_id,
          t_submit=t_submit - t0,
          t_first_token=(t_first if t_first is not None else t_done) - t0,
          t_finish=t_done - t0,
          in_tokens=in_tok,
          out_tokens=out_tok,
        ))
        break  # 退出该请求的消费循环

  # 并发跑所有请求
  tasks = [asyncio.create_task(submit_and_collect(i, at))
            for i, at in enumerate(arrival_times)]
  await asyncio.gather(*tasks)

  return records

def make_arrivals(mode: str) -> Tuple[str, List[float]]:
  if mode == "burst":
    # 全部同时到达
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

async def main():
  title_burst, arrivals_burst = make_arrivals("burst")
  title_pois, arrivals_pois   = make_arrivals("poisson")

  print("Running Burst scenario...")
  rec_burst = await run_scenario(arrivals_burst)
  summarize(rec_burst, title_burst)

  print("\nRunning Poisson scenario...")
  rec_pois = await run_scenario(arrivals_pois)
  summarize(rec_pois, title_pois)

if __name__ == "__main__":
  asyncio.run(main())