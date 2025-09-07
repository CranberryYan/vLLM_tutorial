import time
from vllm import LLM, SamplingParams

MODEL_PATH = \
    "/home/workspace/vLLM_tutorial/modelzoo/DeepSeek-R1-Distill-Qwen-1.5B"
NUM_REQUESTS        = 128                     # 并发请求数
LONG_PROMPT_TOKENS  = 1024                    # Prefill heavy 长上下文
PREFILL_MAX_TOKENS  = 16                      # Prefill heavy 生成更短
SHORT_PROMPT_TOKENS = 10                      # Decode heavy 短上下文
DECODE_MAX_TOKENS   = 512                     # Decode heavy 生成更长
TEMPERATURE = 0.0

llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, max_num_seqs=NUM_REQUESTS)
tok = llm.get_tokenizer()

def make_test_from_n_token(n: int) -> str:
  # 生成一个大概 n token 的字符串
  base = "机器学习让世界更美好"
  s = (base * ((n * 4) // len(base) + 1))[: n * 4]
  ids = tok.encode(s)
  return tok.decode(ids[:n])

def run_case(name: str, prompts, max_tokens: int):
  sp = SamplingParams(
    temperature=TEMPERATURE,
    max_tokens=max_tokens,
    top_p=1.0,
    stop=None
  )
  t0 = time.time()
  outputs = llm.generate(prompts, sp)
  t1 = time.time()

  # 统计输入/输出 token 数
  in_tokens = sum(len(o.prompt_token_ids) for o in outputs)
  out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

  dur = t1 - t0
  ips = in_tokens / dur if dur > 0 else 0.0
  ops = out_tokens / dur if dur > 0 else 0.0

  print(f"\n== {name} ==")
  print(f"Requests: {len(prompts)} | Time: {dur:.2f}s")
  print(f"Input tokens: {in_tokens}  ({ips:.1f} tok/s)")
  print(f"Output tokens: {out_tokens} ({ops:.1f} tok/s)")

# Prefill heavy: 长 prompt, 短生成
long_pormpts = make_test_from_n_token(LONG_PROMPT_TOKENS)
prompts_prefill = [long_pormpts for _ in range(NUM_REQUESTS)] # 构建 128 个用户

# Decode heavy: 短 promp, 长生成
long_pormpts = make_test_from_n_token(SHORT_PROMPT_TOKENS)
prompts_decode = [long_pormpts for _ in range(NUM_REQUESTS)]

print("Warmup...")
run_case("WARMUP (short prompt, few tokens)", prompts_decode[:8], 8)

print("\nRunning PREFILL-HEAVY ...")
run_case("PREFILL-HEAVY (long prompt, short decode)",
         prompts_prefill, PREFILL_MAX_TOKENS)

print("\nRunning DECODE-HEAVY ...")
run_case("DECODE-HEAVY (short prompt, long decode)",
         prompts_decode, DECODE_MAX_TOKENS)
