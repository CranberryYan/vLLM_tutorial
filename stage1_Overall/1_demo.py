from vllm import LLM, SamplingParams

llm = LLM(model="vLLM_tutorial/modelzoo/DeepSeek-R1-Distill-Qwen-1.5B")
sampling_params = SamplingParams(temperature=0.7, top_p=0.9)

outputs = llm.generate(["用一句话解释一下vLLM"], sampling_params)
for output in outputs:
  print(output.outputs[0].text)
