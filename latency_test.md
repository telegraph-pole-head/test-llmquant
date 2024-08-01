## Latency Test

1. serving in vllm:
```bash
python -m vllm.entrypoints.openai.api_server --model /data/llmQuantModels/gemma-2-9b-it-GPTQ --tensor-parallel-size 1 --dtype auto --gpu-memory-utilization 0.4 --port 8000 #--enforce_eager  --max-model-len=4096
```
2. test with lmuses:
```bash
llmuses perf --url 'http://localhost:8000/v1/chat/completions' --model '/data/llmQuantModels/gemma-2-9b-it-GPTQ' -n 1 --api openai --stream --stop '<|im_end|>' --debug --wandb-api-key 'bdcadde280255d759c2abeb2a9d4b364437b6104' --name 'gemma-2-9b-it-GPTQ' --dataset openqa --dataset-path '/data/llmQuantModels/dataset/open_qa.jsonl' --max-prompt-length 128000
```
bdcadde280255d759c2abeb2a9d4b364437b6104 --dataset-path './datasets/open_qa.jsonl' --max-prompt-length 4096 --dataset openqa 

3.livebench:
```bash
python gen_api_answer.py --model '/data/llmQuantModels/phi-3-mini-AWQ' --bench-name live_bench --api-base http://localhost:8000/v1
```
4. lm eval:
```bash
lm_eval --model local-chat-completions --tasks wikitext --model_args model=/data/llmQuantModels/gemma-2-9b-it-GPTQ,num_concurrent=1,max_retries=3,tokenized_requests=False,base_url=http://0.0.0.0:8000/v1/chat --wandb_args project=llm_quant_tests --log_samples --output_path wandb/
```
```bash
CUDA_VISIBLE_DEVICES=1 lm_eval --model hf --model_args pretrained=/data/llmQuantModels/gemma-2-9b-it-GPTQ,autogptq=model.safetensors,gptq_use_triton=True --tasks wikitext --wandb_args project=llm_quant_tests --log_samples --output_path wandb/ --device cuda:1
```

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_ATTENTION_BACKEND=FLASHINFER lm_eval --model vllm --model_args pretrained=/data/llmQuantModels/gemma-2-9b-it-GPTQ,dtype=auto,gpu_memory_utilization=0.43,max_model_len=2048 --tasks wikitext --wandb_args project=llm_quant_tests --log_samples --output_path wandb/ --device cuda:1
```

```bash
CUDA_VISIBLE_DEVICES=1 lm_eval --model vllm --model_args pretrained=/data/llmQuantModels/Llama-2-13B-GPTQ-4bit,dtype=auto,gpu_memory_utilization=0.45,max_model_len=2048 --tasks wikitext --wandb_args project=llm_quant_tests --log_samples --output_path wandb/ --device cuda:1
```

```
VLLM_ATTENTION_BACKEND=FLASHINFER python -m flute.integrations.vllm vllm.entrypoints.openai.api_server \
    --model /data/llmQuantModels/gemma-2-9b-it-Flute\
    --quantization flute\
    --gpu-memory-utilization 0.45

VLLM_ATTENTION_BACKEND=FLASHINFER python -m vllm.entrypoints.openai.api_server --model /data/llmQuantModels/gemma-2-9b-it-GPTQ --tensor-parallel-size 1 --dtype auto --gpu-memory-utilization 0.45

docker run -d -p 3210:3210 \
  -e OPENAI_API_KEY=EMPTY \
  -e OPENAI_PROXY_URL=http://localhost:8000/v1 \
  -e ACCESS_CODE=lobe66 \
  --name lobe-chat \
  docker.m.daocloud.io/lobehub/lobe-chat:latest
```
