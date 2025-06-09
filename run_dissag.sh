#!/bin/bash
SCRATCH=/home/*
if [ ! -d "$SCRATCH" ]; then
  echo "SCRATCH directory does not exist"
  exit 1
fi

#$SCRATCH/models/deepseek-r1-nvfp4-wo
# MODEL=/home/scratch.trt_llm_data/llm-models/DeepSeek-V3-Lite/nvfp4_moe_only/
MODEL=$SCRATCH/models/deepseek-r1-nvfp4-wo
export TRTLLM_USE_UCX_KVCACHE=1
export TRTLLM_UCX_INTERFACE=ens4np0,ens5np0,enp172s0np0,enp188s0np0
export UCX_NET_DEVICES=ens4np0,ens5np0,enp172s0np0,enp188s0np0
export TRTLLM_KVCACHE_TIME_OUTPUT_PATH=kvcache_time_context
SIZE=8

HOSTNAME=$(hostname)
# remove trailing "devel" from hostname if it exists
if [[ $HOSTNAME == *"-devel" ]]; then
  HOSTNAME=${HOSTNAME%-devel}
fi

function context() {
  pkill -9 trtllm-serve || true
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trtllm-serve $MODEL --host $HOSTNAME --port 8001 \
  --max_batch_size 16 --max_num_tokens 5220 --max_seq_len 5320 --tp_size $SIZE --ep_size $SIZE --kv_cache_free_gpu_memory_fraction 0.6 \
  --backend pytorch --extra_llm_api_options extra-llm-api-config.yml &
  trtllm-serve disaggregated --request_timeout 3600 --server_start_timeout 2400 -c disagg_config_multinode.yml &
}

function request() {
  TYPE="$1"
  if [ -z "$TYPE" ]; then
    TYPE="hello"
  fi

  case $TYPE in
    hello)
      # send a small test request
      curl http://$HOSTNAME:8000/v1/completions     -H "Content-Type: application/json"     -d "{ \"model\": \"$MODEL\", \"prompt\": \"NVIDIA is a great company because\", \"max_tokens\": 16, \"temperature\": 0 }" -w "\n"
      ;;
    real)
      # send some real requests
      python tensorrt_llm/serve/scripts/benchmark_serving.py --request-rate 10 --ignore-eos --host $HOSTNAME --port 8000 --model $MODEL --dataset-name file --dataset-path $SCRATCH/envs/sm120/dataset/dataset_4000_serving.json --num-prompts 1024
      ;;
    random)
      # send random requests
      python tensorrt_llm/serve/scripts/benchmark_serving.py --ignore-eos --host $HOSTNAME --port 8000 --model $MODEL --dataset-name random --random-input-len 256 --random-output-len 128 --num-prompts 512
      ;;
    *)
      echo "Unknown request type: $TYPE"
      echo "Available types: hello, real, random"
      exit 1
      ;;
  esac
}

function generation() {
  pkill -9 trtllm-serve || true
  TRTLLM_KVCACHE_TIME_OUTPUT_PATH=kvcache_time_context PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trtllm-serve $MODEL --host $HOSTNAME --port 8002 \
  --max_batch_size 16 --max_num_tokens 5220 --max_seq_len 5320 --tp_size $SIZE --ep_size $SIZE --kv_cache_free_gpu_memory_fraction 0.80 \
  --backend pytorch --extra_llm_api_options extra-llm-api-config.yml &
}

if [ -z "$1" ]; then
  echo "Usage: $0 <command>"
  echo "Available commands: context, request, generation"
  exit 1
fi

case $1 in
  context)
    context
    ;;
  request)
    if [ -z "$2" ]; then
      echo "Usage: $0 request <type>"
      echo "Available types: hello, real, random"
      exit 1
    fi
    request $2
    ;;
  generation)
    generation
    ;;
  *)
    echo "Unknown command: $1"
    echo "Available commands: context, request, generation"
    exit 1
    ;;
esac
