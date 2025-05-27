export SCRATCH=/home/scratch.******
# smc521ge-0036
export MODEL=$SCRATCH/models/deepseek-r1-nvfp4-wo
export TRTLLM_USE_UCX_KVCACHE=1
export TRTLLM_UCX_INTERFACE=ens4np0,ens5np0,enp172s0np0,enp188s0np0
export TRTLLM_KVCACHE_TIME_OUTPUT_PATH=kvcache_time_context

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trtllm-serve $MODEL --host smc521ge-0036 --port 8001 \
 --max_batch_size 16 --max_num_tokens 5220 --max_seq_len 5320 --tp_size 8 --ep_size 8 --kv_cache_free_gpu_memory_fraction 0.6 \
 --backend pytorch --extra_llm_api_options extra-llm-api-config.yml &
trtllm-serve disaggregated --request_timeout 3600 --server_start_timeout 2400 -c disagg_config_multinode.yml &

# wait for the server to start
# send a small test request
curl http://smc521ge-0036:8000/v1/completions     -H "Content-Type: application/json"     -d "{ \"model\": \"$MODEL\", \"prompt\": \"NVIDIA is a great company because\", \"max_tokens\": 16, \"temperature\": 0 }" -w "\n"
# send some real requests
python tensorrt_llm/serve/scripts/benchmark_serving.py --request-rate 10 --ignore-eos --host smc521ge-0036 --port 8000 --model $MODEL --dataset-name file --dataset-path $SCRATCH/envs/sm120/dataset/dataset_4000_serving.json --num-prompts 512
# send random requests
python tensorrt_llm/serve/scripts/benchmark_serving.py --ignore-eos --host smc521ge-0036 --port 8000 --model $MODEL --dataset-name random --random-input-len 256 --random-output-len 128 --num-prompts 512

############################################################################################
# smc521ge-0038
#$SCRATCH/models/deepseek-r1-nvfp4-wo
#/home/scratch.trt_llm_data/llm-models/DeepSeek-V3-Lite/nvfp4_moe_only/
export MODEL=$SCRATCH/models/deepseek-r1-nvfp4-wo
export TRTLLM_USE_UCX_KVCACHE=1
export TRTLLM_UCX_INTERFACE=ens4np0,ens5np0,enp172s0np0,enp188s0np0
export TRTLLM_KVCACHE_TIME_OUTPUT_PATH=kvcache_time_gen

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trtllm-serve $MODEL --host smc521ge-0038 --port 8002 \
 --max_batch_size 16 --max_num_tokens 5220 --max_seq_len 5320 --tp_size 8 --ep_size 8 --kv_cache_free_gpu_memory_fraction 0.80 \
 --backend pytorch --extra_llm_api_options extra-llm-api-config.yml | tee gen.log
