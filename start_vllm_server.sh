cd /home/namb/nambkh 
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Llama-3.3-70B-Instruct \
--dtype bfloat16 \
--disable-log-stats \
--disable-log-requests \
--enable-chunked-prefill \
--enable-prefix-caching \
--gpu-memory-utilization 0.9 \
--host 0.0.0.0 \
--port 8148 \
--seed 148 \
--tensor-parallel-size 4 \