export CUDA_VISIBLE_DEVICES="0,1,2,3"
deepspeed   --num_nodes=1 \
            z_eval.py \
            --deepspeed_config=deepspeed_config.json