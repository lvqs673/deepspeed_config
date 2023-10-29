export TOKENIZERS_PARALLELISM=true
deepspeed   --include=localhost:1,3,4 \
            --master_port=29501 \
            ./src/train.py \
            --deepspeed_config=deepspeed_config.json \
            --dataset=./data/sentences.txt \
            --nepoch=5 \
            --model_save_dir=./model \
            --model_name='epoch_{}.pt' \
            > training.log