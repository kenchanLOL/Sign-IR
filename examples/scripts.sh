
cd ./GFSLT-VLP
# cd examples/GFSLT-VLP

# # check SignCL pretrained
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1236 --use_env train_vlp_v2.py --batch-size 4 --epochs 80 --opt sgd --lr 0.01 --output_dir out/0626_csl_VLP_SignCL --training-refurbish True --noise-rate 0.15 --noise-type omit_last --random-shuffle False --decoder-type LLMD --config ./configs/config_gloss_free_csl.yaml

#GF-VLP
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=1 --master_port=1036 train_vlp_v2.py --batch-size 8 --epochs 80 --opt sgd --lr 0.01 --output_dir out/GF_vlp_b16  --config ./configs/config_gf_vlp.yaml --training-refurbish True --noise-rate 0.15 --noise-type omit_last --random-shuffle False --decoder-type LLMD

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1036 train_slt.py --batch-size 8 --epochs 200 --opt sgd --lr 0.01 --output_dir out/Gloss-Free \
# --finetune ./out/GF_vlp_b16/best_checkpoint.pth --config ./configs/config_gf_vlp.yaml

# # check best approach on CSL-Daily 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=1036 --use_env train_csl_SignCL.py --batch-size 2 --epochs 200 --opt sgd --lr 0.0065 --output_dir out/   0630_GF_SignCL \
# --config ./configs/config_gloss_free_csl.yaml --finetune out/0626_csl_VLP_SignCL/best_checkpoint.pth --decoder-type LLMD

# eval
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1036 --use_env train_csl_SignCL.py --batch-size 2 --epochs 0 --opt sgd --lr 0.0065 --output_dir out/0630_GF_SignCL \
# --config ./configs/config_gloss_free_csl_char.yaml --resume out/0630_GF_SignCL/best_checkpoint.pth --decoder-type LLMD --eval

# eval tsne viz
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1036 \
#     eval_dsl_SignIR.py --batch-size 4 \
#     --output_dir out/GF_vlp_eval --config ./configs/config_gf_vlp.yaml \
#     --resume out/GF_vlp_v2_finetune/best_checkpoint.pth --decoder-type LD \
#     --tsne_visualize

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1036 \
#     eval_dsl_SignIR.py --batch-size 4 \
#     --output_dir out/SignCL_eval --config ./configs/config_gloss_free.yaml \
#     --resume out/GF_SignCL/best_checkpoint.pth --decoder-type LD \
#     --tsne_visualize

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1036 \
#     eval_dsl_SignIR.py --batch-size 4 \
#     --output_dir out/SignIR_CLIP_eval_b32 --config ./configs/config_signIR_finetune.yaml \
#     --resume out/SignIR_CLIP_finetune_b32/best_checkpoint.pth --decoder-type LD \
#     --tsne_visualize

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1036 \
#     eval_dsl_SignIR.py --batch-size 4 \
#     --output_dir out/SignIR_CLIP_eval_b16 --config ./configs/config_signIR_finetune.yaml \
#     --resume out/SignIR_CLIP_finetune/best_checkpoint.pth --decoder-type LD \
#     --tsne_visualize

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1036 \
#     eval_dsl_SignIR.py --batch-size 4 \
#     --output_dir out/SignIR_avgpool_eval --config ./configs/config_signIR_finetune.yaml \
#     --resume out/SignIR_avgpool_finetune/best_checkpoint.pth --decoder-type LD \
#     --tsne_visualize

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1036 \
#     eval_dsl_SignIR.py --batch-size 4 \
#     --output_dir out/SignIR_noSignCL_eval --config ./configs/config_signIR_finetune.yaml \
#     --resume out/SignIR_noSignCL_finetune/best_checkpoint.pth --decoder-type LD \
#     --tsne_visualize

# baseline
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1036 \
#     eval_dsl_SignIR.py --batch-size 4 \
#     --output_dir out/GF_vlp_eval --config ./configs/config_gf_vlp.yaml \
#     --resume out/GF_vlp_v2_finetune/best_checkpoint.pth --decoder-type LD \
#     --tsne_visualize --ouput_feature

# text embedding viz
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1036 \
#     eval_dsl_SignCL_text_embedding.py --batch-size 2 \
#     --output_dir out/GF_SignCL_text_viz --config ./configs/config_signIR_finetune.yaml \
#     --resume out/0630_GF_SignCL/best_checkpoint.pth --decoder-type LD --tsne_visualize

# Contrastive Learning Training 
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1036 \
#     train_vlp_SignIR.py --batch-size 16 --epochs 80 --opt sgd --lr 0.0065\
#     --output_dir out/SignIR_avgpool --config ./configs/config_signIR.yaml \
#     --decoder-type LD --training-refurbish True --noise-rate 0.15 \ --noise-type omit_last --random-shuffle False --decoder-type LLMD

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1036 \
#     train_vlp_SignIR.py --batch-size 16 --epochs 80 --opt sgd --lr 0.0065\
#     --output_dir out/SignIR_avgpool --config ./configs/config_signIR.yaml \
#     --decoder-type LD --training-refurbish True --noise-rate 0.15 \ --noise-type omit_last --random-shuffle False --decoder-type LLMD

# Constrastive Learning Finetuning
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=1037 \
    train_dsl_SignIR.py --batch-size 8 --epochs 100 --opt sgd --lr 0.0065\
    --decoder-type LD --finetune out/GF_vlp_v2/best_checkpoint.pth \
    --output_dir out/GF_vlp_v2_finetune_b16 --config ./configs/config_signIR_finetune.yaml

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=1036 \
#     train_vlp_SignIR.py --batch-size 16 --epochs 80 --opt sgd --lr 0.0065\
#     --output_dir out/GF_vlp_v2 --config ./configs/config_signIR.yaml \
#     --ignore_signcl --loss_fn KL