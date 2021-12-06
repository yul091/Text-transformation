# # WordCNN - MR
# python main.py \
# --do_eval --model_path 'checkpoints/WordCNN-2021-10-12 22:57:14.291700-256.pth' \
# --dataset MR WordCNN --mode rand --epochs 256 

# # WordCNN - ag_news
# python main.py \
# --do_eval --model_path "/glusterfs/data/yxl190090/deep-text-classification-pytorch/checkpoints/WordCNN-2021-10-12 23:42:50.430681-226.pth" \
# --dataset ag_news WordCNN --mode rand --epochs 5 

# # WordCNN - amazon_review
# CUDA_VISIBLE_DEVICES=4 python main.py \
# --do_eval --model_path "/glusterfs/data/yxl190090/deep-text-classification-pytorch/checkpoints/WordCNN-2021-11-01 22:34:43.111120-254.pth" \
# --dataset amazon_review_full WordCNN --mode rand --epochs 5

# yelp_full model path 
CUDA_VISIBLE_DEVICES=3 python main.py \
--do_eval --model_path "/glusterfs/data/yxl190090/deep-text-classification-pytorch/checkpoints/WordCNN-2021-11-01 22:26:33.765244-214.pth" \
--dataset yelp_review_full WordCNN --mode rand --epochs 5 

# # sogou model path
# CUDA_VISIBLE_DEVICES=2 python main.py \
# --do_eval --model_path "/glusterfs/data/yxl190090/deep-text-classification-pytorch/checkpoints/WordCNN-2021-11-01 22:34:43.111120-58.pth" \
# --dataset sogou_news WordCNN --mode rand --epochs 5 

# amazon model path
# "/glusterfs/data/yxl190090/deep-text-classification-pytorch/checkpoints/WordCNN-2021-11-01 22:34:43.111120-254.pth"