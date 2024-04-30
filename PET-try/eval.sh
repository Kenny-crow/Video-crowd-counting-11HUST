CUDA_VISIBLE_DEVICES='0' \
python eval.py \
    --dataset_file="MyData" \
    --resume="resume/SHA_model.pth" \
    --vis_dir="results"