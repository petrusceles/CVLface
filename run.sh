#!/bin/sh

# Ensure the Python script is executable and accessible
# You can specify the full path to the Python script if needed

# Install dependencies
echo "Installing dependencies"
pip install -r requirements.txt
apt install unzip

# # Download dataset
echo "Download datasets"
cd /root/workspace/CVLface/cvlface/data_root
python download.py
unzip -P password -qq "facerec_val.zip" -d .

# Download model
for i in $(seq 0 3)
do
    echo "Downloading model with index: $i"
    cd /root/workspace/CVLface/cvlface/pretrained_models/recognition/adaface_ir50dsc_webface4m/collections
    python download.py $i
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download model with index $i"
    else
        echo "Model with index $i downloaded successfully"
    fi
done


# Evaluate
for i in $(seq 0 3)
do
    cp "/root/workspace/CVLface/cvlface/pretrained_models/recognition/adaface_ir50dsc_webface4m/collections/model_$i.pt" "/root/workspace/CVLface/cvlface/pretrained_models/recognition/adaface_ir50dsc_webface4m"

    mv "/root/workspace/CVLface/cvlface/pretrained_models/recognition/adaface_ir50dsc_webface4m/model_$i.pt" "/root/workspace/CVLface/cvlface/pretrained_models/recognition/adaface_ir50dsc_webface4m/model.pt"
    # Check the exit status of the Python script
    echo "Evaluate model with index: $i"
    cd /root/workspace/CVLface/cvlface/research/recognition/code/adaface_ir50dsc_webface4m
    python eval.py --num_gpu 1 --eval_config_name base --ckpt_dir ../../../../pretrained_models/recognition/adaface_ir50dsc_webface4m
done