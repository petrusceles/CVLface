#!/bin/sh

# Ensure the Python script is executable and accessible
# You can specify the full path to the Python script if needed

# Install dependencies
echo "Installing dependencies"
pip install -r requirements.txt
apt install unzip

# Download dataset
echo "Download datasets"
cd /root/workspace/CVLface/cvlface/data_root
python download.py
unzip -P password -qq "facerec_val.zip" -d .


# Evaluate
for i in $(seq 0 17)
do
    echo "Downloading model with index: $i"
    cd /root/workspace/CVLface/cvlface/pretrained_models/recognition/adaface_ir50dsc_webface4m
    python download.py $i
    # Check the exit status of the Python script
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download model with index $i"
    else
        echo "Model with index $i downloaded successfully"
    fi
    echo "Evaluate model with index: $i"
    cd /root/workspace/CVLface/cvlface/research/recognition/code/adaface_ir50dsc_webface4m
    python eval.py --num_gpu 1 --eval_config_name full --ckpt_dir ../../../../pretrained_models/recognition/adaface_ir50dsc_webface4m
done