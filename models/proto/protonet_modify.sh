# #!/usr/bin/env bash
# # Activate environment
# source .venv/bin/activate

# # Source the .envrc file, if it exists
# if [[ -f ".envrc" ]]; then
#     source .envrc
# fi

# 上面是我把他先註解掉的，因為我用的是 conda

# Simple check on the gpu we will be using
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "hostname: $(hostname)"
command -v nvidia-smi >/dev/null && {
    echo "GPU Devices:"
    nvidia-smi
} || {
    :
}

PYTHONPATH=. python models/proto/protonet_modify.py $@
