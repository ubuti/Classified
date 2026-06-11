#!/usr/bin/env zsh
# unite the training sets in ONE directory called 'train'

target=/Users/inuit/.cache/kagglehub/datasets/ambityga/imagenet100/train/

if [ -d $target ]; then
    # directory exists — check if empty
    if [ -n "$(ls -a $target 2>/dev/null)" ]; then
        echo "Error: '$target' exists and is not empty." >&2
        exit 1
    else
        echo "Directory '$target' exists and is empty."
    fi
else
    mkdir -p $target
fi

filepath=/Users/inuit/.cache/kagglehub/datasets/ambityga/imagenet100/versions/8/
cd $filepath

for i in {1,2,3,4}; do
    label=$filepath"train.X$i"
    cd $label
    cp -r * $target
    cd ..
done


