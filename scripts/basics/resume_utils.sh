#!/bin/bash

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
        --resume)
            resume="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
        esac
    done
}

check_training_completion() {
    if gsutil -q stat gs://$GS_BUCKET_NAME/$GS_CKPT_DIR/$CKPT_NAME-last/**; then
        echo "Training has already finished. Exiting script..."
        exit 0
    fi
}

find_latest_checkpoint() {
    local checkpoints=$(gcloud alpha storage ls gs://$GCS_BUCKET_NAME/$GCS_CKPT_DIR/$CKPT_NAME/ |
        grep -E '/[0-9]+/$' |
        sed 's/.*\/\([0-9]*\)\//\1/' |
        sort -V)

    if [ -n "$checkpoints" ]; then
        latest_checkpoint=$(echo "$checkpoints" | tail -n 1)
        echo "Found checkpoint: $latest_checkpoint"

        # local_ckpt_path=~/$GS_CKPT_DIR/$CKPT_NAME/$latest_checkpoint/
        # mkdir -p $local_ckpt_path
        # gcloud alpha storage cp gs://$GCS_BUCKET_NAME/$GCS_CKPT_DIR/$CKPT_NAME/$latest_checkpoint/trainer_state.json $local_ckpt_path

        # resume=$local_ckpt_path
        resume=/mnt/$GCS_BUCKET_NAME/$GCS_CKPT_DIR/$CKPT_NAME/$latest_checkpoint/

        echo "Resuming training from checkpoint: $resume"
    else
        echo "No checkpoint found. Starting training from scratch."
    fi
}