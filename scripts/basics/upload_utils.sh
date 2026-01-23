#!/bin/bash

upload_checkpoints() {
    local CKPT_PATH=checkpoints/$CKPT_NAME
    if [ ! -d "$CKPT_PATH" ]; then
        echo "Checkpoint path does not exist. Exiting..."
        exit 1
    fi

    echo "Syncing checkpoints to GCS..."
    gcloud alpha storage rsync $CKPT_PATH gs://$GCS_BUCKET_NAME/$GCS_CKPT_DIR/$CKPT_NAME/checkpoint-last
    gcloud alpha storage rsync $CKPT_PATH gs://$GCS_BUCKET_NAME/$GCS_CKPT_DIR/$CKPT_NAME-last
    echo "Syncing finished. Checkpoints available at gs://$GCS_BUCKET_NAME/$GCS_CKPT_DIR/$CKPT_NAME-last"
}