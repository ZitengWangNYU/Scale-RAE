import argparse
import json
import os
import random
import shutil
import numpy as np


def parse_metadata(metadata: dict, metric: str) -> dict:
    
    target_sample_num = 4
    total_sample_num = metadata["n_samples_generated"]

    all_text_responses = metadata["text_responses"]
    
    # Collect all responses with their metrics
    all_samples = []
    for index, text_response in all_text_responses.items():
        score = text_response["score"]
        yes_logits = text_response["yes_logit"]
        no_logits = text_response["no_logit"]
        yes_conf = text_response["yes_conf_score"]
        no_conf = text_response["no_conf_score"]
        prompt_loss = text_response.get("prompt_loss", 0.0)
        prompt_conf = text_response.get("prompt_conf_score", 0.0)
        image_conf = text_response.get("image_conf_score", 0.0)
        response_conf = text_response.get("response_conf_score", 0.0)
        all_samples.append(
            (int(index), score, yes_logits, no_logits, yes_conf, no_conf, prompt_loss, prompt_conf, image_conf, response_conf)
        )
    
    if metric == "loss":
        # based on caption loss
        all_samples.sort(key=lambda x: (x[4], -x[1], -x[2], x[3]))
    elif metric == "answer_logits":
        # based on single token logits for yes / no
        all_samples.sort(key=lambda x: (-x[1], -x[2], x[3]))
    elif metric == "answer_conf":
        #  based on single token confidence for yes / no
        all_samples.sort(key=lambda x: (-x[1], -x[4], x[5]))
    elif metric == "prompt_conf":
        # based on average confidence for the generation prompt
        all_samples.sort(key=lambda x: (-x[-3], -x[2], x[3]))
    elif metric == "image_conf":
        # based on average confidence for the image tokens
        all_samples.sort(key=lambda x: (-x[-2], -x[2], x[3]))
    elif metric == "response_conf":
        # based on average confidence for the generation response
        all_samples.sort(key=lambda x: (-x[-1], -x[2], x[3]))
    else:
        raise ValueError(f"Invalid metric: {metric}")
    
    # Select top target_sample_num samples
    selected_samples = all_samples[:target_sample_num]
    selected_indices = [idx for idx, *_ in selected_samples]
    
    new_metadata = {k: v for k, v in metadata.items() if k != "text_responses"}
    new_metadata["text_responses"] = {
        k: v for k, v in metadata["text_responses"].items() if int(k) in selected_indices
    }
    return selected_indices, all_samples, new_metadata

def verify_samples(sample_dir: str, metric: str) -> None:
    
    out_dir = sample_dir + f"_{metric}_verified"
    ref_dir = sample_dir + "_ref"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)

    total_indices_count = 0
    total_matched_indices_count = 0
    total_correct_indices_count = 0
    for prompt_dir in sorted(os.listdir(sample_dir)):
        prompt_path = os.path.join(sample_dir, prompt_dir)
        metadata_path = os.path.join(prompt_path, "metadata.jsonl")

        out_prompt_path = os.path.join(out_dir, prompt_dir)
        out_metadata_path = os.path.join(out_prompt_path, "metadata.jsonl")

        ref_prompt_path = os.path.join(ref_dir, prompt_dir)
        ref_metadata_path = os.path.join(ref_prompt_path, "metadata.jsonl")

        os.makedirs(out_prompt_path, exist_ok=True)
        os.makedirs(os.path.join(out_prompt_path, "samples"), exist_ok=True)

        os.makedirs(ref_prompt_path, exist_ok=True)
        os.makedirs(os.path.join(ref_prompt_path, "samples"), exist_ok=True)

        if not os.path.exists(metadata_path):
            print(f"WARNING: Skipping {prompt_dir}: metadata.jsonl not found (generation may have failed)")
            # Create empty metadata so GenEval doesn't crash
            dummy_metadata = {"prompt": f"prompt_{prompt_dir}", "n_samples_generated": 0}
            with open(out_metadata_path, "w") as f:
                json.dump(dummy_metadata, f)
            with open(ref_metadata_path, "w") as f:
                json.dump(dummy_metadata, f)
            continue
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            print(f"WARNING: Corrupted metadata at {prompt_dir}, skipping")
            dummy_metadata = {"prompt": f"prompt_{prompt_dir}", "n_samples_generated": 0}
            with open(out_metadata_path, "w") as f:
                json.dump(dummy_metadata, f)
            with open(ref_metadata_path, "w") as f:
                json.dump(dummy_metadata, f)
            continue

        selected_indices, all_samples, new_metadata = parse_metadata(metadata, metric)
        if len(selected_indices) != 4:
            print(f"WARNING: Not enough samples at {prompt_dir} (only {len(selected_indices)}), copying what we have")
            # Still write metadata so GenEval doesn't crash
            with open(out_metadata_path, "w") as f:
                json.dump(new_metadata, f)
            with open(ref_metadata_path, "w") as f:
                json.dump(new_metadata, f)
            # Copy whatever images exist
            for idx in selected_indices:
                src_path = os.path.join(prompt_path, f"samples/{idx:05}.png")
                if os.path.exists(src_path):
                    dst_path = os.path.join(out_prompt_path, f"samples/{idx:05}.png")
                    ref_path = os.path.join(ref_prompt_path, f"samples/{idx:05}.png")
                    shutil.copy(src_path, dst_path)
                    shutil.copy(src_path, ref_path)
            continue


        matched_indices_count = (np.array(selected_indices) == np.array(new_metadata['top_ir_indices'])).sum()
        total_matched_indices_count += matched_indices_count
        total_indices_count += len(selected_indices)


        if len(new_metadata.get('top_ir_indices', [])) != 4:
            print(f"WARNING: Not enough reference samples at {prompt_dir}")
            # Still write metadata
            with open(out_metadata_path, "w") as f:
                json.dump(new_metadata, f)
            with open(ref_metadata_path, "w") as f:
                json.dump(new_metadata, f)
            continue

        correct_indices_count = len([i for i in selected_indices if i in new_metadata['top_ir_indices']])
        total_correct_indices_count += correct_indices_count

        with open(out_metadata_path, "w") as f:
            json.dump(new_metadata, f)
        with open(ref_metadata_path, "w") as f:
            json.dump(new_metadata, f)

        copied_indices = 0

        for index, ref_index in zip(selected_indices, new_metadata['top_ir_indices']):
            src_image_path = os.path.join(prompt_path, f"samples/{(index):05}.png")
            dst_image_path = os.path.join(out_prompt_path, f"samples/{copied_indices:05}.png")

            ref_src_image_path = os.path.join(prompt_path, f"samples/{(ref_index):05}.png")
            ref_image_path = os.path.join(ref_prompt_path, f"samples/{copied_indices:05}.png")
            shutil.copy(src_image_path, dst_image_path)
            shutil.copy(ref_src_image_path, ref_image_path)
            copied_indices += 1
        


        print(f"\nPrompt: {prompt_dir}")
        print(f"  Total samples: {len(all_samples)}")
        print(f"  Selected indices: {selected_indices}")
        print(f"  Selected count: {len(selected_indices)}")
        print(f"  Reference indices: {new_metadata['top_ir_indices']}")
        print(f"  Correct indices: {correct_indices_count}")

    print(f"Matched ratio: {total_matched_indices_count / total_indices_count}")
    print(f"Correct ratio: {total_correct_indices_count / total_indices_count}")
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True, choices=["loss", "prompt_conf", "image_conf", "response_conf", "answer_logits", "answer_conf"])
    args = parser.parse_args()

    verify_samples(args.sample_dir, args.metric)