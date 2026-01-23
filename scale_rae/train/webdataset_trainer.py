#!/usr/bin/env python3

import json
import os
import tarfile
import tempfile
import shutil
import logging
from io import BytesIO
from torch.utils.data import IterableDataset
import random
from concurrent.futures import ThreadPoolExecutor
import math
from PIL import Image, ImageFile
import torch
import torch.utils.data
import copy
import numpy as np

# Scale-RAE imports
from scale_rae.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX
from scale_rae.train.spmd_trainer import preprocess_multimodal, preprocess
from scale_rae.mm_utils import select_best_resolution, resize_and_pad_image, divide_to_patches



import transformers

# Enable PIL to gracefully handle truncated images (avoids stalls on partial files)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Optional fast JPEG decoder (TurboJPEG). If unavailable, we fall back to PIL.
try:
    from turbojpeg import TurboJPEG, TJPF_RGB  # type: ignore
    _TURBOJPEG = TurboJPEG()
except Exception:
    _TURBOJPEG = None

# Monkey patch IterableDatasetShard to add __len__ method
from accelerate.data_loader import IterableDatasetShard

def _iterabledatasetshard_len(self):
    """Add __len__ method to IterableDatasetShard by delegating to wrapped dataset"""
    if hasattr(self.dataset, '__len__'):
        # Rough estimate: divide by number of processes
        total_len = len(self.dataset)
        return total_len // self.num_processes
    else:
        raise TypeError("Wrapped dataset has no __len__ method")

# Monkey patch the __len__ method
IterableDatasetShard.__len__ = _iterabledatasetshard_len


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    print("dataset is", dataset)
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False
    except AttributeError:
        # Ray DataSets raises an AttributeError: https://github.com/ray-project/ray/blob/master/python/ray/data/dataset.py#L5616
        return False

transformers.trainer_utils.has_length = has_length

# Use the same logger as the main trainer  
logger = logging.getLogger(__name__)

# Optional WebDataset support (used only for shard splitting if available)
try:
    import webdataset as wds
    _WDS_AVAILABLE = True
except Exception:
    _WDS_AVAILABLE = False

class WebDatasetLazySupervisedDataset(IterableDataset):
    """WebDataset version of LazySupervisedDataset for efficient tar-based data loading"""
    
    def __init__(self, data_path: str, tokenizer, data_args, model_configs=None):
        super(WebDatasetLazySupervisedDataset, self).__init__()

        logger.info(f"WebDatasetLazySupervisedDataset initialized with data_path: {data_path}")
        
        # Load manifest
        with open(data_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Mark that this dataset performs its own sharding (by node/worker or manual)
        self.already_sharded = True
        
        # Initialize same fields as LazySupervisedDataset  
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.data_args = data_args
        self.model_configs = model_configs
        self.tar_urls = self.manifest['tars']
        self.samples_per_tar_assumption = int(self.manifest.get('samples_per_tar_assumption', 10000))
        
        # TPU/distributed info
        import torch_xla.core.xla_model as xm
        self.rank = xm.get_ordinal()
        
        # Resume state management
        self.epoch = 0
        self.epoch_tar_order = list(self.tar_urls)  # Will be shuffled per epoch
        self.resume_state = {}  # Per-worker resume tracking
        self.resume_checkpoint_dir = None
        
        logger.info(f"WebDataset initialized: {len(self.tar_urls)} tars, ~{self.manifest.get('estimated_total_samples', 0):,} samples")
        
    def __len__(self):
        """Return a per-rank estimate: (assigned_tars_for_this_rank_across_workers × samples_per_tar_assumption)."""
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank, world_size = 0, 1
        except Exception:
            rank, world_size = 0, 1

        # Use node-trimmed tar indices if available; otherwise, fallback to global
        if hasattr(self, "_rank_tar_indices") and isinstance(self._rank_tar_indices, list):
            assigned_tar_count = len(self._rank_tar_indices)
        else:
            # If set_epoch hasn't run yet, simulate trimming to avoid gross overestimation
            try:
                ranks_per_node = int(os.getenv("RANKS_PER_NODE", str(world_size)))
                if ranks_per_node <= 0: ranks_per_node = 8
            except Exception:
                ranks_per_node = 8
            num_nodes = max(1, world_size // max(1, ranks_per_node))
            num_tars = len(self.epoch_tar_order)
            # Per-node trimmed count
            node_trimmed = (num_tars // num_nodes) if num_nodes > 0 else num_tars
            # Per-rank within node (distribute trimmed list across ranks_per_node, difference <=1)
            assigned_tar_count = node_trimmed // max(1, ranks_per_node)

        estimated_samples = assigned_tar_count * self.samples_per_tar_assumption
        if estimated_samples <= 0:
            # Fallback to global estimate if something goes wrong
            estimated_samples = int(self.manifest.get('estimated_total_samples', num_tars * self.samples_per_tar_assumption))

        return estimated_samples

        
        
    def set_epoch(self, epoch):
        """Called by HF Trainer - shuffle tars deterministically per epoch"""
        self.epoch = epoch
        
        # Deterministic shuffle based on epoch
        rng = random.Random(42 + epoch)
        self.epoch_tar_order = list(self.tar_urls) 
        rng.shuffle(self.epoch_tar_order)
        
        logger.info(f"Epoch {epoch}: Shuffled {len(self.epoch_tar_order)} tars")

        # Node-level equalization: discard extra shards so each node has same count
        # Determine rank/world and ranks_per_node
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank, world_size = 0, 1
        except Exception:
            rank, world_size = 0, 1

        try:
            ranks_per_node = int(os.getenv("RANKS_PER_NODE", str(world_size)))
            if ranks_per_node <= 0:
                ranks_per_node = world_size
        except Exception:
            ranks_per_node = world_size

        num_nodes = max(1, world_size // max(1, ranks_per_node))
        node_id = 0 if ranks_per_node == 0 else (rank // max(1, ranks_per_node))

        # Build per-node buckets by round-robin over nodes
        buckets = [[] for _ in range(num_nodes)]
        for idx in range(len(self.epoch_tar_order)):
            buckets[idx % num_nodes].append(idx)

        # Compute common count and trim this node's list
        if num_nodes > 1:
            min_len = min(len(b) for b in buckets)
        else:
            min_len = len(buckets[0])

        self._node_tar_indices = buckets[node_id][:min_len]
        logger.info(f"Node {node_id}/{num_nodes}: assigned {len(buckets[node_id])} tars, trimmed to {len(self._node_tar_indices)}")

        # Further split trimmed node indices across ranks within the node (intra-node rank)
        intra_node_rank = rank % max(1, ranks_per_node)
        self._rank_tar_indices = [idx for i, idx in enumerate(self._node_tar_indices) if (i % max(1, ranks_per_node)) == intra_node_rank]
        logger.info(f"Rank {rank} (intra-node {intra_node_rank}/{max(1, ranks_per_node)}): {len(self._rank_tar_indices)} tars after intra-node split")
         
    def set_resume_checkpoint(self, checkpoint_dir):
        """Set checkpoint directory for resume functionality (legacy method)"""
        self.resume_checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            self._load_resume_state()
    
    def set_resume_checkpoint_file(self, checkpoint_file_path):
        """Set specific checkpoint file path for per-rank resume (more efficient)"""
        try:
            # Handle both GCS and local paths
            resume_path_local = checkpoint_file_path.replace("gs://", "/mnt/")
            if os.path.exists(resume_path_local):
                with open(resume_path_local, 'r') as f:
                    loaded = json.load(f)
                # Support both new-format {epoch, tar_shuffle_seed, workers} and legacy direct-workers dict
                if isinstance(loaded, dict) and isinstance(loaded.get("workers"), dict):
                    self.resume_state = loaded["workers"]
                    # Optional: store for observability
                    self._loaded_epoch = loaded.get("epoch")
                    self._loaded_tar_shuffle_seed = loaded.get("tar_shuffle_seed")
                else:
                    self.resume_state = loaded
                logger.info(f"Loaded WebDataset resume state from {resume_path_local}")
            else:
                logger.info(f"No WebDataset resume state found at {resume_path_local}, starting fresh")
                self.resume_state = {}

                
        except Exception as e:
            logger.warning(f"Failed to load WebDataset resume state from {checkpoint_file_path}: {e}")
            self.resume_state = {}
            
    def _load_resume_state(self):
        """Load WebDataset resume state from checkpoint directory (legacy method)"""
        if not self.resume_checkpoint_dir:
            return
            
        resume_path = os.path.join(self.resume_checkpoint_dir, "webdataset_state.json")
        try:
            # Handle both GCS and local paths
            resume_path_local = resume_path.replace("gs://", "/mnt/")
            if os.path.exists(resume_path_local):
                with open(resume_path_local, 'r') as f:
                    loaded = json.load(f)
                # Support both new-format {epoch, tar_shuffle_seed, workers} and legacy direct-workers dict
                if isinstance(loaded, dict) and isinstance(loaded.get("workers"), dict):
                    self.resume_state = loaded["workers"]
                    # Optional: store for observability
                    self._loaded_epoch = loaded.get("epoch")
                    self._loaded_tar_shuffle_seed = loaded.get("tar_shuffle_seed")
                else:
                    self.resume_state = loaded
                logger.info(f"Loaded WebDataset resume state from {resume_path_local}")
            else:
                logger.info("No WebDataset resume state found, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load WebDataset resume state: {e}")
            self.resume_state = {}
            
    def get_resume_state(self):
        """Get current resume state for checkpoint saving"""
        return {
            "epoch": self.epoch,
            "tar_shuffle_seed": 42 + self.epoch,
            "workers": self.resume_state
        }
        
    def _get_worker_info(self):
        """Get worker distribution info"""
        # TPU distributed info
        import torch.distributed as dist
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
            
        # DataLoader worker info
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
            
        return rank, world_size, worker_id, num_workers
        
    def _get_worker_tars(self, rank, world_size, worker_id, num_workers):
        """Distribute tars evenly across all workers using round-robin"""
        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id
        
        # Round-robin assignment: worker N gets tars [N, N+total, N+2*total, ...]
        worker_tar_indices = []
        for tar_idx in range(len(self.epoch_tar_order)):
            if tar_idx % total_workers == global_worker_id:
                worker_tar_indices.append(tar_idx)
                
        return worker_tar_indices
        
    def __iter__(self):
        """Main iteration logic with worker distribution and resume"""
        rank, world_size, worker_id, num_workers = self._get_worker_info()
        worker_key = f"rank_{rank}_worker_{worker_id}"

        # Prepare resume state entry
        if worker_key not in self.resume_state:
            self.resume_state[worker_key] = {
                "completed_tar_indices": [],
                "current_tar_idx": None,
                "samples_processed_in_current_tar": 0,
            }
        worker_state = self.resume_state[worker_key]
        completed_tars = set(worker_state["completed_tar_indices"])
        # Record identity for optional flushing to shared storage
        worker_state["worker_key"] = worker_key
        worker_state["rank"] = rank
        worker_state["worker_id"] = worker_id

        # Prefer WebDataset shard splitting if available
        if _WDS_AVAILABLE:
            logger.info(f"[WDS] Using WebDataset shard splitting for rank={rank} worker={worker_id}")
            try:
                shardlist = wds.SimpleShardList(self.epoch_tar_order)
                # Split shards by node/process, then by worker
                shardlist = shardlist.split_by_node()
                shardlist = shardlist.split_by_worker()

                # Truncate per-node shards to common min length across nodes
                # Compute min_len = floor(num_tars / num_nodes)
                try:
                    ranks_per_node = int(os.getenv("RANKS_PER_NODE", str(world_size)))
                    if ranks_per_node <= 0:
                        ranks_per_node = world_size
                except Exception:
                    ranks_per_node = world_size
                num_nodes = max(1, world_size // max(1, ranks_per_node))
                min_len = len(self.epoch_tar_order) // num_nodes if num_nodes > 0 else len(self.epoch_tar_order)

                # Iterate assigned shards with truncation
                taken = 0
                for tar_url in shardlist:
                    if taken >= min_len:
                        break
                    try:
                        tar_idx = self.epoch_tar_order.index(tar_url)
                    except ValueError:
                        # Should not happen; skip if tar_url not found
                        continue
 
                    if tar_idx in completed_tars:
                        continue  # Skip completed tars
 
                    worker_state["current_tar_idx"] = tar_idx
                    for sample in self._process_tar(tar_url, worker_state):
                        yield sample
 
                    # Mark completion for this tar
                    worker_state["completed_tar_indices"].append(tar_idx)
                    worker_state["current_tar_idx"] = None
                    worker_state["samples_processed_in_current_tar"] = 0
                    # Optional: flush progress to shared dir
                    self._flush_worker_state_by_key(worker_key)
                    taken += 1
            except Exception as e:
                logger.warning(f"[WDS] Fallback to manual sharding due to: {e}")
                # Fall through to manual distribution below
        else:
            logger.info("[WDS] WebDataset not available; using manual round-robin sharding")

        # Manual round-robin sharding (fallback path)
        if not _WDS_AVAILABLE:
            # Use node-trimmed list; if missing, fall back to global indices
            if hasattr(self, "_node_tar_indices") and isinstance(self._node_tar_indices, list):
                node_tar_indices = self._node_tar_indices
            else:
                # Build buckets and trim on-the-fly
                try:
                    ranks_per_node = int(os.getenv("RANKS_PER_NODE", str(world_size)))
                    if ranks_per_node <= 0: ranks_per_node = 8
                except Exception:
                    ranks_per_node = 8
                num_nodes = max(1, world_size // max(1, ranks_per_node))
                node_id = rank // max(1, ranks_per_node) if ranks_per_node > 0 else 0
                buckets = [[] for _ in range(num_nodes)]
                for idx in range(len(self.epoch_tar_order)):
                    buckets[idx % num_nodes].append(idx)
                min_len = min(len(b) for b in buckets) if num_nodes > 1 else len(buckets[0])
                node_tar_indices = buckets[node_id][:min_len]

            # Split within node among ranks (intra-node), then among this process's dataloader workers
            intra_node_rank = rank % max(1, ranks_per_node)
            rank_tar_indices = [idx for i, idx in enumerate(node_tar_indices) if (i % max(1, ranks_per_node)) == intra_node_rank]

            worker_tar_indices = []
            for i, tar_idx in enumerate(rank_tar_indices):
                if (i % max(1, num_workers)) == worker_id:
                    worker_tar_indices.append(tar_idx)

            if not worker_tar_indices:
                logger.info(f"Worker {worker_key} has no tars assigned")
                return iter([])

            logger.info(f"Worker {worker_key} processing {len(worker_tar_indices)} tars")

            for tar_idx in worker_tar_indices:
                if tar_idx in completed_tars:
                    logger.info(f"Worker {worker_key} skipping completed tar {tar_idx}")
                    continue  # Skip completed tars

                tar_url = self.epoch_tar_order[tar_idx]
                worker_state["current_tar_idx"] = tar_idx
                try:
                    for sample in self._process_tar(tar_url, worker_state):
                        yield sample

                    # Tar completed
                    worker_state["completed_tar_indices"].append(tar_idx)
                    worker_state["current_tar_idx"] = None
                    worker_state["samples_processed_in_current_tar"] = 0
                    # Optional: flush progress to shared dir
                    self._flush_worker_state_by_key(worker_key)
                except Exception as e:
                    logger.error(f"Error processing tar {tar_url}: {e}")
                    continue
                
    def _process_tar(self, tar_url, worker_state):
        """Process a single tar file and yield samples"""
        samples_to_skip = worker_state.get("samples_processed_in_current_tar", 0)
        samples_processed = 0
        worker_key = worker_state.get("worker_key")
        
        # Download tar to local temp file
        local_tar_path = None
        try:
            local_tar_path = self._download_tar(tar_url)
            
            with tarfile.open(local_tar_path, 'r') as tar:
                # Get all JSON files (our samples)
                json_members = [m for m in tar.getmembers() if m.name.endswith('.json')]
                json_members.sort(key=lambda x: x.name)  # Deterministic order
                
                for json_member in json_members:
                    if samples_processed < samples_to_skip:
                        samples_processed += 1
                        logger.info(f"Worker {worker_key} skipping already processed sample {json_member.name}")
                        continue  # Skip already processed samples
                        
                    try:
                        # Extract JSON data
                        json_file = tar.extractfile(json_member)
                        sample_data = json.loads(json_file.read().decode('utf-8'))
                        
                        # Process sample (reuse LazySupervisedDataset logic)  
                        processed_sample = self._process_sample(sample_data, tar)
                        if processed_sample is not None:
                            yield processed_sample
                            
                        samples_processed += 1
                        
                        # Update resume state periodically
                        if samples_processed % 100 == 0:
                            worker_state["samples_processed_in_current_tar"] = samples_processed
                            # Optional: flush to shared dir if configured
                            if worker_key is not None:
                                self._flush_worker_state_by_key(worker_key)
                            
                    except Exception as e:
                        logger.warning(f"Error processing sample {json_member.name}: {e}")
                        samples_processed += 1
                        continue  # Skip bad samples
                        
        finally:
            # No cleanup needed - reading directly from mounted location
            pass
                
    def _download_tar(self, tar_url):
        """Return tar path directly - no copying needed for mounted GCS"""
        # Since tars are mounted from GCS, read directly
        if os.path.exists(tar_url):
            return tar_url
        else:
            raise FileNotFoundError(f"Tar file not found: {tar_url}")

    def _get_image_from_tar(self, sample_data, tar_handle):
        """Extract image from tar file with optional TurboJPEG fast path."""
        if 'image_filename' not in sample_data:
            return None  # No image in this sample

        img_filename = sample_data['image_filename']

        try:
            # Locate file inside tar
            img_member = tar_handle.getmember(img_filename)
            img_file = tar_handle.extractfile(img_member)
            if img_file is None:
                logger.warning(f"Image {img_filename} could not be extracted (None)")
                return None
            img_data = img_file.read()

            # Fast-path for JPEG using TurboJPEG if available
            if _TURBOJPEG is not None and len(img_data) >= 3 and img_data[:3] == b'\xff\xd8\xff':
                try:
                    arr = _TURBOJPEG.decode(img_data, pixel_format=TJPF_RGB)
                    # arr is HxWx3 uint8
                    return Image.fromarray(arr, mode='RGB')
                except Exception as te:
                    logger.debug(f"TurboJPEG failed for {img_filename}: {te}; falling back to PIL")

            # Fallback: Pillow decode
            try:
                return Image.open(BytesIO(img_data)).convert('RGB')
            except Exception:
                # One more attempt: allow loading without convert to catch edge cases
                try:
                    return Image.open(BytesIO(img_data))
                except Exception as e2:
                    logger.warning(f"Error loading image {img_filename}: {e2}")
                    return None

        except KeyError:
            logger.warning(f"Image {img_filename} not found in tar")
            return None
        except Exception as e:
            logger.warning(f"Error loading image {img_filename}: {e}")
            return None
            
    def _process_sample(self, sample_data, tar_handle):
        """Process a single sample - implement your own logic here"""
        
        # Get image from tar
        image = self._get_image_from_tar(sample_data, tar_handle)

        dat = sample_data
        sources = [dat]

        has_image = image is not None
        has_video = self._has_video(dat)

        assert not (has_image and has_video), "Image and video should not appear in a single data sample"

        # Add image token if not present in the conversation
        if has_image or has_video:
            for source in sources:
                if DEFAULT_IMAGE_TOKEN not in json.dumps(source['conversations']):
                    source['conversations'][0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + source['conversations'][0]['value']
        
        images = []
        vision_token_len = self.data_args.vision_tower_aux_token_len_list[0]
        processor_gen = self.data_args.image_processor_gen

        if has_image:
            images = [image]
            
            processor_aux_list = self.data_args.image_processor_aux_list
            max_length = self.model_configs.tokenizer_model_max_length

            # Check image count limit
            if len(images) > (max_length // vision_token_len) - 1:
                logger.warning("Exceeded max length, skipping sample")
                return None
            
            image_size = images[0].size

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            if self.data_args.image_aspect_ratio not in ['pad', 'anyres', 'square']:
                raise NotImplementedError("Only pad and anyres are supported for now.")

            if self.data_args.image_aspect_ratio == 'pad':
                image_aux_list = []
                for processor_aux in processor_aux_list:
                    image_aux = image
                    target_resolution = processor_aux.crop_size['height']
                    image_aux = expand2square(image_aux, tuple(int(x*255) for x in processor_aux.image_mean)).resize((target_resolution, target_resolution))
                    image_aux = processor_aux.preprocess(image_aux, return_tensors='pt')['pixel_values'][0]
                    image_aux_list.append(image_aux)

            


            elif self.data_args.image_aspect_ratio == 'square':
                image_aux_list = []
                for image in images:
                    image_aux = processor_aux_list[0].preprocess(image, return_tensors='pt')['pixel_values'][0]
                    image_aux_list.append(image_aux)

                # Add image_gen_list for generation alignment (VAE training)
                image_gen_list = []
                if processor_gen is not None:
                    for image in images:
                        gen_target_resolution_height, gen_target_resolution_width = processor_gen.crop_size['height'], processor_gen.crop_size['width']
                        image_gen = processor_gen.preprocess(image, height=gen_target_resolution_height, width=gen_target_resolution_width)[0]
                        image_gen_list.append(image_gen)
            
            elif self.data_args.image_aspect_ratio == 'anyres':

                image_aux_list = []
                for processor_aux in processor_aux_list:

                    image_aux = image
                    target_resolution = processor_aux.crop_size['height']

                    # Choose resolutions where number of subimages <= anyres_max_subimages
                    possible_resolutions = [
                        (int(width * target_resolution), int(height * target_resolution))
                        for width in range(1, self.data_args.anyres_max_subimages + 1)
                        for height in range(1, self.data_args.anyres_max_subimages + 1)
                        if (width * height) <= self.data_args.anyres_max_subimages
                    ]

                    best_resolution = select_best_resolution(image.size, possible_resolutions)
                    # Use zero padding for anyres images
                    image_aux_padded = resize_and_pad_image(image, best_resolution)

                    patches = divide_to_patches(image_aux_padded, target_resolution)

                    # image_aux = expand2square(image_aux, tuple(int(x*255) for x in processor_aux.image_mean)).resize((target_resolution, target_resolution))
                    # we should use expand2square and then resize but we choose to use the following code to make sure our codebase aligns well with llava-onevision
                    image_aux = image_aux.resize((target_resolution, target_resolution))

                    image_patches = [image_aux] + patches
                    image_patches = [processor_aux.preprocess(patch, return_tensors='pt')['pixel_values'][0] for patch in image_patches]
                    image_aux_list.append(torch.stack(image_patches))

            else:
                raise NotImplementedError("Only pad, anyres and square are supported for now.")

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)

        elif has_video:
            video_file = dat['video']
            video_folder = self.data_args.video_folder
            video_file = os.path.join(video_folder, video_file)
            
            try:
                if os.path.isdir(video_file):
                    
                    if "shareVideoGPTV" in video_file: # shareVideoGPTV use 2FPS
                        avg_fps = 2
                    elif "TVQA" in video_file: # TVQA use 3FPS
                        avg_fps = 3
                    else: # for unknown video frames, we assume it is 1FPS
                        avg_fps = 1

                    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                    frame_files.sort()  # Ensure the frames are sorted if they are named sequentially
                    
                    video_time = len(frame_files) / avg_fps

                    if 'start' in dat:
                        start_time = float(dat['start'])
                        end_time = float(dat['end'])
                        start_frame = int(start_time * avg_fps)
                        end_frame = int(end_time * avg_fps)
                        end_frame = min(len(frame_files) - 1, end_frame)
                        frame_files = frame_files[start_frame:end_frame+1] # from start to end
                        video_time = end_time - start_time

                    frame_idx = [i for i in range(0, len(frame_files), avg_fps)]
                    frame_time = [i/avg_fps for i in frame_idx]

                    if self.data_args.video_max_frames > 0:
                        if len(frame_files) > self.data_args.video_max_frames or self.data_args.video_force_sample:
                            frame_idx = np.linspace(0, len(frame_files) - 1, self.data_args.video_max_frames, dtype=int).tolist()
                            frame_time = [i/avg_fps for i in frame_idx]


                    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

                    # Read and store the sampled frames
                    num_frames_to_sample = len(frame_idx)
                    video = []
                    for idx in frame_idx:
                        frame_path = frame_files[idx]
                        try:
                            with Image.open(frame_path) as img:
                                frame = img.convert("RGB")
                                video.append(np.array(frame))
                        except IOError:
                            debug_print(f"Failed to read frame at path: {frame_path}")
                    video = np.stack(video)
                elif video_file.endswith(".gif"):
                    if not os.path.exists(video_file):
                        debug_print("File {} not exist!".format(video_file))
                        raise FileNotFoundError
                    assert "start" not in dat and "end" not in dat, "start and end should not be in gif video"
                    assert "start_frame" not in dat and "end_frame" not in dat, "start_frame and end_frame should not be in gif video"
                    video, video_time, frame_time, num_frames_to_sample = process_gif_with_imageio(video_file, self.data_args)
                else:
                    if not os.path.exists(video_file):
                        debug_print("File {} not exist!".format(video_file))
                        raise FileNotFoundError

                    if 'start_frame' in dat:
                        start_frame = dat['start_frame']
                        end_frame = dat['end_frame']
                        current_observation_frame = dat.get('current_observation_frame', None)

                        video, video_time, frame_time, num_frames_to_sample = process_video_with_decord_byframe(video_file, self.data_args, start_frame, end_frame, current_observation_frame)
                        if not video.size > 0:
                            raise ValueError(f"Video {video_file} is empty")
                    elif 'start' in dat:
                        start_time = dat['start']
                        end_time = dat['end']
                        video, video_time, frame_time, num_frames_to_sample = process_video_with_decord_bytime(video_file, self.data_args, start_time, end_time)
                    else:
                        video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)
            except BaseException as error:
                logger.warning(f"Error occurs when load video from {video_file}: {error}")
                return None

            video_h, video_w = video.shape[1:3]
            image_size = (video_w, video_h)

            processor_aux_list = self.data_args.image_processor_aux_list
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            if self.data_args.image_aspect_ratio not in ['pad', 'anyres', 'square']:
                raise NotImplementedError("Only pad and anyres are supported for now.")

            # Video always use pad
            image_aux_list = []
            if self.data_args.image_aspect_ratio in ['anyres', 'pad']:
                for processor_aux in processor_aux_list:
                    target_resolution = processor_aux.crop_size['height']
                    frames = [expand2square(Image.fromarray(video[_], mode="RGB"), tuple(int(x*255) for x in processor_aux.image_mean)) for _ in range(video.shape[0])]
                    processed_frames = processor_aux.preprocess(frames, return_tensors='pt')['pixel_values']
                    image_aux_list.append(processed_frames)
            elif self.data_args.image_aspect_ratio in ['square']:
                for processor_aux in processor_aux_list:
                    target_resolution = processor_aux.crop_size['height']
                    frames = [expand2square(Image.fromarray(video[_], mode="RGB"), tuple(int(x*255) for x in processor_aux.image_mean)) for _ in range(video.shape[0])]
                    processed_frames = processor_aux.preprocess(frames, return_tensors='pt')['pixel_values']
                    image_aux_list.append(processed_frames)
            else:
                raise NotImplementedError("Only pad and anyres are supported for now.")
            
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=has_image or has_video)
        
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                        labels=data_dict["labels"][0])
        
        if has_image and not self.check_image_tokens(data_dict, images, vision_token_len):
            return None

        if (data_dict['labels']!=IGNORE_INDEX).sum()==0:
            logger.warning("All tokens are masked, skipping sample")
            return None


        assert self.data_args.si_token_len >= 0
        assert self.data_args.miv_token_len >= 0
        
        si_token_len = self.data_args.si_token_len
        si_token_hws = int(math.sqrt(si_token_len))
        si_token_len_w_newline = si_token_len + si_token_hws

        miv_token_len = self.data_args.miv_token_len
        miv_token_hws = int(math.sqrt(miv_token_len))
        miv_token_len_w_newline = miv_token_len + miv_token_hws


        # Get token dimensions - simple square case, just use vision_token_len directly
        
        tokens_per_image = vision_token_len


        processor_aux_list = self.data_args.image_processor_aux_list
        processor_aux = processor_aux_list[0]

        # Simple square mode processing for interleaved images - NO newlines
        if has_image and self.data_args.image_aspect_ratio == 'square':
            
            # Process all images in the data
            n_imgs = len(image_aux_list)
            
            # Pad image tensors to fixed shape for TPU
            image_aux_list_padded = []
            image_aux_padded = torch.zeros(self.data_args.max_images_per_sample,  3, processor_aux.crop_size['height'], processor_aux.crop_size['width'])
            
            for i, image in enumerate(image_aux_list):
                image_aux_padded[i] = image
                # image_aux_list_padded.append(image_aux_padded)
            
            data_dict['image_aux_list'] = [image_aux_padded]
            
            # Handle image_gen padding only if processor_gen exists
            if processor_gen is not None:
                image_gen_padded = torch.zeros(self.data_args.max_images_per_sample,  3, processor_gen.crop_size['height'], processor_gen.crop_size['width'])
                for i, image in enumerate(image_gen_list):
                    image_gen_padded[i] = image
                data_dict['image_gen_list'] = [image_gen_padded]
            else:
                data_dict['image_gen_list'] = []
            
            # No newlines in simple square mode
            
            # Find all image token positions in the input
            input_ids = data_dict['input_ids']
            labels = data_dict['labels']
            img_token_positions = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            
            # Calculate total tokens needed for all images
            max_imgs = min(len(img_token_positions), self.data_args.max_images_per_sample)
            img_tokens_total = max_imgs * tokens_per_image
            
            # tokens_per_image is already vision_token_len
            T       = self.data_args.max_images_per_sample * tokens_per_image   # fixed length
            used    = max_imgs * tokens_per_image                               # real tokens
            PAD_VAL = T + 1                                                     # sentinel

            # 1. start fully padded
            vision_token_indices = torch.full((T,), PAD_VAL, dtype=torch.long)

            # 2. overwrite the real part (0 … used‑1) with raster indices
            if used:
                vision_token_indices[:used] = torch.arange(used, dtype=torch.long)

            # 3. permutation vector (same length, guaranteed)
            data_dict["vision_token_indices"] = vision_token_indices.sort()[1]


            
            # Reconstruct the input_ids and labels with multiple image positions
            new_input_ids = []
            new_labels = []
            
            # Handle interleaved images by replacing each image token
            last_pos = 0
            for i, pos in enumerate(img_token_positions[:max_imgs]):
                # Add text before this image
                new_input_ids.append(input_ids[last_pos:pos])
                new_labels.append(labels[last_pos:pos])
                
                # Simple fixed token count per image
                img_tokens = torch.full((tokens_per_image,), IMAGE_TOKEN_INDEX, 
                                    dtype=input_ids.dtype, device=input_ids.device)
                img_labels = torch.full((tokens_per_image,), IGNORE_INDEX, 
                                    dtype=labels.dtype, device=labels.device)
                
                new_input_ids.append(img_tokens)
                new_labels.append(img_labels)
                
                # Update position tracker
                last_pos = pos + 1
            
            # Add any remaining text after the last image
            if last_pos < len(input_ids):
                new_input_ids.append(input_ids[last_pos:])
                new_labels.append(labels[last_pos:])
            
            # Concatenate all parts
            data_dict['input_ids'] = torch.cat(new_input_ids)
            data_dict['labels'] = torch.cat(new_labels)
            
            # Create pseudo image tokens for padding to fixed length
            pseudo_img_tokens = torch.zeros(((self.data_args.max_images_per_sample - n_imgs)*tokens_per_image,), 
                                        dtype=input_ids.dtype, device=input_ids.device) + IMAGE_TOKEN_INDEX

            data_dict['pseudo_img_tokens'] = pseudo_img_tokens
            
            # Truncate input_ids and labels if needed to fit model_max_length
            max_length = self.model_configs.tokenizer_model_max_length
            if len(data_dict['input_ids']) > max_length:
                data_dict['input_ids'] = data_dict['input_ids'][:max_length]
                data_dict['labels'] = data_dict['labels'][:max_length]
            
            # Save image size
            data_dict['image_size'] = image_size
            



        # image exist in the data
        elif has_image:
            n_imgs = image_aux_list[0].size(0)

            image_aux_list_padded = []
            for image_aux in image_aux_list:
                assert image_aux.shape[0] == n_imgs
                image_aux_padded = torch.zeros((self.data_args.max_images_per_sample, *image_aux.size()[1:]))
                image_aux_padded[:n_imgs] = image_aux
                image_aux_list_padded.append(image_aux_padded)

            data_dict['image_aux_list'] = image_aux_list_padded

            num_img_patches = (best_resolution[0] // target_resolution, best_resolution[1] // target_resolution)

            # Calculate the unpadded feature shape and output feature shape
            image_w, image_h = image_size
            original_aspect_ratio = image_w / image_h
            padded_feature_w, padded_feature_h = (best_resolution[0] // target_resolution * si_token_hws, best_resolution[1] // target_resolution * si_token_hws)
            padded_feautre_aspect_ratio = padded_feature_w / padded_feature_h

            # Determine padding size and direction
            if original_aspect_ratio > padded_feautre_aspect_ratio:
                # Padding was added to the height
                scale_factor = padded_feature_w / image_w
                unpadded_feature_w = padded_feature_w
                unpadded_feature_h = int(image_h * scale_factor)
                padding_feature_w = 0
                padding_feature_h = (padded_feature_h - unpadded_feature_h) // 2
            else:
                # Padding was added to the width
                scale_factor = padded_feature_h / image_h
                unpadded_feature_h = padded_feature_h
                unpadded_feature_w = int(image_w * scale_factor)
                padding_feature_h = 0
                padding_feature_w = (padded_feature_w - unpadded_feature_w) // 2

            output_feature_shape = (unpadded_feature_h, unpadded_feature_w)

            # Create image token indexing
            num_img_tokens_total = self.data_args.max_images_per_sample * (miv_token_len_w_newline + si_token_len_w_newline)

            # Multi-image and video token indices
            miv_token_indices = torch.zeros(self.data_args.max_images_per_sample * miv_token_len_w_newline) + num_img_tokens_total + 1

            # Snapshot image token indices (remove newline for snapshot)
            snapshot_token_indices = torch.linspace(0, si_token_len_w_newline - 1, si_token_len_w_newline).reshape(si_token_hws, si_token_hws + 1).long() + miv_token_indices.numel()
            snapshot_token_indices[:, -1] = num_img_tokens_total + 1

            # Anyres image token masks
            anyres_token_masks = torch.zeros(num_img_patches[1] * si_token_hws, num_img_patches[0] * si_token_hws).bool()
            if padding_feature_h > 0:
                anyres_token_masks[:padding_feature_h, :] = True
                anyres_token_masks[padding_feature_h+output_feature_shape[0]:, :] = True
            if padding_feature_w > 0:
                anyres_token_masks[:, :padding_feature_w] = True
                anyres_token_masks[:, padding_feature_w+output_feature_shape[1]:] = True
            # Reshape anyres masks: (nh, h, nw, w) -> (nh, nw, h, w) -> (n, h, w)
            anyres_token_masks = anyres_token_masks.reshape(num_img_patches[1], si_token_hws, num_img_patches[0], si_token_hws).permute(0, 2, 1, 3).reshape(n_imgs - 1, si_token_hws, si_token_hws)

            anyres_token_masks_w_newline = torch.zeros(num_img_patches[1] * num_img_patches[0], si_token_hws,  si_token_hws + 1).bool()
            # Pad anyres images with newline token and mask all newline tokens            
            for index in range(n_imgs - 1):
                if (index + 1) % num_img_patches[0] == 0: # if is end of line, then unmask it
                    ...
                else:
                    anyres_token_masks_w_newline[index, :, -1] = True # mask
            anyres_token_masks_w_newline[:, :, :-1] = anyres_token_masks

            anyres_token_masks_w_newline = anyres_token_masks_w_newline.reshape(num_img_patches[1], num_img_patches[0], si_token_hws, si_token_hws + 1).permute(0, 2, 1, 3).reshape(num_img_patches[1] * si_token_hws, num_img_patches[0] * (si_token_hws + 1)) # (nh * nw, h, w + 1) -> (nh, nw, h, w + 1) -> (nh * h, nw * (w + 1))
            if padding_feature_h > 0:
                anyres_token_masks_w_newline[:padding_feature_h, :] = True
                anyres_token_masks_w_newline[padding_feature_h+output_feature_shape[0]:, :] = True
            anyres_token_masks_w_newline = anyres_token_masks_w_newline.reshape(num_img_patches[1], si_token_hws, num_img_patches[0], (si_token_hws + 1)).permute(0, 2, 1, 3).reshape(num_img_patches[1] * num_img_patches[0], si_token_hws, (si_token_hws + 1))
            anyres_token_masks_w_newline = anyres_token_masks_w_newline.reshape(-1)

            anyres_token_indices = torch.linspace(0, (n_imgs - 1) * si_token_len_w_newline - 1, (n_imgs - 1) * si_token_len_w_newline).long() + miv_token_indices.numel() + snapshot_token_indices.numel()
            anyres_token_indices = anyres_token_indices.reshape(num_img_patches[1], si_token_hws, num_img_patches[0], (si_token_hws + 1)).permute(0, 2, 1, 3).reshape(num_img_patches[1] * num_img_patches[0], si_token_hws, (si_token_hws + 1)).flatten()
            anyres_token_indices = torch.where(anyres_token_masks_w_newline, num_img_tokens_total + 1, anyres_token_indices)

            padding_token_indices = torch.zeros((self.data_args.max_images_per_sample - n_imgs, si_token_hws, si_token_hws + 1)).long() + num_img_tokens_total + 1

            vision_token_indices = torch.cat([miv_token_indices.contiguous().view(-1), snapshot_token_indices.contiguous().view(-1), anyres_token_indices.contiguous().view(-1), padding_token_indices.contiguous().view(-1)], dim=0)

            num_real_img_tokens = (vision_token_indices != num_img_tokens_total + 1).sum()
            data_dict['vision_token_indices'] = vision_token_indices.view(-1).sort()[1]

            # rebuild the input_ids
            input_ids = data_dict['input_ids']
            labels = data_dict['labels']

            img_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]
            pre_img_tokens, post_img_tokens = input_ids[:img_token_indices[0]], input_ids[img_token_indices[0]+1:]
            real_img_tokens = torch.zeros((num_real_img_tokens,)).long() + IMAGE_TOKEN_INDEX
            pseudo_img_tokens = torch.zeros((self.model_configs.tokenizer_model_max_length - num_real_img_tokens,)).long() + IMAGE_TOKEN_INDEX

            data_dict['input_ids'] = torch.cat([pre_img_tokens, real_img_tokens, post_img_tokens])
            data_dict['pseudo_img_tokens'] = pseudo_img_tokens

            pre_img_labels, post_img_labels = labels[:img_token_indices[0]], labels[img_token_indices[0]+1:]
            real_img_labels = torch.zeros((num_real_img_tokens,)).long() + IGNORE_INDEX
            data_dict['labels'] = torch.cat([pre_img_labels, real_img_labels, post_img_labels])

        elif has_video:
            n_imgs = image_aux_list[0].size(0)

            image_aux_list_padded = []
            for image_aux in image_aux_list:
                assert image_aux.shape[0] == n_imgs
                image_aux_padded = torch.zeros((self.data_args.video_max_frames, *image_aux.size()[1:]))
                image_aux_padded[:n_imgs] = image_aux
                image_aux_list_padded.append(image_aux_padded)

            data_dict['image_aux_list'] = image_aux_list_padded

            assert [_.size(0) == self.data_args.video_max_frames for _ in image_aux_list]

            num_img_tokens_total = self.data_args.video_max_frames * miv_token_len
            vision_token_indices = torch.linspace(0, num_img_tokens_total - 1, num_img_tokens_total).long()

            vision_token_indices[n_imgs * miv_token_len:] = num_img_tokens_total + 1
            data_dict['vision_token_indices'] = vision_token_indices.view(-1).sort()[1]
            num_real_img_tokens = (vision_token_indices != num_img_tokens_total + 1).sum()

            # rebuild the input_ids
            input_ids = data_dict['input_ids']
            labels = data_dict['labels']
            
            img_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]

            assert img_token_indices.numel() == 1, "Only one image token should be there"

            pre_img_tokens, post_img_tokens = input_ids[:img_token_indices[0]], input_ids[img_token_indices[0]+1:]
            real_img_tokens = torch.zeros((num_real_img_tokens,)).long() + IMAGE_TOKEN_INDEX
            
            data_dict['input_ids'] = torch.cat([pre_img_tokens, real_img_tokens, post_img_tokens])
            pseudo_img_tokens = torch.zeros((self.model_configs.tokenizer_model_max_length - num_real_img_tokens,)).long() + IMAGE_TOKEN_INDEX
            data_dict['pseudo_img_tokens'] = pseudo_img_tokens

            pre_img_labels, post_img_labels = labels[:img_token_indices[0]], labels[img_token_indices[0]+1:]
            real_img_labels = torch.zeros((num_real_img_tokens,)).long() + IGNORE_INDEX
            data_dict['labels'] = torch.cat([pre_img_labels, real_img_labels, post_img_labels])

        elif self.data_args.is_multimodal:

            # image does not exist in the data, but the model is multimodal
            crop_size = 336
            processor_aux_list = self.data_args.image_processor_aux_list
            
            image_aux_list = []

            for processor_aux in processor_aux_list:
                if self.data_args.max_images_per_sample > 0:
                    image_aux = torch.zeros(self.data_args.max_images_per_sample, 3, processor_aux.crop_size['height'], processor_aux.crop_size['width'])
                else:
                    raise NotImplementedError

                image_aux_list.append(image_aux)

            # same constants
            T       = self.data_args.max_images_per_sample * tokens_per_image
            PAD_VAL = T + 1
            # everything is padding because used = 0
            vision_token_indices = torch.full((T,), PAD_VAL, dtype=torch.long)

            data_dict["vision_token_indices"] = vision_token_indices.sort()[1]

            data_dict['pseudo_img_tokens'] = torch.zeros(self.data_args.max_images_per_sample * tokens_per_image).long() + IMAGE_TOKEN_INDEX

            image_size = (crop_size, crop_size)
            data_dict['image_aux_list'] = image_aux_list

            # Add image_gen_list
            image_gen_list = []
            if processor_gen is not None:
                if self.data_args.max_images_per_sample > 0:
                    image_gen = torch.zeros(self.data_args.max_images_per_sample, 3, processor_gen.crop_size['height'], processor_gen.crop_size['width'])
                else:
                    raise NotImplementedError
                image_gen_list.append(image_gen)
            data_dict['image_gen_list'] = image_gen_list

        data_dict['image_size'] = image_size
        data_dict['has_video'] = has_video
        data_dict['has_image'] = has_image

        return data_dict
            
    def check_image_tokens(self, data_dict, images, vision_token_len):
        """Check image token validation - extracted from LazySupervisedDataset"""
        input_ids = data_dict["input_ids"]
        max_length = self.model_configs.tokenizer_model_max_length
        num_images = len(images)
        placeholder = IMAGE_TOKEN_INDEX
        
        # Placeholder count must match number of images
        ph_count = (input_ids == placeholder).sum().item()
        if ph_count != num_images:
            return False
            
        # Full expansion must fit
        expanded_len = input_ids.numel() + (vision_token_len - 1) * num_images
        if expanded_len > max_length:
            return False
            
        return True
            
    def _has_image(self, sample: dict) -> bool:
        """Reuse exact logic from LazySupervisedDataset"""
        return "image" in sample and not str(sample['image']) in ['', 'None', 'none', 'nan']
    
    def _has_video(self, sample: dict) -> bool:
        """Reuse exact logic from LazySupervisedDataset"""
        return "video" in sample and not str(sample['video']) in ['', 'None', 'none', 'nan'] 

    def _flush_worker_state_by_key(self, worker_key: str) -> None:
        """Optionally flush a single worker's resume state to a shared directory.
        Controlled by env WEBDS_STATE_DIR. Backward-compatible (no-op if unset)."""
        try:
            state_dir = os.getenv("WEBDS_STATE_DIR", "")
            if not state_dir:
                return
            ws = self.resume_state.get(worker_key)
            if not isinstance(ws, dict):
                return
            rank = ws.get("rank")
            worker_id = ws.get("worker_id")
            if rank is None or worker_id is None:
                return
            os.makedirs(state_dir, exist_ok=True)
            filepath = os.path.join(state_dir, f"webds_rank{rank}_worker{worker_id}.json")
            payload = {
                "epoch": self.epoch,
                "tar_shuffle_seed": 42 + self.epoch,
                "workers": {worker_key: ws},
            }
            with open(filepath, "w") as f:
                json.dump(payload, f)
        except Exception as e:
            logger.warning(f"Failed to flush worker state for {worker_key}: {e}") 