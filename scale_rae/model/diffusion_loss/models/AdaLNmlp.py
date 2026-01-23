from torch import nn
import math
import torch
from .model_utils import GaussianFourierEmbedding, TimestepEmbedder, ConditionEmbedder
from .model_utils import VisionRotaryEmbeddingFast, SwiGLUFFN, RMSNorm
from .lightningDiT import PatchEmbed

from scale_rae.utils import IS_XLA_AVAILABLE
import os
if IS_XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm # <-- Import XLA model
    import torch_xla.distributed.spmd as xs



import logging # Using logging for better control over debug messages

# Configure logging (optional, but good practice)
# Set level to INFO to see summary, DEBUG to see details per layer
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)
# --- XLA Master Logging Helper ---
# Optional: Define a helper function to avoid repeating the check
def log_master(level, msg, *args, **kwargs):
    """Logs a message only on the master TPU ordinal."""
    # xm.is_master_ordinal() checks if the current process is rank 0
    if xm.is_master_ordinal():
        log.log(level, msg, *args, **kwargs)
# ---


class CustomRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = (self.weight * hidden_states).to(input_dtype)
        return output

def modulate(x, shift, scale):
    return x * (1 + scale) + shift
    # return x

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        # Debug Note, changed to RMSNorm
        # self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.in_ln = CustomRMSNorm(channels, eps=1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            # nn.Identity(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h

class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        # Debug Note, changed to RMSNorm
        # self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.norm_final = CustomRMSNorm(model_channels, eps=1e-6)

        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class TokenProjector(nn.Module):
    """
    ViT‑style single‑token embedder.
    (B, C, 1, 1)  →  (B, 1, hidden_size)
    
    Design   :  flatten → Linear(C → hidden) → optional LayerNorm
    Inspired :  timm.models.vision_transformer.PatchEmbed
    """
    def __init__(self, in_channels: int, hidden_size: int,
                 use_norm: bool = True, bias: bool = True):
        super().__init__()
        # 1) channel projection (Conv1×1 in PatchEmbed -> Linear here)
        self.proj = nn.Linear(in_channels, hidden_size, bias=bias)

        # 2) optional normalization (PatchEmbed has norm layer)
        self.norm = (CustomRMSNorm(hidden_size, eps=1e-6)
                 if use_norm else nn.Identity())

    def forward(self, x):
        # x: (B, C, 1, 1)
        x_dtype = x.dtype
        x = x.flatten(2).transpose(1, 2)   # → (B, 1, C)
        x = self.proj(x)                   # → (B, 1, hidden_size)
        x = self.norm(x).to(x_dtype)  # → (B, 1, hidden_size)   # LayerNorm across hidden dim
        return x

        
class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        input_size: int ,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
        depth: int,
        class_dropout_prob: float,
        z_channels: int ,
        use_gembed=True,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = hidden_size
        self.depth = depth
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(hidden_size) if not use_gembed else GaussianFourierEmbedding(hidden_size)
        # self.y_embedder = ConditionEmbedder(z_channels, dropout_prob=class_dropout_prob)
        self.y_proj = nn.Linear(z_channels, hidden_size, bias=True)
        # self.input_proj = nn.Linear(in_channels, model_channels)
        # self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias = True)
        self.x_embedder = TokenProjector(
            in_channels=in_channels,
            hidden_size=hidden_size,
            use_norm=True,      # keep norm for stability
            bias=True
        )
        res_blocks = []
        for i in range(depth):
            res_blocks.append(ResBlock(
                hidden_size,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(hidden_size, in_channels)
        if IS_XLA_AVAILABLE:
            self.initialize_weights()

        # init_ok = self.check_initialization()
    
        # if not init_ok:
        #     print("\n!!! WARNING: INITIALIZATION VERIFICATION FAILED !!!")
        #     print("This will likely cause NaN/Inf issues during forward passes")

        #Add it here, similar to the one in forward

    def check_initialization(self):
        """Verify initialization was performed correctly before any forward passes."""
        print("\n--- INITIALIZATION VERIFICATION CHECK ---")
        
        # Track overall success
        xavier_init_ok = True
        zero_init_ok = True
        
        # 1. Check the Xavier Initialization on regular layers
        print("\n1. CHECKING XAVIER INITIALIZATION ON REGULAR LAYERS:")
        
        # Define acceptable limits for Xavier
        XAVIER_MEAN_THRESHOLD = 0.1  # Should be close to 0
        XAVIER_MAX_THRESHOLD = 1.0   # Should typically be < 1.0
        
        # Check main MLP layers in ResBlocks
        for i, block in enumerate(self.res_blocks):
            for j, layer in enumerate(block.mlp):
                if isinstance(layer, nn.Linear):
                    # Check weight
                    w_max = layer.weight.abs().max().item()
                    w_mean = layer.weight.abs().mean().item()
                    w_std = layer.weight.std().item()
                    
                    # Ideal Xavier has standard deviation of 1/sqrt(fan_in)
                    fan_in = layer.weight.size(1)
                    ideal_std = 1.0 / math.sqrt(fan_in)
                    
                    w_ok = (w_max < XAVIER_MAX_THRESHOLD) and (w_mean < XAVIER_MEAN_THRESHOLD)
                    
                    print(f"ResBlock[{i}].mlp[{j}] Linear weight stats:")
                    print(f"  max: {w_max:.6f}, mean: {w_mean:.6f}, std: {w_std:.6f}")
                    print(f"  Xavier ideal std: {ideal_std:.6f}")
                    print(f"  XAVIER INIT OK: {'✓' if w_ok else '✗'}")
                    
                    if not w_ok:
                        xavier_init_ok = False
                        print(f"  !!! ABNORMAL VALUES DETECTED - Expected max < {XAVIER_MAX_THRESHOLD}, mean < {XAVIER_MEAN_THRESHOLD}")
                    
                    # Check for extreme values
                    extreme_values = (layer.weight.abs() > 5.0).sum().item()
                    if extreme_values > 0:
                        xavier_init_ok = False
                        print(f"  !!! FOUND {extreme_values} EXTREME VALUES (>5.0) !!!")
        
        # 2. Check the Zero Initialization on specific layers
        print("\n2. CHECKING ZERO INITIALIZATION ON SPECIFIC LAYERS:")
        zero_threshold = 1e-6  # Values below this are considered zero
        
        # Check ResBlock modulation layers
        for i, block in enumerate(self.res_blocks):
            try:
                if isinstance(block.adaLN_modulation, nn.Sequential) and len(block.adaLN_modulation) > 0:
                    # Check final linear layer in adaLN_modulation
                    final_layer = block.adaLN_modulation[-1]
                    if isinstance(final_layer, nn.Linear):
                        w_sum = final_layer.weight.abs().sum().item()
                        w_zero_ok = w_sum < zero_threshold
                        print(f"ResBlock[{i}].adaLN_modulation[-1].weight abs sum: {w_sum:.8f} - ZERO INIT: {'✓' if w_zero_ok else '✗'}")
                        
                        if not w_zero_ok:
                            zero_init_ok = False
                            # Sample some values to see what they actually are
                            sample_vals = final_layer.weight.flatten()[:5].tolist()
                            print(f"  Sample values: {sample_vals}")
                        
                        if final_layer.bias is not None:
                            b_sum = final_layer.bias.abs().sum().item()
                            b_zero_ok = b_sum < zero_threshold
                            print(f"ResBlock[{i}].adaLN_modulation[-1].bias abs sum: {b_sum:.8f} - ZERO INIT: {'✓' if b_zero_ok else '✗'}")
                            
                            if not b_zero_ok:
                                zero_init_ok = False
            except Exception as e:
                print(f"Error checking ResBlock[{i}]: {e}")
        
        # Check FinalLayer zero init
        try:
            # Check adaLN modulation final layer
            final_ada_layer = self.final_layer.adaLN_modulation[-1]
            if isinstance(final_ada_layer, nn.Linear):
                w_sum = final_ada_layer.weight.abs().sum().item()
                w_zero_ok = w_sum < zero_threshold
                print(f"FinalLayer.adaLN_modulation[-1].weight abs sum: {w_sum:.8f} - ZERO INIT: {'✓' if w_zero_ok else '✗'}")
                
                if not w_zero_ok:
                    zero_init_ok = False
                
                if final_ada_layer.bias is not None:
                    b_sum = final_ada_layer.bias.abs().sum().item()
                    b_zero_ok = b_sum < zero_threshold
                    print(f"FinalLayer.adaLN_modulation[-1].bias abs sum: {b_sum:.8f} - ZERO INIT: {'✓' if b_zero_ok else '✗'}")
                    
                    if not b_zero_ok:
                        zero_init_ok = False
        except Exception as e:
            print(f"Error checking FinalLayer adaLN: {e}")
        
        # Check final linear layer
        try:
            w_sum = self.final_layer.linear.weight.abs().sum().item()
            w_zero_ok = w_sum < zero_threshold
            print(f"FinalLayer.linear.weight abs sum: {w_sum:.8f} - ZERO INIT: {'✓' if w_zero_ok else '✗'}")
            
            if not w_zero_ok:
                zero_init_ok = False
            
            if self.final_layer.linear.bias is not None:
                b_sum = self.final_layer.linear.bias.abs().sum().item()
                b_zero_ok = b_sum < zero_threshold
                print(f"FinalLayer.linear.bias abs sum: {b_sum:.8f} - ZERO INIT: {'✓' if b_zero_ok else '✗'}")
                
                if not b_zero_ok:
                    zero_init_ok = False
        except Exception as e:
            print(f"Error checking FinalLayer linear: {e}")
        
        # 3. Check for NaN/Inf in any parameter
        print("\n3. CHECKING FOR NaN/Inf IN ALL PARAMETERS:")
        nan_or_inf_found = False
        for name, param in self.named_parameters():
            if param is None:
                continue
            
            if torch.isnan(param).any() or torch.isinf(param).any():
                nan_or_inf_found = True
                print(f"!!! NaN or Inf found in {name}")
        
        if not nan_or_inf_found:
            print("✓ No NaN or Inf values found in any parameters")
        
        # 4. Overall summary
        print("\n--- INITIALIZATION VERIFICATION SUMMARY ---")
        print(f"Xavier Init OK: {'✓' if xavier_init_ok else '✗'}")
        print(f"Zero Init OK: {'✓' if zero_init_ok else '✗'}")
        print(f"No NaN/Inf: {'✓' if not nan_or_inf_found else '✗'}")
        
        if not (xavier_init_ok and zero_init_ok and not nan_or_inf_found):
            print("!!! INITIALIZATION PROBLEMS DETECTED !!!")
        else:
            print("✓ All initialization checks passed")
        
        print("--- END VERIFICATION CHECK ---\n")
        
        # Return overall success status
        return xavier_init_ok and zero_init_ok and not nan_or_inf_found


# <<< --- ADDED DEBUG FUNCTION (v2 - Clearer Output) --- >>>
    @torch.no_grad()
    def _print_weight(self, identifier="", tolerance=1e-6):
        """Prints the abs sum of specific weights expected to be zero with clear OK/NOT OK status."""
        log_lines = [f"--- WEIGHT CHECK ({identifier}) ---"]
        try:
            # --- Final Layer ---
            final_linear_w = self.final_layer.linear.weight
            final_linear_w_sum = final_linear_w.abs().sum().item()
            final_linear_w_ok = abs(final_linear_w_sum) < tolerance
            log_lines.append(
                f"  FinalLinear W Sum : {final_linear_w_sum:.4e} (Should be 0.0) -> {'OK' if final_linear_w_ok else 'NOT OK'}"
            )

            final_linear_b_sum = 0.0
            final_linear_b_ok = True # OK if bias doesn't exist
            if self.final_layer.linear.bias is not None:
                final_linear_b = self.final_layer.linear.bias
                final_linear_b_sum = final_linear_b.abs().sum().item()
                final_linear_b_ok = abs(final_linear_b_sum) < tolerance
            log_lines.append(
                f"  FinalLinear B Sum : {final_linear_b_sum:.4e} (Should be 0.0) -> {'OK' if final_linear_b_ok else 'NOT OK'}"
            )

            # --- ResBlock[0] ---
            res0_adaLN_w_sum = float('nan')
            res0_adaLN_b_sum = float('nan')
            res0_adaLN_w_ok = False
            res0_adaLN_b_ok = False
            resblock_status_msg = ""

            if self.res_blocks and len(self.res_blocks) > 0:
                 res0_module = self.res_blocks[0].adaLN_modulation
                 if isinstance(res0_module, nn.Sequential) and len(res0_module) > 0:
                      last_layer = res0_module[-1]
                      if isinstance(last_layer, nn.Linear):
                           res0_adaLN_w = last_layer.weight
                           res0_adaLN_w_sum = res0_adaLN_w.abs().sum().item()
                           res0_adaLN_w_ok = abs(res0_adaLN_w_sum) < tolerance

                           res0_adaLN_b_ok = True # OK if bias doesn't exist
                           if last_layer.bias is not None:
                                res0_adaLN_b = last_layer.bias
                                res0_adaLN_b_sum = res0_adaLN_b.abs().sum().item()
                                res0_adaLN_b_ok = abs(res0_adaLN_b_sum) < tolerance
                           else:
                                res0_adaLN_b_sum = 0.0 # Explicitly set to 0 if no bias
                      else:
                           resblock_status_msg = " | ResBlock[0] AdaLN: Last layer not Linear."
                 else:
                      resblock_status_msg = " | ResBlock[0] AdaLN: Structure issue."
            else:
                 resblock_status_msg = " | ResBlocks: Empty or missing."

            # Only add ResBlock lines if successfully accessed
            if not resblock_status_msg:
                 log_lines.append(
                    f"  ResBlock[0] AdaLN W Sum: {res0_adaLN_w_sum:.4e} (Should be 0.0) -> {'OK' if res0_adaLN_w_ok else 'NOT OK'}"
                 )
                 log_lines.append(
                    f"  ResBlock[0] AdaLN B Sum: {res0_adaLN_b_sum:.4e} (Should be 0.0) -> {'OK' if res0_adaLN_b_ok else 'NOT OK'}"
                 )
            else:
                 # Add the status message if there was an issue accessing ResBlock[0] details
                 log_lines[0] += resblock_status_msg # Append status to the header line

            # --- Log Combined Message ---
            log_master(logging.INFO, "\n".join(log_lines)) # Log all lines together

        except AttributeError as e:
            log_master(logging.ERROR, f"--- WEIGHT CHECK ({identifier}) --- Error accessing weights: {e}")
    # <<< --- END ADDED DEBUG FUNCTION --- >>>

    def wrong_initialize_weights(self):
        # --- Basic Initialization ---
        # First, apply Xavier Uniform to weights and 0 to biases for all linear layers
        @torch.no_grad()
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    # Using fill_ here as well for consistency, though constant_ is likely fine too
                    module.bias.data.fill_(0.0)
        self.apply(_basic_init)

        xm.mark_step()


        # --- Time Embedding MLP Initialization ---
        # Initialize specific layers of timestep embedding MLP with Normal distribution
        # Add checks for existence and type
        try:
            if hasattr(self.time_embed, 'mlp') and isinstance(self.time_embed.mlp, nn.Sequential) and len(self.time_embed.mlp) > 2:
                 if isinstance(self.time_embed.mlp[0], nn.Linear):
                     nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
                     # Bias already zeroed by _basic_init
                 if isinstance(self.time_embed.mlp[2], nn.Linear):
                     nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
                     # Bias already zeroed by _basic_init
        except Exception:
             # Silently ignore if structure is not as expected, basic_init already applied
             pass
        xm.mark_step()



        # --- Specific Zero-out using .data.fill_ ---
        # Zero-out specific adaLN modulation layers and final layers
        with torch.no_grad(): # Ensure no gradients are tracked here
            # Zero-out ResBlock adaLN modulation final layers
            for block in self.res_blocks:
                try:
                     # Check if adaLN_modulation is Sequential and has layers
                     if isinstance(block.adaLN_modulation, nn.Sequential) and len(block.adaLN_modulation) > 0:
                         # Check if the last layer is Linear
                         last_layer = block.adaLN_modulation[-1]
                         if isinstance(last_layer, nn.Linear):
                             last_layer.weight.data.fill_(0.0)
                             if last_layer.bias is not None:
                                 last_layer.bias.data.fill_(0.0)
                except Exception:
                     # Silently ignore potential errors (e.g., wrong layer type)
                     pass

            # Zero-out FinalLayer adaLN modulation final layer
            try:
                 if isinstance(self.final_layer.adaLN_modulation, nn.Sequential) and len(self.final_layer.adaLN_modulation) > 0:
                     last_ada_layer = self.final_layer.adaLN_modulation[-1]
                     if isinstance(last_ada_layer, nn.Linear):
                          last_ada_layer.weight.data.fill_(0.0)
                          if last_ada_layer.bias is not None:
                               last_ada_layer.bias.data.fill_(0.0)
            except Exception:
                 pass

            # Zero-out FinalLayer linear layer
            try:
                 if isinstance(self.final_layer.linear, nn.Linear):
                     self.final_layer.linear.weight.data.fill_(0.0)
                     if self.final_layer.linear.bias is not None:
                         self.final_layer.linear.bias.data.fill_(0.0)
            except Exception:
                 pass
        xm.mark_step()


    def initialize_weights(self):
        """Robust TPU-friendly initialization with explicit manual operations instead of relying on nn.init functions."""
        print("--- Starting Manual TPU-friendly Initialization ---")
        
        # --- Helper function to manually implement xavier_uniform_ ---
        @torch.no_grad()
        def manual_xavier_uniform(tensor):
            """Manually implement xavier_uniform_ in a TPU-friendly way."""
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            
            
            bound = math.sqrt(6.0 / (fan_in + fan_out))
            
            # std = math.sqrt(2.0 / (fan_in + fan_out))
            # bound = math.sqrt(3.0) * std  # Calculate the bound
            
            # Generate uniform values between -bound and bound
            with torch.no_grad():
                # First create tensor with values in [0, 1]
                random_tensor = torch.empty_like(tensor, dtype=torch.float32).uniform_(0, 1)
                # Scale to [-bound, bound]
                random_tensor = random_tensor * (2 * bound) - bound
                # Use copy_ for reliable TPU behavior
                tensor.data.copy_(random_tensor.to(tensor.dtype))
            
            # Check value range
            with torch.no_grad():
                max_val = tensor.abs().max().item()
                if max_val > 2 * bound:
                    print(f"WARNING: After manual xavier init, max value {max_val} exceeds expected bound {bound}")
                    
            return tensor
        
        # --- Manual initialization for all layers ---
        print("Manually initializing linear layers...")
        
        # 1. First initialize all standard linear layers with xavier
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    # Skip the specific layers that need zero init (will handle them later)
                    if ("adaLN_modulation" in name and name.endswith("[-1]")) or ("final_layer.linear" in name):
                        print(f"Skipping xavier init for zero-init layer: {name}")
                        continue
                    
                    # Apply manual xavier to weight
                    print(f"Applying manual xavier init to: {name}")
                    manual_xavier_uniform(module.weight)
                    # nn.init.xavier_uniform_(tensor)
                    
                    # Zero out bias
                    if module.bias is not None:
                        module.bias.data.fill_(0.0)
                except Exception as e:
                    print(f"Error during init of {name}: {e}")
        
        # Force synchronization
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        print("XLA mark_step after manual xavier init")
        
        # 2. Initialize time embedding MLP with normal distribution
        try:
            if hasattr(self.time_embed, 'mlp') and isinstance(self.time_embed.mlp, nn.Sequential):
                if len(self.time_embed.mlp) > 0 and isinstance(self.time_embed.mlp[0], nn.Linear):
                    print("Initializing time_embed.mlp[0] with normal(0, 0.02)")
                    std = 0.02
                    normal_tensor = torch.zeros_like(self.time_embed.mlp[0].weight, dtype=torch.float32)
                    normal_tensor.normal_(mean=0.0, std=std)
                    self.time_embed.mlp[0].weight.data.copy_(normal_tensor.to(self.time_embed.mlp[0].weight.dtype))
                
                if len(self.time_embed.mlp) > 2 and isinstance(self.time_embed.mlp[2], nn.Linear):
                    print("Initializing time_embed.mlp[2] with normal(0, 0.02)")
                    std = 0.02
                    normal_tensor = torch.zeros_like(self.time_embed.mlp[2].weight, dtype=torch.float32)
                    normal_tensor.normal_(mean=0.0, std=std)
                    self.time_embed.mlp[2].weight.data.copy_(normal_tensor.to(self.time_embed.mlp[2].weight.dtype))
        except Exception as e:
            print(f"Error initializing time embed MLP: {e}")
        
        # Force synchronization
        xm.mark_step()
        print("XLA mark_step after time embed init")
        
        # 3. Zero-out specific final layers
        print("Zero-initializing special layers...")
        with torch.no_grad():
            # ResBlock adaLN modulation
            for i, block in enumerate(self.res_blocks):
                try:
                    if isinstance(block.adaLN_modulation, nn.Sequential) and len(block.adaLN_modulation) > 0:
                        last_layer = block.adaLN_modulation[-1]
                        if isinstance(last_layer, nn.Linear):
                            print(f"Zero-initializing ResBlock[{i}].adaLN_modulation[-1]")
                            last_layer.weight.data.fill_(0.0)
                            if last_layer.bias is not None:
                                last_layer.bias.data.fill_(0.0)
                except Exception as e:
                    print(f"Error during zero-init of ResBlock[{i}]: {e}")
            
            # Final layer adaLN modulation
            try:
                if isinstance(self.final_layer.adaLN_modulation, nn.Sequential) and len(self.final_layer.adaLN_modulation) > 0:
                    last_layer = self.final_layer.adaLN_modulation[-1]
                    if isinstance(last_layer, nn.Linear):
                        print("Zero-initializing final_layer.adaLN_modulation[-1]")
                        last_layer.weight.data.fill_(0.0)
                        if last_layer.bias is not None:
                            last_layer.bias.data.fill_(0.0)
            except Exception as e:
                print(f"Error during zero-init of final_layer.adaLN_modulation: {e}")
            
            # Final layer linear
            try:
                print("Zero-initializing final_layer.linear")
                self.final_layer.linear.weight.data.fill_(0.0)
                if self.final_layer.linear.bias is not None:
                    self.final_layer.linear.bias.data.fill_(0.0)
            except Exception as e:
                print(f"Error during zero-init of final_layer.linear: {e}")
        
        # Force synchronization one last time
        xm.mark_step()
        print("XLA mark_step after zero-init")
        
        # Quick checking helper
        def check_weight_statistics(name, tensor):
            if tensor is None:
                return
            
            try:
                tensor_abs_max = tensor.abs().max().item()
                tensor_mean = tensor.abs().mean().item()
                print(f"{name} - max: {tensor_abs_max:.6f}, mean: {tensor_mean:.6f}")
                
                if tensor_abs_max > 10.0:
                    print(f"!!! WARNING: {name} has very large values (max={tensor_abs_max:.6f}) !!!")
                
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"!!! ERROR: {name} has NaN or Inf values !!!")
            except Exception as e:
                print(f"Error checking {name}: {e}")
        
        # Quick check of a few key layers
        print("\n--- Quick Weight Statistics Check ---")
        
        # Check a couple of ResBlock MLP weights
        if len(self.res_blocks) > 0:
            # Check first ResBlock
            check_weight_statistics("ResBlock[0].mlp[0].weight", self.res_blocks[0].mlp[0].weight)
            check_weight_statistics("ResBlock[0].mlp[2].weight", self.res_blocks[0].mlp[2].weight)
            
            # Check a modulation layer
            check_weight_statistics("ResBlock[0].adaLN_modulation[-1].weight", self.res_blocks[0].adaLN_modulation[-1].weight)
        
        # Check final layer
        check_weight_statistics("final_layer.linear.weight", self.final_layer.linear.weight)
        
        print("--- Initialization Complete ---")


    def forward(self, x, t, y):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param y: conditioning from LLM transformer.
        :return: an [N x C] Tensor of outputs.
        """

        # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #     # xs.mark_sharding(t, xs.get_global_mesh(), ("fsdp", None))
        #     xs.mark_sharding(y, xs.get_global_mesh(), ("fsdp", None))
        #     xs.mark_sharding(x, xs.get_global_mesh(), ("fsdp", None))

        
        x = self.x_embedder(x)
        
        x = x.squeeze(dim=(-1,-2))
     
        t = self.time_embed(t)

        # y = self.y_embedder(y, train = self.training)

        y = self.y_proj(y)

        # (1) You can remove it, won't trigger all gather
        # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #     xs.mark_sharding(t, xs.get_global_mesh(), ("fsdp", None))
        #     xs.mark_sharding(y, xs.get_global_mesh(), ("fsdp", None))
        #     xs.mark_sharding(x, xs.get_global_mesh(), ("fsdp", None))

        c = t + y

        # c = t + y # Line 695


        # (2) This is the only line that is needed for SPMD
        if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
            xs.mark_sharding(c, xs.get_global_mesh(), ("fsdp", None))




        for block in self.res_blocks:
            x = block(x, c) # Line 698
            
            # (3)
            # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
            #     xs.mark_sharding(x, xs.get_global_mesh(), ("fsdp", None))

        x =  self.final_layer(x, c)


        # (4)                                                                                              
        # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #     xs.mark_sharding(x, xs.get_global_mesh(), ("fsdp", None))


        x = x.view(*x.shape, 1, 1)

        # if os.getenv("SCALE_RAE_LAUNCHER", "") == "TORCHXLA_SPMD":
        #     xs.mark_sharding(x, xs.get_global_mesh(), ("fsdp", None, None))


        return x




    def forward_with_cfg(self, x, t, y, cfg_scale, cfg_interval=(-1e4, -1e4), interval_cfg: float = 0.0):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        learned_embed = self.y_embedder.dropout_embedding[None, :].expand(y.shape[0], -1)
        y = torch.cat([y, learned_embed], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        #eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        t = t[0] # check if t < cfg_interval
        if t > cfg_interval[0] and t < cfg_interval[1]:
            if interval_cfg > 1.0:
                half_eps = uncond_eps  + interval_cfg * (cond_eps - uncond_eps)
            else:
                half_eps = cond_eps # only use conditional generation
        else:
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)