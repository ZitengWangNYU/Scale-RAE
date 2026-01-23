from diffloss import RectifiedFlowProjector
import torch

def test_rectified_flow_projector(rf_proj: RectifiedFlowProjector):
    """
    Test the RectifiedFlowProjector class.
    """
    # Create a dummy input tensor
    z = torch.randn(6, rf_proj.z_channels)
    x = torch.randn(6, rf_proj.diffusion_tokens, rf_proj.diffusion_channels)

    print("z and y dypes are:", z.dtype, x.dtype)

    # Forward pass
    output = rf_proj(z, x)
    print(f"--- [FORWARD] output shape: {output.shape} --")
    
    
    # loss compute 
    loss = rf_proj.training_loss(z, x)
    print(f"--- [LOSS] loss: {loss} --")
    
    # inference
    
    x_end = rf_proj.infer(z)
    print(f"--- [INFER] x_end shape: {x_end.shape} --")
    

setups = {
    "default": {
        "diffusion_tokens": 1,
        "diffusion_channels": 1152,
        "z_channels": 4096,
        "split_per_token": 1,
        "model_hidden_size": 1152,
        "model_depth": 28,
        "model_heads": 16,
        "guidance_scale": 1.0,
    },
     "cfg": {
        "diffusion_tokens": 1,
        "diffusion_channels": 1152,
        "z_channels": 4096,
        "split_per_token": 1,
        "model_hidden_size": 1152,
        "model_depth": 28,
        "model_heads": 16,
        "guidance_scale": 2.0,
    },
    "split": {
        "diffusion_tokens": 1,
        "diffusion_channels": 1152,
        "z_channels": 4096,
        "split_per_token": 4,
        "model_hidden_size": 1152,
        "model_depth": 28,
        "model_heads": 16,
        "guidance_scale": 1.0,
    },
    "multi_tokens": {
        "diffusion_tokens": 4,
        "diffusion_channels": 1152,
        "z_channels": 4096,
        "split_per_token": 1,
        "model_hidden_size": 1152,
        "model_depth": 28,
        "model_heads": 16,
        "guidance_scale": 1.0,
    },
    "all": {
        "diffusion_tokens": 4,
        "diffusion_channels": 1152,
        "z_channels": 4096,
        "split_per_token": 4,
        "model_hidden_size": 1152,
        "model_depth": 28,
        "model_heads": 16,
        "guidance_scale": 2.0,
    },
}

def create_rf_projector(model_kwargs:dict) -> RectifiedFlowProjector:
    """
    Create a RectifiedFlowProjector instance with dummy parameters.
    """
    # Dummy parameters for testing
    return RectifiedFlowProjector(
        **model_kwargs,
        inference_step=2
    )

def test_single_setup(setup: dict):
    """
    Test a single setup for the RectifiedFlowProjector.
    """
    rf_proj = create_rf_projector(setup)
    test_rectified_flow_projector(rf_proj)

def main():
    for name, setup in setups.items():
        print(f"Testing setup: {name}")
        test_single_setup(setup)
        print("\n")
if __name__ == "__main__":
    main()