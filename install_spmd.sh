pip install --upgrade pip
pip install -e '.[tpu]'
pip install torch==2.6.0 && pip install torchvision==0.21.0

# torch_xla-2.6.0+git0bb4f6f-cp310-cp310-linux_x86_64.whl is provided in the repo
pip install torch_xla-2.6.0+git0bb4f6f-cp310-cp310-linux_x86_64.whl
pip install 'torch_xla[tpu]' -f https://storage.googleapis.com/libtpu-releases/index.html
pip install 'torch_xla[pallas]' -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
pip install zstandard

# Patch TorchXLA custom_kernel.py for compatibility
sed -i '258c\      if ab is not None:' $HOME/.local/lib/python3.10/site-packages/torch_xla/experimental/custom_kernel.py
sed -i '381c\      if ab is not None:' $HOME/.local/lib/python3.10/site-packages/torch_xla/experimental/custom_kernel.py