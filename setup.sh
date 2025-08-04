pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
pip install numpy datasets matplotlib psutil protobuf transformers==4.50.3