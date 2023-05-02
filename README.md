# purple-floyd
DeepFloyd server that runs on 8gb vram

Instructions:

Make a new Conda env:
```
conda create -n "deepfloyd" python=3.10
conda activate deepfloyd
```

Install pytorch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

Install requirements:
```
pip3 install --upgrade diffusers diffuser transformers safetensors sentencepiece accelerate bitsandbytes torch torchvision click protobuf six
```

Run:
```
python deepfloyd.py
```
