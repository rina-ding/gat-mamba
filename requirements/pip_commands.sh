apt-get update -y
apt-get install -y openslide-tools
apt-get install python3-openslide -y
pip install openslide-python 
pip install scikit-image
pip install natsort
pip install timm
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse  -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install lifelines
pip install causal-conv1d==1.0.2
pip install mamba-ssm==1.2.0.post1
pip install scikit-survival