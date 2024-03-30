### Installation with conda

Pull the project:
```bash
git clone https://github.com/ashkolikov/chimaera.git
cd chimaera
```

Create environment:
```bash
conda env create -f environment.yml
conda activate chimaera
```

Install (dev mode):
```bash
pip install -e .
```

Run tests: 
```bash
pip install -e .
```

Enable Jupyter support: 
```bash
pip install ipykernel
python -m ipykernel install --user --name chimaera --display-name "chimaera"
```
