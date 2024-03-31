### Run code: training, inference and figures

Colab to run existing models for 4 organisms (data links are also there) to get figures and metrics:
https://colab.research.google.com/drive/16AW88yDjz9d0IZulpMk2fQ6i0LqmTjbL?usp=sharing

Colab to train a new model (for S. cerevisiae):
https://colab.research.google.com/drive/1L1GWDiRznlWnwKXXgaJN7-TksQdK5juw?usp=sharing

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
