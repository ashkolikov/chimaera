### Run code: training, inference and figures

[Colab](https://colab.research.google.com/drive/1h7yygrz1L0Sd3FlHMJREgv2B8cQIf-Wf?usp=sharing) with model for Homo sapiens - training, evaluation and interpretation methods.

[Colab](https://colab.research.google.com/drive/1L1GWDiRznlWnwKXXgaJN7-TksQdK5juw?usp=sharing) with model for Mus musculus including pattern calling on cell cycle and CTCF motif removal.

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

Enable Jupyter support: 
```bash
pip install ipykernel
python -m ipykernel install --user --name chimaera --display-name "chimaera"
```
