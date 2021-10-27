![image](https://user-images.githubusercontent.com/79586312/127057849-d2785b2e-8f5b-4daf-bc44-ad3dbc3fd47e.png)
Hi-C is one of the most popular methods for studying the spatial organization of the genome. The result of the experiment and subsequent processing of the sequencing data is a map that shows the frequency of contacts of genome regions with each other. This model can predict such maps from a raw DNA sequence.
![Poster MCCMB](https://user-images.githubusercontent.com/79586312/138556876-739fa8c5-3939-4a50-abdd-12316b66c9e0.png)

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
python -m ipykernel install --user --name chimaera --display-name "chimaera"
```