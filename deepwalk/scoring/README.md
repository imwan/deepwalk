 
## `gp_deepwalk.py`: performs optimization of parameters for random walks, word2vec, fastText.

Example parameter bounds for GPyOpt:

```python
mixed_domain = [
    {'name': 'size', 'type': 'discrete', 'domain': (32, 64, 100, 300, 500)},
    {'name': 'method', 'type': 'discrete', 'domain': (0, 1)},
    {'name': 'window', 'type': 'discrete', 'domain': (3, 5, 10)},
    {'name': 'negative', 'type': 'discrete', 'domain': (3, 5, 10)}
]
```

```bash
# Run optimization with run ID "dw-test"
python gp_deepwalk.py bo dw-test

# Check progress logs
python gp_deepwalk.py load dw-test
```

Everything gets saved to a directory speciffic to each graph:

```python
# This is the directory for a speciffic graph
base_dir = '/mnt/raid1/deepwalk/blogcat'
```

## `scoring.py`: performs multi-label classification on BlogCatalog data.

```bash
# Test with:
python scoring.py
```