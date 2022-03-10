# LibVQ
LibVQ 


## Install
```
git clone https://github.com/staoxiao/LibVQ.git
cd LibVQ
pip install .
```

## Data Format
Please refer to [dataset README](./LibVQ/dataset/README.md)


## Index
```python
import numpy as np
docs = np.
queries = np.

```

- ScaNN
```python
from LibVQ.baseindex import ScaNNIndex

```

- Faiss
```python
from LibVQ.baseindex import FaissIndex
faiss_index = FaissIndex()

```


- ours
```python
from LibVQ.learnable_index import LearnableIndex


```






