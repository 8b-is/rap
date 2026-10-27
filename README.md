# RAP: Retrieval-Augmented Planning with Contextual Memory for Multimodal LLM Agents
This repository is the implementation of [RAP](https://arxiv.org/abs/2402.03610).

![RAP figure](./figure/figure.png)

# Get Started
Please refer to the following README's for each benchmark.
* [ALFWorld](./alfworld/README.md)
* [WebShop](./webshop/README.md)
* Franka Kitchen
* MetaWorld

# Standalone BFP-64 memory prototype

[`rap_mem8_bfp64_pq.py`](./rap_mem8_bfp64_pq.py) is an experimental, process-local
memory bank. It accepts float embeddings, quantizes them in blocks of up to 64
values, and ranks episodes using approximate integer cosine scores. The ALFWorld
and WebShop scripts use their own sentence-transformer retrieval paths and do not
import this prototype. It provides no persistence, update or deletion API, and
its stored `reward_pq` field does not scale retrieval scores.

`raw_block_bytes` and `compression_ratio_pq()` estimate a hypothetical packed
payload: one byte per mantissa plus one byte per block exponent. For 384 values,
that is 390 bytes versus 1536 bytes of FP32 data. The implementation stores Python
integer tuples and cached norm metadata; it does not serialize a 390-byte row or
measure actual memory savings. Query preparation uses floating-point operations
and an integer square root; only scoring of prepared rows is integer arithmetic.

Run the prototype's standard-library tests with:

```sh
python -m unittest discover -s tests -p 'test_rap_mem8_bfp64_pq.py'
```

With NumPy and PyTorch installed from the benchmark dependencies, run all focused
Python tests with:

```sh
python -m unittest discover -s tests -p 'test_*.py'
python -O -m unittest discover -s tests -p 'test_*.py'
```

The benchmark retrieval tests extract only the relevant function definitions,
using synthetic embeddings and a test cosine function. They skip when NumPy or
PyTorch is unavailable. They do not import benchmark startup, download models,
contact a model API, or validate full benchmark outcomes. The standalone Rust
files under `tests/` are not part of this Python suite; no Cargo project or GPU
integration is defined here.

# Citation
If you find RAP helpful in your research, please consider citing.
```bibtex
@misc{kagaya2024rap,
      title={RAP: Retrieval-Augmented Planning with Contextual Memory for Multimodal LLM Agents}, 
      author={Tomoyuki Kagaya and Thong Jing Yuan and Yuxuan Lou and Jayashree Karlekar and Sugiri Pranata and Akira Kinose and Koki Oguri and Felix Wick and Yang You},
      year={2024},
      eprint={2402.03610},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# License
MIT license
