# RNN Basics

Character-level language modeling on the **tiny-Shakespeare** dataset, implementing and comparing three classic recurrent architectures **manually** (built from `nn.Linear`/tensor ops rather than `nn.RNN`/`nn.LSTM`/`nn.GRU`): a vanilla RNN, an LSTM, and a GRU.

## Contents

| Notebook | Description |
|---|---|
| `RNN-manual.ipynb` | A vanilla RNN cell implemented manually, trained on character-level text. |
| `LSTM-manual.ipynb` | An LSTM cell implemented manually. |
| `GRU-manual.ipynb` | A GRU cell implemented manually. |
| `comparision.ipynb` | Trains all three architectures side by side and compares them via a results table and training curves. |

## Dataset

Character-level text from `data/tinyshakespeare/input.txt`, split 90/10 into train/validation.

## Technologies

- Python
- PyTorch
- Matplotlib

## Getting Started

```bash
git clone https://github.com/shehryar11w/RNN-basics.git
cd RNN-basics
pip install torch matplotlib
```

Open any notebook in Jupyter or VS Code and run. `comparision.ipynb` trains and compares all three models.

## Learning Outcomes

- How RNN/LSTM/GRU cells work internally (gates, hidden state updates, backpropagation through time)
- Practical differences in training stability and performance between the three architectures

## License

Educational / personal use.
