# atlas-tokenizer

A custom BPE tokenizer trained on Dolmino dataset for the Atlas language model.

## Training Details

- **Algorithm**: SentencePiece BPE (same as LLaMA)
- **Vocabulary Size**: 32000
- **Training Data**: Dolmino mix (subset of Dolma)
- **Special Tokens**: <pad>, <unk>, <s>, </s>

## Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path/to/atlas-tokenizer")

text = "Hello, world!"
tokens = tokenizer.encode(text)
print(tokens)
```

## Model Compatibility

This tokenizer is designed for:
- Atlas 50M/100M language models
- Any causal language model with vocab_size=32000

## License

Apache 2.0 (same as training data)
