# CLCC

## Requirements
* Python 3.8+
* Refer to `requirements.txt` for other dependencies

## Usage

### Extract dataset
```bash
tar -xzf dataset/java-python-clones.db.tar.gz -C dataset/
```

### Configuration
Please refer to `src/config.py` for configuration options.

### Load database
```bash
python3 src/load_data.py
```

### Generate dataset
```bash
python3 src/generate_dataset.py
```

### Embedding with CodeBERT
```bash
python3 src/codebert_embedding.py
```

### Train model
```bash
python3 src/train.py
```

### Evaluate model
```bash
python3 src/classify.py
```
