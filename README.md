# DGSDNET: DUAL-GRAPH SPECTRAL DIFFUSION NETWORK FOR INCOMPLETE MULTIMODAL EMOTION RECOGNITION IN CONVERSATIONS
An official pytorch implementation for the paper: DGSDNET: DUAL-GRAPH SPECTRAL DIFFUSION NETWORK FOR INCOMPLETE MULTIMODAL EMOTION RECOGNITION IN CONVERSATIONS
## Environment

- Python 3.8+
- CUDA 11.8+ (recommended for training)
- PyTorch 1.12.0+
- NumPy <2.0 

For complete dependencies, see `requirements.txt`
## Datasets

The following datasets are used in this research:

- [IEMOCAP](https://sail.usc.edu/iemocap/index.html) - 7,433 utterances, 4-class emotion recognition
- [CMU-MOSI](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/) - 2,219 utterances, sentiment analysis
- We also provide the [dataset features](https://pan.baidu.com/s/1tXYsXSUGjagZjD9SqRb8Vw?pwd=dgsg) used in the code.

### Dataset Structure

```
DGSDNet/
└── dataset/
    ├── IEMOCAPFour/
    │   ├── IEMOCAP_features_raw_4way.pkl
    │   └── features/
    │       ├── wav2vec-large-c-UTT/
    │       ├── deberta-large-4-UTT/
    │       └── manet_UTT/
    └── CMUMOSI/
        ├── CMUMOSI_features_raw_2way.pkl
        └── features/
            ├── wav2vec-large-c-UTT/
            ├── deberta-large-4-UTT/
            └── manet_UTT/
```

Please place your dataset features in the `dataset/` directory following this structure.
### Running Experiments
```bash
python3 experiment_missing_rate.py \
    --dataset iemocap \
    --missing_rate 0.0 \
    --epochs 100 \
    --hidden_dim 128
```

**Parameters**:
- `--dataset`: Dataset selection (`iemocap` or `cmumosi`)
- `--missing_rate`: Missing rate (0.0 to 0.7)
- `--cv_fold`: Cross-validation fold (0-4 for IEMOCAP)
- `--epochs`: Number of training epochs (default: 100)
- `--hidden_dim`: Hidden dimension 
- `--ablation_type`: Ablation type (`full`, `w/o_SP`, `w/o_TP`, `w/o_DSD`, `w/o_GSC`)
- `--run_all`: Run all experiments for paper
### Saving Results:
The results will be saved in the `results/` directory. Different hardware and runtime environments may lead to variations in results, so it's recommended to try different hyperparameter settings or random seeds to achieve the best performance.
