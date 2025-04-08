**Key Features & Optimizations**:

1. **Custom Weighted MSE Loss**  
   - Implements `WeightedMSE(weight_nonzero=1000.0)` to prioritize non-zero pixel reconstruction (critical for sparse jet images).  
   - Balances focus between sparse/dense regions by weighting non-zero pixels 1000x more than zeros.

2. **Efficient Data Handling**  
   - Processes data in parallel using `ThreadPoolExecutor` with all available CPU workers (`num_workers=min(8, os.cpu_count())`).  
   - Uses only 30K samples (subset of full dataset) for training, verified to preserve generalization—confirmed by testing the final model on the full dataset with consistent loss metrics.  
   - Implements memory cleanup with `gc.collect()` during full-dataset inference.

3. **Architecture & Training**  
   - **Autoencoder**: 4-layer Conv2D encoder (64→512 channels) + symmetric ConvTranspose2D decoder with BatchNorm and LeakyReLU.  
   - **Latent Space**: Optimal 4096-dimension latent space (experimentally determined for minimal loss vs training speed).  
   - **Normalization**: Inputs rescaled to [0,1] and sparse pixels (<1e-6) augmented with noise via `HandleSparseImages` to avoid anomalies.  
   - **LR Scheduling**: `ReduceLROnPlateau` with patience=5 to dynamically adjust learning rate.
   - **Overfitting Mitigation**: Employed Dropout to minimize overfitting, though no significant increase in validation loss was observed—suggesting the model                                         generalizes well without severe overfitting.
     ![reconstructions_epoch_77.png](https://github.com/Abdelrahman10101/Genie/blob/main/Autoencoder%20of%20the%20quarkgluon%20events/reconstructions_epoch_77.png)
