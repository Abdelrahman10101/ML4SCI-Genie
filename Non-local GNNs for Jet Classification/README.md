### **Key Features**


1. **Memory-Optimized Graph Construction**  
   - Processes 30,000-jet chunks to handle large datasets efficiently  
   - Converts 3-channel jet images (ECAL, HCAL, tracks) into graphs with:  
     - **Nodes**: Spatial coordinates (normalized x/y) + energy deposits  
     - **Edges**: Radius-based connectivity (0.15 threshold) via `radius_neighbors_graph`  
     - Energy threshold (0.01) filters noise while fallback ensures single-node graphs  

2. **Non-Local Attention Module**  
   - **Key Components**:  
     - Query/Key/Value projections (`theta`, `phi`, `g` linear layers)  
     - Scaled dot-product attention across all nodes  
     - Residual connection to preserve local features  
   - Captures global relationships beyond local neighborhoods  

3. **Hybrid Architecture**  
   - **Local Processing**:  
     - 3× EdgeConv layers with mean aggregation (hidden_dim=64)  
     - LeakyReLU activations + batch normalization  
   - **Global Context**:  
     - Non-local block after convolutional layers  
     - Jet properties (mass `m0`, transverse momentum `pt`) fused via MLP  

4. **Training Optimization**  
   - **Early Stopping**: Halts training if validation loss plateaus (patience=5)  
   - **Dynamic LR**: `ReduceLROnPlateau` (factor=0.5) adjusts learning rate from 0.001  
   - **Metrics**: Tracks ROC-AUC (0.7299 baseline → 0.7293 with non-local)  

5. **Efficiency Features**  
   - Chunked data loading with memory cleanup (`gc.collect()`)  
   - Batch processing (size=32) for GPU utilization  

**Results**: The non-local variant achieves comparable performance (ΔAUC -0.0006) while demonstrating the feasibility of attention mechanisms in particle physics graph networks. The architecture maintains efficiency with a 25-epoch training cycle (early stopping) and modular design for further experimentation.
