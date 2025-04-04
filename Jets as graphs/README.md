**Key Features**:

1. **Memory-Optimized Processing**:
   - Handles large datasets by splitting into manageable chunks (30,000 jets each)
   - Processes each chunk into a lightweight point cloud format 
   - Uses threshold=0.01 for energy filtering to remove anomalies/noise

2. **Graph Construction**:
   - Nodes represent detector hits with features from:
     - Spatial coordinates (normalized x,y positions)
     - Multi-channel energy deposits (ECAL, HCAL, Track)
   - Edges created via radius_neighbors_graph (radius=0.15) for local connectivity
   - Fallback to single-node graphs when no connections exist

3. **GNN Architecture**:
   - EdgeConv layers with mean aggregation to capture neighborhood information
   - Combines graph features with global jet properties (mass, pt)
   - Uses optimized hyperparameters from experimentation:
     - Hidden dimension: 128
     - Learning rate: 0.001 with ReduceLROnPlateau scheduling
     - Batch size: 32

4. **Training Optimization**:
   - Early stopping (patience=5) to prevent overfitting
   - Cross-entropy loss with ROC-AUC monitoring
   - Achieves 0.73 AUC score on validation set

