# Occluded Face Recognition (LFW)

## 1) Install
```powershell
pip install -r requirements.txt
```

## 2) Train
```powershell
python train_occluded_gcn.py ^
  --data-root data/lfw-deepfunneled/lfw-deepfunneled ^
  --epochs 20 ^
  --batch-size 64 ^
  --max-classes 200 ^
  --min-images-per-identity 10
```

## 3) Outputs
Saved in `runs/occluded_gcn/`:
- `best_model.pt`
- `class_to_idx.json`
- `history.json`
- `summary.json`
- `landmark_cache.npz`

## Notes
- LFW is long-tail; `--min-images-per-identity` and `--max-classes` are used to stabilize training.
- Occlusion is simulated on landmark-node features during training via random rectangle masking.
