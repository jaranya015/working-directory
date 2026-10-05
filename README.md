# Custom Lane Segmentation U-Net (from scratch)

Single-class lane segmentation on the PSU-reservoir dataset (polygon labels only, Ultralytics YOLO-seg format).
Network is designed and trained from scratch (no pretrained weights), small enough for a laptop.

## Structure
```
model.py                      custom U-Net (48x48x3 -> 48x48x1)
utils.py                      YOLO-seg parser, polygon rasterizer, dataset, augmentation
train.py                      training (TensorBoard, 30 epochs)
inference.py                  writes binary masks to run/masks, overlays, memory.json
evaluation.py                 pixel-wise IoU vs ground truth
custom-unet-architecture.md   mermaid diagram
```

## Usage
```bash
pip install -r requirements.txt
python train.py --images /dir/images --labels /dir/labels --lane-class 0 --epochs 30
tensorboard --logdir runs/train/tb
python inference.py --weights runs/train/best.pt --source /dir/images --out run
python evaluation.py --masks run/masks --labels /dir/labels --lane-class 0 --out run
```
- Images may have any size: they are resized to 48x48 for the network; polygons are rasterized at the original size then resized (nearest) to make the 48x48 target. In `inference.py` the predicted probability map is resized back to the original size, so evaluation is done at full resolution.
- Only polygon rows are used (>= 3 points); bounding-box / polyline-style rows are ignored.
- `--batch` defaults to 4 on GPU and 2 on CPU. The validation split (20%) is saved in `runs/train/val_files.txt`; use that subset for the final evaluation (copy those images to a folder and pass it to `inference.py`).

## Network design and reasons
See `custom-unet-architecture.md`. ~0.48 M parameters (~1.9 MB fp32).
| Choice | Reason |
|---|---|
| U-Net encoder-decoder, 3 down-samplings (48->24->12->6) | Lane is a thin, elongated structure; skip connections keep fine spatial detail lost by pooling. 3 levels is enough for a 48x48 input (6x6 bottleneck still has a large receptive field). |
| Channels 16-32-64-128 | Small memory footprint for laptop / CPU inference. |
| GroupNorm instead of BatchNorm | Batch size is only 2-4; BatchNorm statistics would be noisy. |
| ConvTranspose 2x2 up-sampling | Learnable up-sampling, from scratch. |
| Loss = BCE + Dice | Lane pixels are a small fraction of the image (class imbalance); Dice directly optimizes overlap. |
| Adam 1e-3 + cosine LR | Stable convergence within 30 epochs. |
| Augmentation: white-balance gain (0.92-1.08), brightness (+-0.08), 3x3 blur | Mimics sunlight / camera variation, kept slight so labels stay valid. |

## Results (fill in after running)
- Loss curve: `runs/train/loss_curve.png` (and TensorBoard)  
  ![loss](runs/train/loss_curve.png)
- Metrics (`run/eval_summary.json`, IoU threshold 0.6):

| images | detected (IoU>0.6) | detection rate | mean IoU of detected |
|---|---|---|---|
| TBD | TBD | TBD | TBD |

- Inference snapshot (left: input, right: predicted lane overlay): `run/overlays/*.png`  
  ![snapshot](run/overlays/example.png)
- Memory footprint: `run/memory.json` (weights size, process RSS, CUDA peak, ms/image)
# working-directory
