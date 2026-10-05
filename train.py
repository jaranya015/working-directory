import argparse, os, random, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import LaneUNet
from utils import LaneDataset, list_images


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    return (1 - (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def batch_iou(logits, target):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * target).sum((1, 2, 3))
    union = ((p + target) > 0).float().sum((1, 2, 3))
    return ((inter + 1e-6) / (union + 1e-6)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of images (various sizes)")
    ap.add_argument("--labels", required=True, help="folder of YOLO-seg .txt (polygon)")
    ap.add_argument("--lane-class", type=int, default=None, help="class id of lane (default: all polygons)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=0, help="0 = auto (4 on GPU, 2 on CPU)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--out", default="runs/train")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bs = a.batch or (4 if dev == "cuda" else 2)
    os.makedirs(a.out, exist_ok=True)

    paths = list_images(a.images)
    random.shuffle(paths)
    nval = max(1, int(len(paths) * a.val_split))
    val_p, tr_p = paths[:nval], paths[nval:]
    open(os.path.join(a.out, "val_files.txt"), "w").write("\n".join(val_p))
    print(f"device={dev} batch={bs} train={len(tr_p)} val={len(val_p)}")

    tr = DataLoader(LaneDataset(tr_p, a.labels, a.lane_class, train=True), bs, shuffle=True)
    va = DataLoader(LaneDataset(val_p, a.labels, a.lane_class, train=False), bs)

    net = LaneUNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    bce = nn.BCEWithLogitsLoss()
    tb = SummaryWriter(os.path.join(a.out, "tb"))
    rows, best = [], 1e9

    for ep in range(1, a.epochs + 1):
        net.train(); tl = 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            out = net(x)
            loss = bce(out, y) + dice_loss(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(x)
        tl /= len(tr.dataset)
        net.eval(); vl = vi = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(dev), y.to(dev)
                out = net(x)
                vl += (bce(out, y) + dice_loss(out, y)).item() * len(x)
                vi += batch_iou(out, y) * len(x)
        vl /= len(va.dataset); vi /= len(va.dataset)
        sched.step()
        tb.add_scalar("loss/train", tl, ep); tb.add_scalar("loss/val", vl, ep); tb.add_scalar("iou/val", vi, ep)
        rows.append((ep, tl, vl, vi))
        print(f"epoch {ep:02d}/{a.epochs} train_loss={tl:.4f} val_loss={vl:.4f} val_iou(48x48)={vi:.4f}")
        torch.save(net.state_dict(), os.path.join(a.out, "last.pt"))
        if vl < best:
            best = vl; torch.save(net.state_dict(), os.path.join(a.out, "best.pt"))
    tb.close()

    with open(os.path.join(a.out, "losses.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss", "val_loss", "val_iou"]); w.writerows(rows)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e = [r[0] for r in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(e, [r[1] for r in rows], label="train"); plt.plot(e, [r[2] for r in rows], label="val")
    plt.xlabel("epoch"); plt.ylabel("BCE + Dice loss"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(a.out, "loss_curve.png"), dpi=150)
    print("saved:", a.out)


if __name__ == "__main__":
    main()
