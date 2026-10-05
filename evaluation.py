import argparse, os, csv, json
import cv2, numpy as np
from utils import read_polygons, polygons_to_mask, label_path_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default="run/masks", help="predicted masks from inference.py")
    ap.add_argument("--labels", required=True, help="folder of YOLO-seg ground-truth .txt")
    ap.add_argument("--lane-class", type=int, default=None)
    ap.add_argument("--iou-thr", type=float, default=0.6)
    ap.add_argument("--out", default="run")
    a = ap.parse_args()

    rows = []
    for f in sorted(os.listdir(a.masks)):
        if not f.lower().endswith(".png"):
            continue
        pred = cv2.imread(os.path.join(a.masks, f), cv2.IMREAD_GRAYSCALE) > 127
        h, w = pred.shape
        polys = read_polygons(label_path_for(f, a.labels), a.lane_class)
        if not polys:
            continue                                    # no ground truth lane -> skip
        gt = polygons_to_mask(polys, h, w).astype(bool)
        inter, union = (pred & gt).sum(), (pred | gt).sum()
        iou = inter / union if union else 0.0
        rows.append((f, float(iou), "Yes" if iou > a.iou_thr else "No"))

    det = [r[1] for r in rows if r[2] == "Yes"]
    summary = {"images_evaluated": len(rows), "iou_threshold": a.iou_thr,
               "detected": len(det), "detection_rate": round(len(det) / max(1, len(rows)), 4),
               "mean_iou_of_detected": round(float(np.mean(det)), 4) if det else None,
               "mean_iou_all": round(float(np.mean([r[1] for r in rows])), 4) if rows else None}
    with open(os.path.join(a.out, "eval_per_image.csv"), "w", newline="") as fh:
        w_ = csv.writer(fh); w_.writerow(["image", "iou", "detected"]); w_.writerows(rows)
    json.dump(summary, open(os.path.join(a.out, "eval_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
