import argparse, os, json, time
import cv2, numpy as np, torch
from model import LaneUNet
from utils import list_images, preprocess


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/train/best.pt")
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--out", default="run")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "overlays"), exist_ok=True)
    net = LaneUNet().to(a.device)
    net.load_state_dict(torch.load(a.weights, map_location=a.device)); net.eval()
    paths = [a.source] if os.path.isfile(a.source) else list_images(a.source)

    rss0 = rss_mb()
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for p in paths:
            img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            prob = torch.sigmoid(net(preprocess(img).to(a.device)))[0, 0].cpu().numpy()
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)    # back to original size
            mask = (prob > 0.5).astype(np.uint8) * 255
            stem = os.path.splitext(os.path.basename(p))[0]
            cv2.imwrite(os.path.join(a.out, "masks", stem + ".png"), mask)
            ov = img.copy(); ov[mask > 0] = (0.5 * ov[mask > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            cv2.imwrite(os.path.join(a.out, "overlays", stem + ".png"),
                        cv2.cvtColor(np.hstack([img, ov]), cv2.COLOR_RGB2BGR))   # before | after
    dt = (time.time() - t0) / max(1, len(paths))

    n_par = sum(p.numel() for p in net.parameters())
    mem = {"images": len(paths), "params": n_par, "weights_MB_fp32": round(n_par * 4 / 2**20, 3),
           "process_RSS_MB": round(rss_mb(), 1), "RSS_increase_during_inference_MB": round(rss_mb() - rss0, 1),
           "cuda_peak_allocated_MB": round(torch.cuda.max_memory_allocated() / 2**20, 2) if a.device == "cuda" else None,
           "avg_time_per_image_ms": round(dt * 1000, 2), "device": a.device}
    json.dump(mem, open(os.path.join(a.out, "memory.json"), "w"), indent=2)
    print(json.dumps(mem, indent=2))


if __name__ == "__main__":
    main()
