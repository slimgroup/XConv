"""Publication figures for the XConv spleen-UNet finetuning study.

Uses MONAI's own visualization utilities (``blend_images``, ``matshow3d``):

* ``segmentation_panel`` — input CT | ground-truth overlay | prediction overlay on
  the most-spleen axial slice (the model is the bundle's pretrained UNet = the
  finetuning init; 60 finetune steps are negligible on this near-converged model).
* ``segmentation_3d`` — a grid of axial slices with the predicted overlay (matshow3d).
* ``comparison`` — peak memory and Dice, exact vs XConv, read from results/*.json.

    python visualize.py                 # all figures (seg needs GPU; comparison is CPU)
    python visualize.py --compare_only  # just the bar chart (CPU, from JSONs)
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.visualize import blend_images, matshow3d

import bundle as B
import xconv_ops as xc  # noqa: F401 (kept for optional converted-model viz)

_HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.unicode_minus": False, "font.size": 11})
EXACT, XCONV = "#1f77b4", "#d62728"


def _save(fig, path):
    fig.savefig(path + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}.pdf / .png", flush=True)


def _to_rgb(blend):  # (3,H,W) tensor -> (H,W,3) numpy in [0,1]
    a = blend.detach().cpu().numpy()
    a = np.moveaxis(a, 0, -1)
    return np.clip(a, 0, 1)


def segmentation_figures(dataset_dir, out_dir, device):
    model = B.pretrained_net(dataset_dir, device)
    model.eval()
    batch = next(iter(B.val_loader(dataset_dir, 1, 0)))
    img = batch["image"].to(device)           # (1,1,H,W,D)
    lbl = batch["label"][0].cpu().float()      # (1,H,W,D)
    with torch.no_grad():
        logits = sliding_window_inference(img, (96, 96, 96), 4, model, overlap=0.25)
    pred = torch.argmax(logits, dim=1, keepdim=True)[0].cpu().float()   # (1,H,W,D)
    ct = img[0].cpu().float()                                           # (1,H,W,D)

    # axial slice with the most spleen in the ground truth
    z = int(lbl[0].sum(dim=(0, 1)).argmax())
    ct_s, gt_s, pr_s = ct[:, :, :, z], lbl[:, :, :, z], pred[:, :, :, z]   # each (1,H,W)
    gt_b = blend_images(ct_s, gt_s, alpha=0.5, cmap="hsv", rescale_arrays=True)
    pr_b = blend_images(ct_s, pr_s, alpha=0.5, cmap="hsv", rescale_arrays=True)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.7))
    ax[0].imshow(ct_s[0].numpy().T, cmap="gray", origin="lower"); ax[0].set_title("input CT")
    ax[1].imshow(np.transpose(_to_rgb(gt_b), (1, 0, 2)), origin="lower"); ax[1].set_title("ground truth")
    ax[2].imshow(np.transpose(_to_rgb(pr_b), (1, 0, 2)), origin="lower"); ax[2].set_title("prediction")
    for a in ax:
        a.axis("off")
    _save(fig, os.path.join(out_dir, "segmentation_panel"))

    # 3D grid of slices with predicted overlay (only slices that contain spleen)
    zc = torch.where(pred[0].sum(dim=(0, 1)) > 0)[0]
    if len(zc) >= 4:
        lo, hi = int(zc.min()), int(zc.max())
        vol = blend_images(ct[:, :, :, lo:hi + 1], pred[:, :, :, lo:hi + 1],
                           alpha=0.5, cmap="hsv", rescale_arrays=True)  # (3,H,W,d)
        try:
            fig2 = plt.figure(figsize=(11, 8))
            matshow3d(volume=vol, fig=fig2, title="XConv-finetuned spleen segmentation (axial slices)",
                      frames_per_row=6, frame_dim=-1, channel_dim=0, every_n=1, cmap=None)
            _save(fig2, os.path.join(out_dir, "segmentation_3d"))
        except Exception as e:
            print(f"matshow3d failed ({e}); skipping 3D grid", flush=True)


def comparison_figure(res_dir, out_dir):
    bfs = glob.glob(os.path.join(res_dir, "baseline_unet_spleen_b*.json"))
    xfs = glob.glob(os.path.join(res_dir, "xconv_unet_spleen_b*.json"))
    if not bfs or not xfs:
        print("comparison: missing result JSONs", flush=True)
        return
    # pin to the HEADLINE batch (the largest we ran = 64); smaller-batch runs that
    # only exist for the loss/segmentation figures must not hijack this chart.
    bcands = [d for d in (json.load(open(f)) for f in bfs) if "peak_mib" in d]
    if not bcands:
        print("comparison: no baseline JSON with peak_mib", flush=True)
        return
    b = max(bcands, key=lambda d: d.get("resolved_batch") or 0)
    bb = b.get("resolved_batch")
    xs = [d for d in (json.load(open(f)) for f in xfs)
          if "peak_mib" in d and d.get("resolved_batch") == bb]
    xs.sort(key=lambda d: d.get("resolved_ps") or 0)
    labels = ["exact"] + [f"XConv\nr={x['resolved_ps']}, {x['n_steps']} st." for x in xs]
    mems = [b["peak_mib"]] + [x["peak_mib"] for x in xs]
    dices = [b.get("val_dice") or 0] + [x.get("val_dice") or 0 for x in xs]
    # exact blue; XConv reds, darker = smaller r (more saving)
    reds = ["#fb6a4a", "#de2d26", "#a50f15", "#67000d"]
    colors = [EXACT] + [reds[min(i, len(reds) - 1)] for i in range(len(xs))]

    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6))
    ax[0].bar(labels, mems, color=colors, width=0.7)
    ax[0].axhline(b["peak_mib"], ls="--", lw=0.8, color=EXACT)
    ax[0].set_ylabel("peak memory (MiB)"); ax[0].set_title("memory")
    ax[1].bar(labels, dices, color=colors, width=0.7)
    ax[1].axhline(b.get("val_dice") or 0, ls="--", lw=0.8, color=EXACT)
    ax[1].set_ylabel("foreground Dice"); ax[1].set_ylim(0, 1); ax[1].set_title("accuracy")
    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        a.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"spleen UNet finetune at batch {b.get('resolved_batch')}", y=1.02, fontsize=11)
    _save(fig, os.path.join(out_dir, "comparison"))


def _pick_loss_npy(res_dir, train_glob):
    """Newest matching train-loss .npy, preferring a run that also recorded
    validation loss, then the longest (the 600-step run). Returns (train, val) paths."""
    fs = [f for f in glob.glob(os.path.join(res_dir, train_glob))
          if not f.endswith("_val_losses.npy")]
    if not fs:
        return None, None
    valf = lambda f: f[:-len("_losses.npy")] + "_val_losses.npy"
    withval = [f for f in fs if os.path.exists(valf(f))]
    train = max(withval or fs, key=lambda f: (len(np.load(f)), os.path.getmtime(f)))
    vf = valf(train)
    return train, (vf if os.path.exists(vf) else None)


def loss_figure(res_dir, out_dir):
    """Training (solid) and, if available, validation (dashed) loss vs step:
    exact/conv = blue, XConv = red. Auto-picks the 600-step r=4 run vs the matching
    baseline (preferring whichever runs recorded validation loss)."""
    bt_f, bv_f = _pick_loss_npy(res_dir, "baseline_unet_spleen_b*_losses.npy")
    xt_f, xv_f = _pick_loss_npy(res_dir, "xconv_unet_spleen_b*_r4_independent_conv_losses.npy")
    if bt_f is None or xt_f is None:
        print("loss: missing 600-step train curves", flush=True)
        return
    bt, xt = np.load(bt_f), np.load(xt_f)
    bv = np.load(bv_f) if bv_f else None                 # rows (step, loss)
    xv = np.load(xv_f) if xv_f else None

    def smooth(a, k=15):
        if len(a) < k:
            return a
        return np.convolve(a, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(np.arange(1, len(bt) + 1)[len(bt) - len(smooth(bt)):], smooth(bt),
            color=EXACT, lw=1.6, label="conv — train")
    ax.plot(np.arange(1, len(xt) + 1)[len(xt) - len(smooth(xt)):], smooth(xt),
            color=XCONV, lw=1.6, label="XConv r=4 — train")
    if bv is not None and len(bv):
        ax.plot(bv[:, 0], bv[:, 1], "--o", ms=3, color=EXACT, lw=1.2, label="conv — val")
    if xv is not None and len(xv):
        ax.plot(xv[:, 0], xv[:, 1], "--o", ms=3, color=XCONV, lw=1.2, label="XConv r=4 — val")
    import re
    m = re.search(r"_b(\d+)_", os.path.basename(bt_f))
    bsz = m.group(1) if m else "?"
    ax.set_xlabel("training step"); ax.set_ylabel("DiceCE loss")
    ax.set_title(f"spleen UNet finetune — loss (batch {bsz}, lr 2e-4, {len(bt)} steps)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    _save(fig, os.path.join(out_dir, "loss_curves"))


def _load_for_inference(dataset_dir, ckpt_path, device):
    """A plain UNet with finetuned weights loaded. XConv's FORWARD is the exact conv
    (it only changes the weight gradient), so a plain conv net reproduces either
    model's predictions exactly; any extra XConv buffers are ignored (strict=False)."""
    net = B.bare_net(dataset_dir, device)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net


def _fg_dice(pred, gt):  # both (1,H,W,D) class-index maps; foreground = spleen (>0)
    p, g = (pred[0] > 0).float(), (gt[0] > 0).float()
    return float(2 * (p * g).sum() / (p.sum() + g.sum() + 1e-8))


def segmentation_comparison(dataset_dir, out_dir, device, n_volumes=3, n_slices=4):
    """Per-slice GT │ conv │ XConv predicted-spleen overlays on held-out volumes,
    from the two finetuned checkpoints (results/checkpoints/). One figure per volume,
    annotated with each model's foreground Dice — the qualitative companion to the
    headline 'same accuracy at less memory' claim."""
    ck = os.path.join(_HERE, "results", "checkpoints")
    pick = lambda stem: (sorted(glob.glob(os.path.join(ck, stem)), key=os.path.getmtime) or [None])[-1]
    conv_ckpt = pick("baseline_unet_spleen_b*.pth")
    xconv_ckpt = pick("xconv_unet_spleen_b*_r4_independent_conv.pth")
    if not conv_ckpt or not xconv_ckpt:
        print("seg_compare: missing checkpoints under results/checkpoints/ "
              "(run finetune with --save_ckpt 1)", flush=True)
        return
    print(f"seg_compare: conv={os.path.basename(conv_ckpt)} "
          f"xconv={os.path.basename(xconv_ckpt)}", flush=True)
    seg_dir = os.path.join(out_dir, "segmentation")
    os.makedirs(seg_dir, exist_ok=True)
    m_conv = _load_for_inference(dataset_dir, conv_ckpt, device)
    m_xconv = _load_for_inference(dataset_dir, xconv_ckpt, device)
    for vi, batch in enumerate(B.val_loader(dataset_dir, n_volumes, 0)):
        img = batch["image"].to(device)
        lbl = batch["label"][0].cpu().float()                  # (1,H,W,D)
        with torch.no_grad():
            lc = sliding_window_inference(img, (96, 96, 96), 4, m_conv, overlap=0.25)
            lx = sliding_window_inference(img, (96, 96, 96), 4, m_xconv, overlap=0.25)
        pc = torch.argmax(lc, 1, keepdim=True)[0].cpu().float()
        px = torch.argmax(lx, 1, keepdim=True)[0].cpu().float()
        ct = img[0].cpu().float()
        dc, dx = _fg_dice(pc, lbl), _fg_dice(px, lbl)
        sums = lbl[0].sum(dim=(0, 1))
        zsel = sorted(int(z) for z in torch.argsort(sums, descending=True)[:n_slices]
                      if sums[int(z)] > 0)
        if not zsel:
            continue
        cols = [("ground truth", lbl), ("conv (exact)", pc), ("XConv r=4", px)]
        fig, ax = plt.subplots(len(zsel), 3, figsize=(9, 3 * len(zsel)))
        ax = np.atleast_2d(ax)
        for r, z in enumerate(zsel):
            ct_s = ct[:, :, :, z]
            for c, (name, vol) in enumerate(cols):
                bimg = blend_images(ct_s, vol[:, :, :, z], alpha=0.5, cmap="hsv",
                                    rescale_arrays=True)
                ax[r, c].imshow(np.transpose(_to_rgb(bimg), (1, 0, 2)), origin="lower")
                ax[r, c].axis("off")
                if r == 0:
                    ax[r, c].set_title(name)
            ax[r, 0].text(-0.06, 0.5, f"z={z}", transform=ax[r, 0].transAxes,
                          rotation=90, va="center", ha="right", fontsize=9)
        fig.suptitle(f"held-out volume {vi + 1} — foreground Dice:  "
                     f"conv {dc:.3f}  ·  XConv r=4 {dx:.3f}", y=1.0, fontsize=11)
        _save(fig, os.path.join(seg_dir, f"volume_{vi + 1}"))


def _save_tile(rgb, path):
    """Save a borderless, full-bleed image tile (no axes/title/padding) as png+pdf,
    so tiles for the same slice are pixel-identical and assemble cleanly side by side."""
    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(rgb, origin="lower")
    fig.savefig(path + ".png", dpi=200)
    fig.savefig(path + ".pdf")
    plt.close(fig)


def segmentation_individual(dataset_dir, out_dir, device, n_volumes=3, n_slices=4):
    """Fully individual per-slice tiles: for each held-out volume and axial slice, one
    standalone borderless image per column — raw CT, ground truth, conv (exact), and
    XConv r=4 — named so the set for a slice sorts together (assemble side by side)."""
    ck = os.path.join(_HERE, "results", "checkpoints")
    pick = lambda stem: (sorted(glob.glob(os.path.join(ck, stem)), key=os.path.getmtime) or [None])[-1]
    conv_ckpt = pick("baseline_unet_spleen_b*.pth")
    xconv_ckpt = pick("xconv_unet_spleen_b*_r4_independent_conv.pth")
    if not conv_ckpt or not xconv_ckpt:
        print("seg_individual: missing checkpoints (run finetune with --save_ckpt 1)", flush=True)
        return
    out = os.path.join(out_dir, "segmentation", "slices")
    os.makedirs(out, exist_ok=True)
    m_conv = _load_for_inference(dataset_dir, conv_ckpt, device)
    m_xconv = _load_for_inference(dataset_dir, xconv_ckpt, device)
    n = 0
    for vi, batch in enumerate(B.val_loader(dataset_dir, n_volumes, 0)):
        img = batch["image"].to(device)
        lbl = batch["label"][0].cpu().float()
        with torch.no_grad():
            lc = sliding_window_inference(img, (96, 96, 96), 4, m_conv, overlap=0.25)
            lx = sliding_window_inference(img, (96, 96, 96), 4, m_xconv, overlap=0.25)
        pc = torch.argmax(lc, 1, keepdim=True)[0].cpu().float()
        px = torch.argmax(lx, 1, keepdim=True)[0].cpu().float()
        ct = img[0].cpu().float()
        sums = lbl[0].sum(dim=(0, 1))
        zsel = sorted(int(z) for z in torch.argsort(sums, descending=True)[:n_slices]
                      if sums[int(z)] > 0)
        for z in zsel:
            ct_s = ct[:, :, :, z]
            stem = os.path.join(out, f"vol{vi + 1}_z{z:03d}")
            # raw CT (grayscale, no overlay)
            ct_rgb = np.repeat(np.clip(ct_s[0].numpy().T, 0, 1)[..., None], 3, axis=2)
            _save_tile(ct_rgb, stem + "_ct")
            for kind, vol in (("gt", lbl), ("conv", pc), ("xconv", px)):
                bimg = blend_images(ct_s, vol[:, :, :, z], alpha=0.5, cmap="hsv",
                                    rescale_arrays=True)
                _save_tile(np.transpose(_to_rgb(bimg), (1, 0, 2)), f"{stem}_{kind}")
            n += 4
    print(f"seg_individual: wrote {n} tiles (png+pdf) to {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default=os.path.join(_HERE, "data", "Task09_Spleen"))
    p.add_argument("--out_dir", default=os.path.join(_HERE, "results", "figures"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--compare_only", action="store_true")
    p.add_argument("--seg_only", action="store_true")
    p.add_argument("--loss_only", action="store_true")
    p.add_argument("--seg_compare", action="store_true",
                   help="per-slice GT|conv|XConv overlays from the two finetuned checkpoints")
    p.add_argument("--seg_individual", action="store_true",
                   help="fully individual borderless tiles (CT, GT, conv, XConv) per slice")
    a = p.parse_args()
    res = os.path.join(_HERE, "results")
    os.makedirs(a.out_dir, exist_ok=True)
    if a.loss_only:
        os.makedirs(os.path.join(a.out_dir, "loss"), exist_ok=True)
        loss_figure(res, os.path.join(a.out_dir, "loss"))
        return
    if a.seg_compare:
        dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
        segmentation_comparison(a.dataset_dir, a.out_dir, dev)
        return
    if a.seg_individual:
        dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
        segmentation_individual(a.dataset_dir, a.out_dir, dev)
        return
    if not a.compare_only:
        dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
        segmentation_figures(a.dataset_dir, a.out_dir, dev)
        segmentation_comparison(a.dataset_dir, a.out_dir, dev)
    if not a.seg_only:
        comparison_figure(res, a.out_dir)
        os.makedirs(os.path.join(a.out_dir, "loss"), exist_ok=True)
        loss_figure(res, os.path.join(a.out_dir, "loss"))


if __name__ == "__main__":
    main()
