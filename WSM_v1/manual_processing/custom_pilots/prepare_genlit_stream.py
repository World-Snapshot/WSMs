#!/usr/bin/env python3
"""Memory-bounded GenLit converter for the IC-Light pilot.

Lighting condition = an IBL light-probe sphere: a chrome ball that reflects the
sample's REAL HDRI environment (rotated) plus a point-light highlight. One image
carries environment light (direction/color/space from the HDRI) + point-light
direction/intensity, and is approximately invertible (mirror sphere <->
environment; highlight <-> point light). This replaces the earlier flat
"direction as a solid color" encoding.

Output is self-describing named-key npz (string keys, not WSG numeric channels):
  "0-2"                     source RGB (delight = environment-only)
  "ICLight_Lighting"        IBL light-probe condition
  "ICLight_Relighted_RGB"   target (environment + point light)
"""
from pathlib import Path
import argparse, csv, io, json, math, os, random, urllib.request
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

try:
    import cv2
    def _read_hdr(p):
        img = cv2.imread(p, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        return img[..., ::-1].astype(np.float32)  # BGR->RGB
except Exception:
    import imageio.v2 as iio
    def _read_hdr(p):
        return np.asarray(iio.imread(p)).astype(np.float32)

HDRI_DIR = Path("data/custom_pilots/iclight_relighting_v3/hdris")
POINT_MAX = 120.0  # fixed scale for invertible point-intensity encoding

_HDRI_CACHE = {}
def load_hdri(name):
    # name like "scythian_tombs_puresky/scythian_tombs_puresky_2k.hdr"
    base = os.path.basename(name)
    if base not in _HDRI_CACHE:
        path = HDRI_DIR / base
        if not path.exists():
            # Fetch on demand from Poly Haven (the GenLit HDRIs are the free PH set).
            slug = base.replace("_2k.hdr", "")
            url = f"https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/{slug}_2k.hdr"
            HDRI_DIR.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".hdr.tmp")
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < 100000:
                tmp.unlink(missing_ok=True)
                raise FileNotFoundError(f"HDRI download too small/failed: {slug}")
            tmp.replace(path)
        _HDRI_CACHE[base] = _read_hdr(str(path))
    return _HDRI_CACHE[base]

def _sphere_NR(size):
    u = np.linspace(-1, 1, size); uu, vv = np.meshgrid(u, -u)
    r2 = uu**2 + vv**2; inside = r2 <= 1.0
    nz = np.sqrt(np.clip(1 - r2, 0, 1)); N = np.stack([uu, vv, nz], -1)
    V = np.array([0, 0, 1.0]); NdV = (N * V).sum(-1, keepdims=True); R = 2 * NdV * N - V
    return N, R, inside

_NRI = {}
def sphere_NR(size):
    if size not in _NRI:
        _NRI[size] = _sphere_NR(size)
    return _NRI[size]

def _dir_to_uv(d, rot_deg):
    a = math.radians(rot_deg)
    x = d[..., 0] * math.cos(a) + d[..., 2] * math.sin(a)
    z = -d[..., 0] * math.sin(a) + d[..., 2] * math.cos(a)
    y = d[..., 1]
    u = 0.5 + np.arctan2(x, z) / (2 * np.pi)
    v = 0.5 - np.arcsin(np.clip(y, -1, 1)) / np.pi
    return u, v

def render_ibl(hdri, rot, L, point_int, env_int, size=480, power=300, bg=0.1):
    """Chrome light-probe: mirror-reflect the HDRI (rotated) + a white point-light
    highlight. Reinhard tonemap + gamma into [0,255]. Approximately invertible:
    sphere<->environment, highlight position<->point direction, highlight
    magnitude<->point intensity (fixed POINT_MAX scale), env brightness<->env_int."""
    N, R, inside = sphere_NR(size)
    Hh, Ww = hdri.shape[:2]
    u, v = _dir_to_uv(R, rot)
    xi = np.clip((u * Ww).astype(int), 0, Ww - 1); yi = np.clip((v * Hh).astype(int), 0, Hh - 1)
    env = hdri[yi, xi] * float(env_int)
    L = np.asarray(L, float); L = L / (np.linalg.norm(L) + 1e-8)
    spec = np.clip((R * L).sum(-1), 0, 1) ** power
    point = (float(point_int) / POINT_MAX) * spec  # white point light, invertible scale
    total = env + point[..., None]
    total = total / (1.0 + total)                  # Reinhard tonemap
    total = np.clip(total, 0, 1) ** (1 / 2.2)      # gamma
    out = np.where(inside[..., None], total, np.full(3, bg))
    return (out * 255).astype(np.float32)

def image(value, size=480):
    with Image.open(io.BytesIO(value["bytes"])) as im:
        im = im.convert("RGB"); im.thumbnail((size, size), Image.Resampling.LANCZOS)
        out = Image.new("RGB", (size, size), (0, 0, 0)); out.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        return np.asarray(out, dtype=np.uint8)

def do_refill(a):
    """REFILL: add same-scene/same-direction, varying-intensity frames for scenes already used.

    GenLit ramps the point light (0 -> max, e.g. [0,30,60,90,100,120]) at a FIXED direction for
    the first frames, then plateaus at max while the direction drifts. The base run took one
    plateau frame per scene (intensity == max), so intensity never varied. Here we pull the
    RAMP frames (distinct sub-max intensities, direction fixed) for the exact same scenes, giving
    the model a clean intensity axis. Matched by seq_id (unique within a parquet)."""
    man = json.loads((a.refill_from / "manifest.json").read_text())
    want = {s["seq_id"] for s in man["samples"]}
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = []; count = 0
    pf = pq.ParquetFile(a.parquet)
    for batch in pf.iter_batches(batch_size=1, columns=["images", "metadata", "seq_id", "object_id"]):
        row = batch.to_pydict(); seq = row["seq_id"][0]
        if seq not in want: continue
        images = row["images"][0]; meta = row["metadata"][0] or {}
        traj = meta.get("point_light_trajectory") or []; ints = meta.get("point_light_intensity") or []
        envs = meta.get("env_light_intensity") or []
        hdri_meta = meta.get("hdri") or {}; hdri_name = hdri_meta.get("name", ""); hdri_rot = float(hdri_meta.get("rot", 0.0))
        try:
            hdri = load_hdri(hdri_name)
        except Exception as e:
            print(f"skip scene (hdri load failed {hdri_name}): {e}"); continue
        source = image(images[0])
        # Candidate pool = ALL distinct intensity LEVELS (first frame reaching each),
        # including 0 (env-only / point light off) and the max (full). GenLit renders
        # {0,30,60,90,100,120}, and all these first-occurrence frames share the same fixed
        # light direction (drift only starts after the ramp), so this is a clean intensity axis.
        seen = set(); cand = []
        for fi, v in enumerate(ints):
            v = float(v)
            if v not in seen and fi < len(images):
                seen.add(v); cand.append(fi)
        if not cand: continue
        # Randomly pick refill_count distinct levels PER SCENE (seeded by seq_id -> deterministic
        # & resumable). Random per-scene selection spreads coverage across every level over the
        # ~10k scenes, instead of pinning the same few levels everywhere.
        k = min(a.refill_count, len(cand))
        cand = sorted(random.Random(seq).sample(cand, k))
        for frame in cand:
            polar = np.asarray(traj[frame] if frame < len(traj) else [1, 0, 90], dtype=np.float32).reshape(-1)
            polar = np.pad(polar, (0, max(0, 3 - polar.size)))[:3]
            az, el = math.radians(float(polar[1])), math.radians(float(polar[2]))
            xyz = np.asarray([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], dtype=np.float32)
            pi = float(ints[frame]); ei = float(envs[frame]) if frame < len(envs) else 1.0
            probe = render_ibl(hdri, hdri_rot, xyz, pi, ei, size=source.shape[0])
            sid = f"genlit_refill_{a.offset + count:07d}"
            np.savez_compressed(
                a.output / f"{sid}.wsg.npz",
                **{"0-2": source.astype(np.uint8),
                   "ICLight_Lighting": np.clip(probe, 0, 255).astype(np.uint8),
                   "ICLight_Relighted_RGB": image(images[frame]).astype(np.uint8)},
            )
            manifest.append({"sample_id": str(a.output / f"{sid}.wsg.npz"), "seq_id": seq,
                             "object_id": row["object_id"][0], "frame": frame, "light_direction_xyz": xyz.tolist(),
                             "point_light_intensity": pi, "environment_light_intensity": ei,
                             "hdri_name": hdri_name, "hdri_rot": hdri_rot, "refill": True})
            count += 1
    (a.output / "manifest.json").write_text(json.dumps(
        {"task": "iclight_relighting", "control_encoding": "ibl_lightprobe_sphere", "mode": "refill", "samples": manifest}, indent=2))
    print(f"[refill] {a.parquet.name}: matched {len(want)} scenes by seq_id, wrote {count} extra ramp frames -> {a.output}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=390)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max-per-file", type=int, default=0,
                    help="If >0, uniformly sample this many scenes per parquet (one mid-trajectory frame each) "
                         "for diverse object/lighting coverage; 0 = all frames up to --limit.")
    ap.add_argument("--refill-from", type=Path, default=None,
                    help="REFILL MODE. Path to an existing named/ dir (with manifest.json). For exactly the "
                         "scenes already used in that dir (matched by seq_id within --parquet), emit EXTRA "
                         "frames drawn from the intensity RAMP -- fixed light direction, varying intensity -- "
                         "so the model learns same-scene/same-direction intensity control. Source (0-2) stays "
                         "frame 0 (env-only), exactly as the base run. Writes to --output; sids are "
                         "genlit_refill_* so they never collide with the base data.")
    ap.add_argument("--refill-count", type=int, default=2,
                    help="Refill mode: how many distinct intensity LEVELS to RANDOMLY sample per scene "
                         "from the full set GenLit offers {0,30,60,90,100,120}. Seeded by seq_id so it is "
                         "deterministic/resumable; across scenes it covers all levels.")
    a = ap.parse_args()
    if a.refill_from is not None:
        do_refill(a); return
    a.output.mkdir(parents=True, exist_ok=True); manifest = []; captions = []; count = 0
    pf = pq.ParquetFile(a.parquet)
    nscene = pf.metadata.num_rows
    if a.max_per_file > 0:
        k = min(a.max_per_file, nscene)
        chosen = set(int(x) for x in np.linspace(0, nscene - 1, k))
    else:
        chosen = None
    for si, batch in enumerate(pf.iter_batches(batch_size=1, columns=["images", "metadata", "seq_id", "object_id"])):
        if count >= a.limit: break
        if chosen is not None and si not in chosen: continue
        row = batch.to_pydict(); images = row["images"][0]; meta = row["metadata"][0] or {}
        traj = meta.get("point_light_trajectory") or []; ints = meta.get("point_light_intensity") or []
        envs = meta.get("env_light_intensity") or []
        hdri_meta = meta.get("hdri") or {}; hdri_name = hdri_meta.get("name", ""); hdri_rot = float(hdri_meta.get("rot", 0.0))
        source = image(images[0])
        try:
            hdri = load_hdri(hdri_name)
        except Exception as e:
            print(f"skip scene (hdri load failed {hdri_name}): {e}"); continue
        frame_list = [1 + (len(images) - 1) // 2] if chosen is not None else list(range(1, len(images)))
        for frame in frame_list:
            value = images[frame]
            if count >= a.limit: break
            polar = np.asarray(traj[frame] if frame < len(traj) else [1, 0, 90], dtype=np.float32).reshape(-1)
            polar = np.pad(polar, (0, max(0, 3 - polar.size)))[:3]
            az, el = math.radians(float(polar[1])), math.radians(float(polar[2]))
            xyz = np.asarray([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], dtype=np.float32)
            pi = float(ints[frame]) if frame < len(ints) else 1.0
            ei = float(envs[frame]) if frame < len(envs) else 1.0
            probe = render_ibl(hdri, hdri_rot, xyz, pi, ei, size=source.shape[0])
            sid = f"genlit_{a.offset + count:06d}"
            np.savez_compressed(
                a.output / f"{sid}.wsg.npz",
                # Store as uint8 (0-255 image data): the dataset does frame/127.5-1
                # regardless of dtype, so this is loss-equivalent to float32 but ~4x
                # less data through zlib -> the real per-parquet bottleneck. Pre-existing
                # float32 npz stay loadable (same /127.5-1 path).
                **{"0-2": source.astype(np.uint8),
                   "ICLight_Lighting": np.clip(probe, 0, 255).astype(np.uint8),
                   "ICLight_Relighted_RGB": image(value).astype(np.uint8)},
            )
            manifest.append({"sample_id": str(a.output / f"{sid}.wsg.npz"), "seq_id": row["seq_id"][0],
                             "object_id": row["object_id"][0], "frame": frame, "light_direction_xyz": xyz.tolist(),
                             "point_light_intensity": pi, "environment_light_intensity": ei,
                             "hdri_name": hdri_name, "hdri_rot": hdri_rot})
            captions.append({"name": sid, "caption": f"Relight the object using the supplied light direction. Point-light intensity: {pi:.6g}. Environment-light intensity: {ei:.6g}. Preserve the object geometry and material appearance."})
            count += 1
        if count >= a.limit: break
    (a.output / "manifest.json").write_text(json.dumps({"task": "iclight_relighting", "control_encoding": "ibl_lightprobe_sphere", "samples": manifest}, indent=2))
    with (a.output / "captions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "caption"]); w.writeheader(); w.writerows(captions)
    print(f"prepared {count} IBL samples in {a.output}")

if __name__ == "__main__":
    main()
