import time
import argparse
import numpy as np

import Sophia_control  # uses Sophia_control.call_remote(index=..., value=[x,y,z])

def load_pose_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    if "poses" not in d.files:
        raise ValueError(f"Missing 'poses' in {npz_path}, keys={list(d.files)}")

    poses = d["poses"].astype(np.float32)
    fps = int(d["mocap_frame_rate"]) if "mocap_frame_rate" in d.files else 30
    return poses, fps


# -----------------------------
# Mapping: SMPL-X joint index -> motor index
# -----------------------------
ALLOWED = [16,17,18,19,20,21] # ,25,28,31,34,37,40,43,46,49,52 these are fingers which set to 0


# Optional: if your robot expects an "A-pose offset" like in web visualizer, you can apply it here.
# In your UI file, VISUAL_OFFSET is used for visualization only. For robot playback, you may or may not want it.
APPLY_A_POSE_OFFSET = False
A_POSE_OFFSET = {
    16: np.array([0.0, 0.0, -0.78], dtype=np.float32),  # from your UI file's to_axisangle((0, -0.78),16)
    17: np.array([0.0, 0.0,  0.78], dtype=np.float32),
}

# -----------------------------
# Extractors
# -----------------------------

def get_body_axisangle(pose165: np.ndarray, body_joint_idx: int) -> np.ndarray:
    body = pose165[3:66].reshape(21, 3)
    return body[body_joint_idx].copy()

def get_hand_axisangle(pose165: np.ndarray, hand: str, hand_joint_idx: int) -> np.ndarray:
    if hand == "lhand":
        h = pose165[75:120].reshape(15, 3)
    elif hand == "rhand":
        h = pose165[120:165].reshape(15, 3)
    else:
        raise ValueError(hand)
    return h[hand_joint_idx].copy()


def maybe_apply_offset(motor_idx: int, aa: np.ndarray) -> np.ndarray:
    if APPLY_A_POSE_OFFSET and motor_idx in A_POSE_OFFSET:
        return aa + A_POSE_OFFSET[motor_idx]
    return aa


# -----------------------------
# Playback
# -----------------------------

def play(npz_path: str, speed: float, dry_run: bool, max_fps: float | None):
    poses, fps = load_pose_npz(npz_path)
    T = poses.shape[0]
    if poses.shape[1] % 3 != 0:
        raise ValueError(f"poses second dim must be multiple of 3, got {poses.shape}")
    J = poses.shape[1] // 3
    bad = [i for i in ALLOWED if i >= J]
    if bad:
        print(f"[WARN] Some allowed motor indices are out of bounds for this dataset (J={J} joints): {bad}")
    base_dt = 1.0 / fps
    dt = base_dt / max(speed, 1e-6)

    # optional clamp on command rate
    if max_fps is not None and max_fps > 0:
        dt = max(dt, 1.0 / max_fps)

    print(f"[INFO] Loaded: {npz_path}")
    print(f"[INFO] frames={T}, fps={fps}, speed={speed}, send_dt={dt:.4f}s (~{1/dt:.1f} Hz)")

    t0 = time.perf_counter()
    for k in range(T):
        pose = poses[k]

        pose = poses[k]
        aa_all = pose.reshape(J, 3) # axis-angle for all joints
        for idx in ALLOWED:
            if idx >= J:
                continue
            aa = aa_all[idx].copy()
            aa = maybe_apply_offset(idx, aa_all[idx])
            value = [float(aa[0]), float(aa[1]), float(aa[2])]
            if dry_run:
                if k % 30 == 0:
                    print(f"[DRY] frame = {k} index = {idx} value = {value}")
            else:
                Sophia_control.call_remote(index=idx, value=value)
        # pacing
        target = t0 + (k + 1) * dt
        now = time.perf_counter()
        sleep_time = target - now
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("[INFO] Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-fps", type=float, default=0.0, help="Clamp send rate. 0 = no clamp.")
    args = ap.parse_args()

    max_fps = None if args.max_fps <= 0 else args.max_fps
    play(args.npz, speed=args.speed, dry_run=args.dry_run, max_fps=max_fps)


if __name__ == "__main__":
    main()