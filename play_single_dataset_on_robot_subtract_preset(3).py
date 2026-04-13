import time
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

import Sophia_control  # uses Sophia_control.call_remote(index=..., value=[x,y,z])


def normalize_motion_poses(poses: np.ndarray) -> np.ndarray:
    """
    Normalize motion poses to shape (T, J, 3).

    Supported input shapes:
      - (T, J*3)
      - (J*3,)          -> treated as one frame
      - (J, 3)          -> treated as one frame
      - (T, J, 3)
      - (1, J*3)
    """
    poses = np.asarray(poses, dtype=np.float32)

    if poses.ndim == 1:
        # (J*3,)
        if poses.shape[0] % 3 != 0:
            raise ValueError(f"1D poses length must be multiple of 3, got {poses.shape}")
        J = poses.shape[0] // 3
        return poses.reshape(1, J, 3)

    if poses.ndim == 2:
        # Could be (T, J*3) or (J, 3)
        if poses.shape[1] == 3:
            # (J, 3) -> one frame
            return poses[None, :, :]
        if poses.shape[1] % 3 == 0:
            # (T, J*3)
            J = poses.shape[1] // 3
            return poses.reshape(poses.shape[0], J, 3)
        raise ValueError(f"Unsupported 2D poses shape for motion: {poses.shape}")

    if poses.ndim == 3:
        # (T, J, 3)
        if poses.shape[2] != 3:
            raise ValueError(f"3D poses last dim must be 3, got {poses.shape}")
        return poses

    raise ValueError(f"Unsupported motion poses shape: {poses.shape}")


def normalize_preset_pose(poses: np.ndarray) -> np.ndarray:
    """
    Normalize preset poses to shape (J, 3).

    Supported input shapes:
      - (J*3,)
      - (1, J*3)
      - (T, J*3)   -> use first frame
      - (J, 3)
      - (1, J, 3)
      - (T, J, 3)  -> use first frame
    """
    poses = np.asarray(poses, dtype=np.float32)

    if poses.ndim == 1:
        if poses.shape[0] % 3 != 0:
            raise ValueError(f"Preset 1D pose length must be multiple of 3, got {poses.shape}")
        return poses.reshape(-1, 3)

    if poses.ndim == 2:
        if poses.shape[1] == 3:
            # (J, 3)
            return poses.copy()
        if poses.shape[1] % 3 == 0:
            # (T, J*3), use first frame
            if poses.shape[0] < 1:
                raise ValueError(f"Preset npz has empty poses: shape={poses.shape}")
            return poses[0].reshape(-1, 3)
        raise ValueError(f"Unsupported preset 2D shape: {poses.shape}")

    if poses.ndim == 3:
        if poses.shape[2] != 3:
            raise ValueError(f"Preset 3D last dim must be 3, got {poses.shape}")
        if poses.shape[0] < 1:
            raise ValueError(f"Preset npz has empty poses: shape={poses.shape}")
        # (T, J, 3), use first frame
        return poses[0].copy()

    raise ValueError(f"Unsupported preset poses shape: {poses.shape}")


def load_pose_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    if "poses" not in d.files:
        raise ValueError(f"Missing 'poses' in {npz_path}, keys={list(d.files)}")

    poses = normalize_motion_poses(d["poses"].astype(np.float32))
    fps = int(d["mocap_frame_rate"]) if "mocap_frame_rate" in d.files else 30
    return poses, fps


# -----------------------------
# Mapping: SMPL-X joint index -> motor index
# -----------------------------
ALLOWED = [16, 17, 18, 19, 20, 21]  # fingers omitted for now


# -----------------------------
# Extractors / pose utilities
# -----------------------------

PRESET_AFFECTED_INDICES = {16, 17}


def subtract_axisangle_offset(pose_aa: np.ndarray, offset_aa: np.ndarray) -> np.ndarray:
    """
    Compute pose' = offset^{-1} * pose in rotation space, then return as rotvec.
    This is the correct way to 'subtract' an axis-angle offset.
    """
    r_pose = R.from_rotvec(np.asarray(pose_aa, dtype=np.float64))
    r_offset = R.from_rotvec(np.asarray(offset_aa, dtype=np.float64))
    r_new = r_offset.inv() * r_pose
    return r_new.as_rotvec().astype(np.float32)


def load_preset_npz(preset_path: str, expected_joints: int | None = None) -> np.ndarray:
    """
    Load preset npz and extract one reference pose for offsets.

    Returns:
      preset_aa_all: (J, 3)
    """
    d = np.load(preset_path, allow_pickle=True)
    if "poses" not in d.files:
        raise ValueError(f"Missing 'poses' in preset npz {preset_path}, keys={list(d.files)}")

    preset_aa_all = normalize_preset_pose(d["poses"].astype(np.float32))

    j = preset_aa_all.shape[0]
    if expected_joints is not None and j != expected_joints:
        raise ValueError(
            f"Preset joint count mismatch: preset has {j}, target npz has {expected_joints}"
        )

    return preset_aa_all.copy()


def apply_preset_to_joint(motor_idx: int, aa: np.ndarray, preset_aa_all: np.ndarray | None) -> np.ndarray:
    if preset_aa_all is None:
        return aa
    if motor_idx >= preset_aa_all.shape[0]:
        return aa
    offset_aa = preset_aa_all[motor_idx]
    return subtract_axisangle_offset(aa, offset_aa)


# -----------------------------
# Playback
# -----------------------------

def play(npz_path: str, speed: float, dry_run: bool, max_fps: float | None, preset_path: str | None):
    poses, fps = load_pose_npz(npz_path)   # poses is always (T, J, 3)
    T, J, _ = poses.shape

    preset_aa_all = None
    if preset_path is not None:
        preset_aa_all = load_preset_npz(preset_path, expected_joints=J)

    bad = [i for i in ALLOWED if i >= J]
    if bad:
        print(f"[WARN] Some allowed motor indices are out of bounds for this dataset (J={J} joints): {bad}")

    base_dt = 1.0 / fps
    dt = base_dt / max(speed, 1e-6)

    # optional clamp on command rate
    if max_fps is not None and max_fps > 0:
        dt = max(dt, 1.0 / max_fps)

    print(f"[INFO] Loaded motion: {npz_path}")
    print(f"[INFO] Motion poses normalized shape: {poses.shape}")
    if preset_path is not None:
        print(f"[INFO] Loaded preset: {preset_path}")
        print(f"[INFO] Preset poses normalized shape: {preset_aa_all.shape}")
    print(f"[INFO] frames={T}, joints={J}, fps={fps}, speed={speed}, send_dt={dt:.4f}s (~{1/dt:.1f} Hz)")

    t0 = time.perf_counter()
    for k in range(T):
        aa_all = poses[k]  # (J, 3)

        for idx in ALLOWED:
            if idx >= J:
                continue
            aa = aa_all[idx].copy()
            if idx in PRESET_AFFECTED_INDICES:
                aa = apply_preset_to_joint(idx, aa, preset_aa_all)
            value = [float(aa[0]), float(aa[1]), float(aa[2])]

            if idx == 19:
                print("------right elbow pitch original-------")
                print(float(aa[1])*57.29578)
                print("-----------------")

            if dry_run:
                if k % 30 == 0:
                    print(f"[DRY] frame={k} index={idx} value={value}")
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
    ap.add_argument("--npz", required=True, help="Motion npz to play")
    ap.add_argument(
        "--preset",
        default=None,
        help="Preset/offset npz. Its first pose is treated as the axis-angle offset to subtract."
    )
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-fps", type=float, default=0.0, help="Clamp send rate. 0 = no clamp.")
    args = ap.parse_args()

    max_fps = None if args.max_fps <= 0 else args.max_fps
    play(args.npz, speed=args.speed, dry_run=args.dry_run, max_fps=max_fps, preset_path=args.preset)


if __name__ == "__main__":
    main()