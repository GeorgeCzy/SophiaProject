import numpy as np
import torch
import smplx
import trimesh
import pyrender
import imageio
import os

NPZ_PATH = r"./dataset/12_zhao_2_100_100.npz"
MODEL_DIR = r"./dataset"          # 目录，不是 ./dataset/SMPLX_MALE.npz
OUT_MP4 = "smpl.mp4"
RES = 1024

def main():
    d = np.load(NPZ_PATH, allow_pickle=True)

    poses = d["poses"].astype(np.float32)          # (T,165)
    trans = d["trans"].astype(np.float32)          # (T,3)
    betas = d["betas"].astype(np.float32)          # (300,)
    expr  = d["expressions"].astype(np.float32)    # (T,100)
    fps   = int(d["mocap_frame_rate"]) if "mocap_frame_rate" in d.files else 30
    gender = str(d["gender"]).lower() if "gender" in d.files else "neutral"
    if gender not in ("male","female","neutral"):
        gender = "neutral"

    device = "cpu"
    model = smplx.create(
        model_path=MODEL_DIR,
        model_type="smplx",
        gender=gender,
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=100,
        batch_size=1
    ).to(device)

    faces = model.faces.astype(np.int32)

    scene = pyrender.Scene(bg_color=[255,255,255,255], ambient_light=[0.35,0.35,0.35])
    cam = pyrender.PerspectiveCamera(yfov=np.pi/3.2)
    cam_pose = np.eye(4)
    cam_pose[2,3] = 2.6
    scene.add(cam, pose=cam_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.8)
    scene.add(light, pose=cam_pose)

    r = pyrender.OffscreenRenderer(viewport_width=RES, viewport_height=RES)
    writer = imageio.get_writer(OUT_MP4, fps=fps, codec="libx264", quality=8)

    poses_t = torch.from_numpy(poses).float().to(device)
    trans_t = torch.from_numpy(trans).float().to(device)
    betas_t = torch.from_numpy(betas[:10]).float().to(device).unsqueeze(0)
    expr_t  = torch.from_numpy(expr).float().to(device)

    mesh_node = None
    try:
        with torch.no_grad():
            for t in range(poses.shape[0]):
                p = poses_t[t]
                out = model(
                    global_orient=p[0:3].unsqueeze(0),
                    body_pose=p[3:66].unsqueeze(0),
                    jaw_pose=p[66:69].unsqueeze(0),
                    leye_pose=p[69:72].unsqueeze(0),
                    reye_pose=p[72:75].unsqueeze(0),
                    left_hand_pose=p[75:120].unsqueeze(0),
                    right_hand_pose=p[120:165].unsqueeze(0),
                    betas=betas_t,
                    expression=expr_t[t:t+1],
                    transl=trans_t[t:t+1],
                )
                verts = out.vertices[0].cpu().numpy()
                verts = verts - verts.mean(axis=0, keepdims=True)

                tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                pm = pyrender.Mesh.from_trimesh(tm, smooth=True)

                if mesh_node is not None:
                    scene.remove_node(mesh_node)
                mesh_node = scene.add(pm)

                color, _ = r.render(scene)
                writer.append_data(color)
    finally:
        writer.close()
        r.delete()

    print(f"Saved {OUT_MP4} (fps={fps}, frames={poses.shape[0]})")

if __name__ == "__main__":
    main()
