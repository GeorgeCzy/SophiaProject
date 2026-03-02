import time
import threading
import numpy as np
import torch

import smplx
import trimesh
import pyrender

import dearpygui.dearpygui as dpg


# ----------------------------
# Config
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = r"./SMPLX_MODEL_DIR/smplx"   # <-- 改成你的路径
GENDER = "neutral"
BATCH_SIZE = 1

# Slider ranges (radians) for axis-angle components
AA_MIN, AA_MAX = -1.0, 1.0

# SMPL-X sizes
# body_pose: 21 joints * 3 (pelvis excluded; root_orient separate)
# left_hand_pose/right_hand_pose: 15 joints * 3 each
BODY_POSE_DIM = 21 * 3
HAND_POSE_DIM = 15 * 3


# ----------------------------
# Joint mapping (SMPL-X semantic)
# ----------------------------
# SMPL-X body 22: pelvis + 21 joints
# body_pose corresponds to joints:
# [left_hip, right_hip, spine1, left_knee, right_knee, spine2,
#  left_ankle, right_ankle, spine3, left_foot, right_foot, neck,
#  left_collar, right_collar, head, left_shoulder, right_shoulder,
#  left_elbow, right_elbow, left_wrist, right_wrist]
#
# We'll CONTROL only:
# spine1, spine2, spine3, neck, head,
# left_collar, right_collar,
# left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist
#
# We'll FREEZE lower-body joints:
# left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, left_foot, right_foot

BODY_JOINT_NAMES_IN_BODY_POSE = [
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist"
]
NAME_TO_BODYPOSE_JIDX = {n: i for i, n in enumerate(BODY_JOINT_NAMES_IN_BODY_POSE)}

CONTROL_BODY = [
    "spine1", "spine2", "spine3", "neck", "head",
    "left_collar", "right_collar",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]
FREEZE_BODY = [n for n in BODY_JOINT_NAMES_IN_BODY_POSE if n not in CONTROL_BODY]

# Hand joint labels (15) in MANO order (common convention)
HAND_JOINTS_15 = [
    "index1","index2","index3",
    "middle1","middle2","middle3",
    "pinky1","pinky2","pinky3",
    "ring1","ring2","ring3",
    "thumb1","thumb2","thumb3",
]


# ----------------------------
# State (shared between GUI thread and render thread)
# ----------------------------
state = {
    "root_orient": np.zeros(3, dtype=np.float32),          # global orientation
    "body_pose": np.zeros(BODY_POSE_DIM, dtype=np.float32),
    "lhand_pose": np.zeros(HAND_POSE_DIM, dtype=np.float32),
    "rhand_pose": np.zeros(HAND_POSE_DIM, dtype=np.float32),
    "quit": False,
}


def set_body_joint_rotvec(joint_name: str, xyz: np.ndarray):
    j = NAME_TO_BODYPOSE_JIDX[joint_name]
    state["body_pose"][j*3:(j+1)*3] = xyz.astype(np.float32)


def get_body_joint_rotvec(joint_name: str) -> np.ndarray:
    j = NAME_TO_BODYPOSE_JIDX[joint_name]
    return state["body_pose"][j*3:(j+1)*3].copy()


def set_hand_joint_rotvec(side: str, joint_idx_0_14: int, xyz: np.ndarray):
    key = "lhand_pose" if side == "L" else "rhand_pose"
    state[key][joint_idx_0_14*3:(joint_idx_0_14+1)*3] = xyz.astype(np.float32)


def get_hand_joint_rotvec(side: str, joint_idx_0_14: int) -> np.ndarray:
    key = "lhand_pose" if side == "L" else "rhand_pose"
    return state[key][joint_idx_0_14*3:(joint_idx_0_14+1)*3].copy()


# Freeze lower body to zero
for n in FREEZE_BODY:
    set_body_joint_rotvec(n, np.zeros(3, dtype=np.float32))


# ----------------------------
# SMPL-X Model
# ----------------------------
model = smplx.create(
    model_path=MODEL_DIR,
    model_type="smplx",
    gender=GENDER,
    use_pca=False,              # IMPORTANT: use full hand pose (45 dims) not PCA
    num_pca_comps=45,
    batch_size=BATCH_SIZE,
).to(DEVICE)
model.eval()


# ----------------------------
# Render Thread
# ----------------------------
def render_loop():
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.3, 0.3, 0.3])
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = 2.5
    scene.add(cam, pose=cam_pose)

    light = pyrender.DirectionalLight(intensity=2.0)
    scene.add(light, pose=cam_pose)

    mesh_node = None
    viewer = pyrender.Viewer(
        scene,
        use_raymond_lighting=False,
        run_in_thread=True,
        viewport_size=(960, 960),
    )

    last_update = 0.0
    while not state["quit"] and viewer.is_active:
        now = time.time()
        if now - last_update < 1.0 / 30.0:
            time.sleep(0.001)
            continue
        last_update = now

        # read state -> tensors
        root_orient = torch.tensor(state["root_orient"][None, :], device=DEVICE)
        body_pose = torch.tensor(state["body_pose"][None, :], device=DEVICE)
        lhand = torch.tensor(state["lhand_pose"][None, :], device=DEVICE)
        rhand = torch.tensor(state["rhand_pose"][None, :], device=DEVICE)

        with torch.no_grad():
            out = model(
                root_orient=root_orient,
                body_pose=body_pose,
                left_hand_pose=lhand,
                right_hand_pose=rhand,
                jaw_pose=None,           # no face
                leye_pose=None,
                reye_pose=None,
                expression=None,
            )
            verts = out.vertices[0].detach().cpu().numpy()

        tri = model.faces
        tm = trimesh.Trimesh(vertices=verts, faces=tri, process=False)
        pm = pyrender.Mesh.from_trimesh(tm, smooth=True)

        # update scene
        if mesh_node is not None:
            scene.remove_node(mesh_node)
        mesh_node = scene.add(pm)

    viewer.close_external()
    state["quit"] = True


# ----------------------------
# GUI
# ----------------------------
def add_rotvec_sliders(prefix: str, get_fn, set_fn):
    # prefix used to disambiguate widget tags
    def make_axis(axis_i: int, label: str):
        tag = f"{prefix}_{label}"
        init = float(get_fn()[axis_i])

        def on_change(sender, app_data):
            v = float(app_data)
            cur = get_fn()
            cur[axis_i] = v
            set_fn(cur)

        dpg.add_slider_float(
            label=label,
            tag=tag,
            default_value=init,
            min_value=AA_MIN,
            max_value=AA_MAX,
            callback=on_change,
            width=220,
        )

    make_axis(0, "rx")
    make_axis(1, "ry")
    make_axis(2, "rz")


def gui_main():
    dpg.create_context()
    dpg.create_viewport(title="SMPL-X UpperBody (Torso + Arms + Hands) GUI", width=1200, height=900)

    with dpg.window(label="Controls", width=560, height=860):
        dpg.add_text("Axis-angle (rotvec) sliders (radians).")
        dpg.add_separator()

        # Root orient (global)
        with dpg.collapsing_header(label="Global Root Orient (pelvis root_orient)", default_open=False):
            def get_root(): return state["root_orient"].copy()
            def set_root(v): state["root_orient"] = v.astype(np.float32)
            add_rotvec_sliders("root", get_root, set_root)

        dpg.add_separator()
        dpg.add_text("Torso + Arms (Body Pose)")
        dpg.add_separator()

        # Body joints
        for jn in CONTROL_BODY:
            with dpg.collapsing_header(label=jn, default_open=False):
                def make_getter(name=jn):
                    return lambda: get_body_joint_rotvec(name)
                def make_setter(name=jn):
                    return lambda v: set_body_joint_rotvec(name, v)

                add_rotvec_sliders(f"body_{jn}", make_getter(), make_setter())

        dpg.add_separator()
        dpg.add_text("Hands (15 joints each side)")
        dpg.add_separator()

        # Left hand
        with dpg.collapsing_header(label="Left Hand", default_open=False):
            for i, hname in enumerate(HAND_JOINTS_15):
                with dpg.collapsing_header(label=f"L_{hname}", default_open=False):
                    def make_getter(idx=i):
                        return lambda: get_hand_joint_rotvec("L", idx)
                    def make_setter(idx=i):
                        return lambda v: set_hand_joint_rotvec("L", idx, v)
                    add_rotvec_sliders(f"lh_{i}", make_getter(), make_setter())

        # Right hand
        with dpg.collapsing_header(label="Right Hand", default_open=False):
            for i, hname in enumerate(HAND_JOINTS_15):
                with dpg.collapsing_header(label=f"R_{hname}", default_open=False):
                    def make_getter(idx=i):
                        return lambda: get_hand_joint_rotvec("R", idx)
                    def make_setter(idx=i):
                        return lambda v: set_hand_joint_rotvec("R", idx, v)
                    add_rotvec_sliders(f"rh_{i}", make_getter(), make_setter())

        dpg.add_separator()
        def reset_all():
            state["root_orient"][:] = 0
            state["body_pose"][:] = 0
            state["lhand_pose"][:] = 0
            state["rhand_pose"][:] = 0
            # re-freeze lower body
            for n in FREEZE_BODY:
                set_body_joint_rotvec(n, np.zeros(3, dtype=np.float32))
        dpg.add_button(label="Reset All (Keep Legs Frozen)", callback=lambda: reset_all())
        dpg.add_same_line()
        dpg.add_button(label="Quit", callback=lambda: setattr(state, "quit", True))

    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running() and not state["quit"]:
        dpg.render_dearpygui_frame()

    state["quit"] = True
    dpg.destroy_context()


if __name__ == "__main__":
    t = threading.Thread(target=render_loop, daemon=True)
    t.start()
    gui_main()
    state["quit"] = True
    t.join(timeout=1.0)