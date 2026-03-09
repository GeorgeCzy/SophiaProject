"""SMPL model visualizer

Visualizer for SMPL human body models. Requires a .npz model file.

See here for download instructions:
    https://github.com/vchoutas/smplx?tab=readme-ov-file#downloading-the-model
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import tyro

import viser
import viser.transforms as tf

import Sophia_control
from scipy.spatial.transform import Rotation as R

import math
import threading

####################################################################################################
# ‼️  UI-RELATED LOGIC OVERVIEW                                                                    #
# The visualizer has two layers of UI:                                                             #
#   1. *Viewer widgets* shown in the browser (tabs, sliders, colour picker, check-boxes, etc.).     #
#      They are created **inside the make_gui_elements(..) function**.                             #
#   2. *Scene gizmos* (TransformControls) that appear as draggable handles on top of joints.        #
#      They are also spawned in make_gui_elements(..) and synchronised through callbacks.           #
#                                                                                                  #
#  ────────────────────────────────────────────────────────────────────────────────────────────────  #
#  Where to look:                                                                                  #
#    • make_gui_elements(..)  ← builds everything UI-related.                                       #
#    • main(..)                 ← calls make_gui_elements and drives updates every frame.           #
####################################################################################################

def to_axisangle(val: tuple[float, ...] ,index = 0) -> np.ndarray:
    if isinstance(val, (int, float)):
        if index in (25,26,27,28,29,30):
            return np.array([0.0, 0.0, float(val)], dtype=np.float32)
        if index in (31,32,33):
            return np.array([float(val), 0.0, float(val)], dtype=np.float32)
        if index in (34,35,36):
            return np.array([float(val)*0.3, 0.0, float(val)], dtype=np.float32)
        if index in (20,21):
            return np.array([float(val), 0.0, 0.0], dtype=np.float32)
        if index in (38,39):
            return np.array([float(val), 0.2 * float(val),   -float(val)], dtype=np.float32)
        if index in (40,41,42,43,44,45):
            return np.array([0.0, 0.0, -float(val)], dtype=np.float32)
        if index in (46,47,48):
            return np.array([float(val), 0.0, -float(val)], dtype=np.float32)
        if index in (49,50,51):
            return np.array([float(val)*0.3, 0.0, -float(val)], dtype=np.float32)
        if index in (53,54):
            return np.array([float(val), 0.2 * float(val),   -float(val)], dtype=np.float32)
            # dk, copied from 38,39. wait to be checked
        return np.array([0.0, float(val), 0.0], dtype=np.float32)
    
    if len(val) == 2:
      if index == 16:
          x,z = val
          y_linked = x        
          return np.array([x, 0.2 * y_linked, z], dtype=np.float32)

      if index == 17:
          x,z = val
          y_linked = x        
          return np.array([ x, -0.2 * y_linked, z], dtype=np.float32)
      
      if index == 18:
          x,y=val
          z_linked = -x
          return np.array([0.1 *x, y, z_linked], dtype=np.float32)
      
      if index == 19:
          x,y=val
          z_linked = x
          return np.array([x, y, z_linked], dtype=np.float32)
      
      if index == 37:
          x,y = val
          z_linked = y
          return np.array([x, y, z_linked], dtype=np.float32)
      
      if index == 52:
          x,y = val
          z_linked = y
          return np.array([x, y, z_linked], dtype=np.float32)
      

    return np.asarray(val, dtype=np.float32)


@dataclass(frozen=True)
class SmplOutputs:
    vertices: np.ndarray
    faces: np.ndarray
    T_world_joint: np.ndarray  # (num_joints, 4, 4)
    T_parent_joint: np.ndarray  # (num_joints, 4, 4)


class SmplHelper:
    """Helper for models in the SMPL family, implemented in numpy."""

    def __init__(self, model_path: Path) -> None:
        assert model_path.suffix.lower() == ".npz", "Model should be an .npz file!"
        body_dict = dict(**np.load(model_path, allow_pickle=True))

        self.J_regressor = body_dict["J_regressor"]
        self.weights = body_dict["weights"]
        self.v_template = body_dict["v_template"]
        self.posedirs = body_dict["posedirs"]
        self.shapedirs = body_dict["shapedirs"]
        self.faces = body_dict["f"]

        self.num_joints: int = self.weights.shape[-1]
        self.num_betas: int = self.shapedirs.shape[-1]
        self.parent_idx: np.ndarray = body_dict["kintree_table"][0]

    def get_outputs(self, betas: np.ndarray, joint_rotmats: np.ndarray) -> SmplOutputs:
        """Run the SMPL forward pass and return posed mesh & FK transforms."""
        # Shape blend-shapes
        v_tpose = self.v_template + np.einsum("vxb,b->vx", self.shapedirs, betas)
        j_tpose = np.einsum("jv,vx->jx", self.J_regressor, v_tpose)

        # Build local joint transforms (SE(3))
        T_parent_joint = np.zeros((self.num_joints, 4, 4)) + np.eye(4)
        T_parent_joint[:, :3, :3] = joint_rotmats
        T_parent_joint[0, :3, 3] = j_tpose[0]
        T_parent_joint[1:, :3, 3] = j_tpose[1:] - j_tpose[self.parent_idx[1:]]

        # Forward kinematics
        T_world_joint = T_parent_joint.copy()
        for i in range(1, self.num_joints):
            T_world_joint[i] = T_world_joint[self.parent_idx[i]] @ T_parent_joint[i]

        # Linear blend skinning (LBS)
        pose_delta = (joint_rotmats[1:, ...] - np.eye(3)).flatten()
        v_blend = v_tpose + np.einsum("byn,n->by", self.posedirs, pose_delta)
        v_delta = np.ones((v_blend.shape[0], self.num_joints, 4))
        v_delta[:, :, :3] = v_blend[:, None, :] - j_tpose[None, :, :]
        v_posed = np.einsum(
            "jxy,vj,vjy->vx", T_world_joint[:, :3, :], self.weights, v_delta
        )
        return SmplOutputs(v_posed, self.faces, T_world_joint, T_parent_joint)


########################################
#      APPLICATION ENTRY-POINT (main)  #
########################################



def deg(x):
    return x * math.pi / 180.0

VISUAL_OFFSET = {
    # web-pose startup for default motor = 0
    # 16: to_axisangle((0.0, -1.32), 16), # left shoulder roll
    # 18: to_axisangle((0.0, -0.96), 18), # left elbow pitch
    # 17: to_axisangle((0.0, 1.32), 17), # right shoulder roll
    # 19: to_axisangle((0.0, 0.96), 19), # right elbow pitch

    #-----------
    
    # web-pose startup for A-pose
    16: to_axisangle((0.0, -0.78), 16), # left shoulder roll
    17: to_axisangle((0.0, 0.78), 17), # right shoulder roll

}

# ---------------------------------------------------------------------------------
# Preset poses (values are in *GUI space*, i.e. the same numbers you type into sliders)
# ---------------------------------------------------------------------------------
PRESET_POSES = {
    # Example: shoulder yaw test (radians)
    "ShoulderYaw+1.5": {
        18: (1.5, 0.0),  # Left: (ShoulderYaw, ElbowPitch)
        19: (1.5, 0.0),  # Right: (ShoulderYaw, ElbowPitch)
    },
    # Example: simple A-ish pose tweak + a bit of elbow yaw
    "APoseTest": {
        16: (0.0, -0.30),  # Left shoulder: (Pitch, Roll)
        17: (0.0,  0.30),  # Right shoulder: (Pitch, Roll)
        18: (0.0,  0.20),  # Left: (Yaw, ElbowPitch)
        19: (0.0,  0.20),  # Right: (Yaw, ElbowPitch)
        20: deg(10),       # Left elbow yaw (slider expects radians)
        21: deg(10),       # Right elbow yaw
    },
}


# ---------------------------------------------------------------------------------
# Motion sequences (keyframes). Each entry is a list of (duration_seconds, pose_dict).
# pose_dict values are in *GUI space* (same types as gui slider values).
# The sequence player will interpolate between keyframes at the chosen FPS.
# ---------------------------------------------------------------------------------
MOTION_SEQUENCES = {
    "Greeting_Wave_Hands": [
        # Settle / neutral-ish
        (0.8, {
            16: (0.6, 0.05), 17: (0.6, 0.05),
            18: (0.0, 0.0),  19: (0.0, 0.0),
            20: deg(10),     21: deg(10),
            25: 0.0, 28: 0.0, 31: 0.0, 34: 0.0, 37: (0.0, 0.0),
            40: 0.0, 43: 0.0, 46: 0.0, 49: 0.0, 52: (0.0, 0.0),
        }),

        # Raise right arm
        (0.9, {17: (1.0, 0.25), 19: (0.0, 0.8), 21: deg(25)}),

        # Wave: right elbow yaw oscillation
        (0.35, {21: deg(55)}),
        (0.35, {21: deg(10)}),
        (0.35, {21: deg(55)}),
        (0.35, {21: deg(10)}),
        (0.35, {21: deg(55)}),
        (0.35, {21: deg(10)}),

        # Right hand open -> close -> open
        (0.50, {40: deg(-10), 43: deg(-10), 46: deg(-10), 49: deg(-10), 52: (deg(5), deg(10))}),
        (0.55, {40: deg(-60), 43: deg(-60), 46: deg(-45), 49: deg(-55), 52: (deg(10), deg(45))}),
        (0.50, {40: deg(-10), 43: deg(-10), 46: deg(-10), 49: deg(-10), 52: (deg(5), deg(10))}),

        # Bring left arm up + small pinch
        (0.8, {16: (0.9, 0.20), 18: (0.0, 0.55), 20: deg(20)}),
        (0.45, {25: deg(-30), 28: deg(-20), 31: deg(-15), 34: deg(-20), 37: (deg(8), deg(-30))}),
        (0.45, {25: deg(-8),  28: deg(-8),  31: deg(-8),  34: deg(-8),  37: (deg(4), deg(-10))}),

        # Return
        (1.0, {
            16: (0.6, 0.05), 17: (0.6, 0.05),
            18: (0.0, 0.0),  19: (0.0, 0.0),
            20: deg(10),     21: deg(10),
            25: 0.0, 28: 0.0, 31: 0.0, 34: 0.0, 37: (0.0, 0.0),
            40: 0.0, 43: 0.0, 46: 0.0, 49: 0.0, 52: (0.0, 0.0),
        }),
    ],

    "Arms_Open_Cross_Open": [
        (0.7, {16: (0.6, 0.00), 17: (0.6, 0.00), 18: (0.0, 0.0), 19: (0.0, 0.0), 20: deg(8), 21: deg(8)}),

        # Open arms
        (1.0, {16: (1.05, 0.35), 17: (1.05, 0.35), 18: (0.0, 0.45), 19: (0.0, 0.45), 20: deg(18), 21: deg(18)}),

        # Cross in front
        (1.0, {16: (0.85, 0.10), 17: (0.85, 0.10), 18: (0.0, 1.05), 19: (0.0, 1.05), 20: deg(28), 21: deg(28)}),

        # Elbow yaw shake
        (0.35, {20: deg(55), 21: deg(5)}),
        (0.35, {20: deg(5),  21: deg(55)}),
        (0.35, {20: deg(55), 21: deg(5)}),
        (0.35, {20: deg(5),  21: deg(55)}),

        # Open again
        (1.0, {16: (1.05, 0.35), 17: (1.05, 0.35), 18: (0.0, 0.45), 19: (0.0, 0.45), 20: deg(18), 21: deg(18)}),
        (0.8, {16: (0.6, 0.00), 17: (0.6, 0.00), 18: (0.0, 0.0), 19: (0.0, 0.0), 20: deg(8), 21: deg(8)}),
    ],

    "Finger_Warmup_BothHands": [
        # Open
        (0.6, {25: 0.0, 28: 0.0, 31: 0.0, 34: 0.0, 37: (0.0, 0.0),
               40: 0.0, 43: 0.0, 46: 0.0, 49: 0.0, 52: (0.0, 0.0)}),

        # Close gradually
        (0.6, {25: deg(-12), 28: deg(-12), 31: deg(-10), 34: deg(-10), 37: (deg(5), deg(-12)),
               40: deg(-12), 43: deg(-12), 46: deg(-10), 49: deg(-10), 52: (deg(5), deg(-12))}),
        (0.6, {25: deg(-25), 28: deg(-25), 31: deg(-22), 34: deg(-22), 37: (deg(10), deg(-25)),
               40: deg(-25), 43: deg(-25), 46: deg(-22), 49: deg(-22), 52: (deg(10), deg(-25))}),
        (0.6, {25: deg(-35), 28: deg(-35), 31: deg(-30), 34: deg(-30), 37: (deg(15), deg(-35)),
               40: deg(-35), 43: deg(-35), 46: deg(-30), 49: deg(-30), 52: (deg(15), deg(-35))}),

        # Wiggle thumbs
        (0.35, {37: (deg(20), deg(-20)), 52: (deg(20), deg(-20))}),
        (0.35, {37: (deg(-5), deg(-20)), 52: (deg(-5), deg(-20))}),
        (0.35, {37: (deg(20), deg(-20)), 52: (deg(20), deg(-20))}),
        (0.35, {37: (deg(-5), deg(-20)), 52: (deg(-5), deg(-20))}),

        # Open
        (0.8, {25: 0.0, 28: 0.0, 31: 0.0, 34: 0.0, 37: (0.0, 0.0),
               40: 0.0, 43: 0.0, 46: 0.0, 49: 0.0, 52: (0.0, 0.0)}),
    ],

    "Custom_Test_Sequence": [
      # 1. Left shoulder pitch = -2.25
      (0.6, {16: (-2.25, 0.0)}),

      # 2. Left shoulder roll = 0.85 (keep pitch)
      (0.6, {16: (-2.25, 0.85)}),

      # 3. Left elbow pitch = -1.0
      (0.6, {18: (0.0, -1.0)}),

      # 4. Left elbow pitch = 0
      (0.6, {18: (0.0, 0.0)}),

      # 5. Left elbow pitch = 0 (hold)
      (0.6, {18: (0.0, 0.0)}),

      # 6. All set to zero (return left arm)
      (0.8, {
          16: (0.0, 0.0),
          18: (0.0, 0.0),
      }),

      # 7. Right elbow pitch = 2.1
      (0.6, {19: (0.0, 2.1)}),

      # 8. Right shoulder pitch = -1.8
      (0.6, {17: (-1.8, 0.0)}),

      # 9. Right index finger = -1.3
      (0.5, {40: -1.3}),

      # 10. Right index finger = 0
      (0.5, {40: 0.0}),
  ] 
}
def main(model_path: Path) -> None:
    # ————————————————————————————————————————
    # 1)  Spin-up TCP/WebSocket server (Viser)
    # ————————————————————————————————————————
    server = viser.ViserServer()
    server.scene.set_up_direction("+y")
    server.scene.add_grid("/grid", position=(0.0, -1.3, 0.0), plane="xz")

    # 2)  Initialise SMPL helper & GUI
    #     └─ make_gui_elements(..) **creates all widgets & gizmos**
    model = SmplHelper(model_path)

    gui_elements = make_gui_elements(
        server,
        num_betas=model.num_betas,
        num_joints=model.num_joints,
        parent_idx=model.parent_idx,
    )

    # 3)  Add mesh to scene (updated each frame)
    body_handle = server.scene.add_mesh_simple(
        "/human",
        model.v_template,
        model.faces,
        wireframe=gui_elements.gui_wireframe.value,
        color=gui_elements.gui_rgb.value,
    )

    # 4)  Main render/update loop – recompute SMPL when GUI changed
    red_sphere = trimesh.creation.icosphere(radius=0.001, subdivisions=1)
    red_sphere.visual.vertex_colors = (255, 0, 0, 255)  # type: ignore

    # print("render loop started")



    while True:
        time.sleep(0.02)  # crude throttling
        if not gui_elements.changed:
            continue

        gui_elements.changed = False

        

        # (Re)-evaluate SMPL with current GUI values

        # modify web pose
        # axes = []
        # for i, g in enumerate(gui_elements.gui_joints):
        #     a = to_axisangle(g.value, i)
        #     a = a + VISUAL_OFFSET.get(i, np.array([0.0, 0.0, 0.0]))
        #     axes.append(a)

        axes = []
        wxyz_vis = []
        for i, g in enumerate(gui_elements.gui_joints):
            a = to_axisangle(g.value, i)
            R_gui = tf.SO3.exp(a)
            off = VISUAL_OFFSET.get(i, None)
            if off is None:
                R_vis = R_gui
            else:
                R_off = tf.SO3.exp(off)
                R_vis = R_off @ R_gui
            axes.append(R_vis.log())
            wxyz_vis.append(R_vis.wxyz)

        smpl_outputs = model.get_outputs(
            betas=np.array([x.value for x in gui_elements.gui_betas]),
            joint_rotmats= tf.SO3.exp(np.array(axes)).as_matrix(),
        )

        # modified version
        # smpl_outputs = model.get_outputs(
        #     betas=np.array([x.value for x in gui_elements.gui_betas]),
        #     joint_rotmats= tf.SO3.exp(np.array(axes)).as_matrix(),
        # )

        # print("model has output")

        for i, control in enumerate(gui_elements.transform_controls):
            control.wxyz = wxyz_vis[i]
            control.position = smpl_outputs.T_parent_joint[i, :3, 3]

        # smpl_outputs = model.get_outputs(
        #     betas=np.array([x.value for x in gui_elements.gui_betas]),
        #     joint_rotmats= tf.SO3.exp(np.array([to_axisangle(g.value, i) for i,g in enumerate(gui_elements.gui_joints)])).as_matrix(),
        # )



        # Reflect into scene
        body_handle.vertices = smpl_outputs.vertices
        body_handle.wireframe = gui_elements.gui_wireframe.value
        body_handle.color = gui_elements.gui_rgb.value

        # Update gizmo positions so they stick to joints
        for i, control in enumerate(gui_elements.transform_controls):
            control.position = smpl_outputs.T_parent_joint[i, :3, 3]
        # updated gizmo
        # for i, control in enumerate(gui_elements.transform_controls):
        #     control.position = smpl_outputs.T_world_joint[i, :3, 3]


##############################################
#      GUI FACTORY – builds all user widgets  #
##############################################

hidden_indices = {26,27,29,30,32,33,35,36,38,39,41,42,44,45,47,48,50,51,53,54}

def make_gui_elements(
    server: viser.ViserServer,
    num_betas: int,
    num_joints: int,
    parent_idx: np.ndarray,
) -> "GuiElements":
    """Create every GUI widget and scene handle used by the app."""

    # ──────────────────────────────────────────
    # Tab layout container (left side of viewer)
    # ──────────────────────────────────────────
    tab_group = server.gui.add_tab_group()

    # Internal helper toggling a global "dirty" flag so the main loop knows
    # when to recompute the mesh after *any* widget changes.
    def set_changed(_):
        out.changed = True  # 'out' will be defined later after widgets exist.

    # ==============================================================================================
    # 1. VIEW TAB  — general render settings
    # ==============================================================================================
    with tab_group.add_tab("View", viser.Icon.VIEWFINDER):
        gui_rgb = server.gui.add_rgb("Color", initial_value=(90, 200, 255))
        gui_wireframe = server.gui.add_checkbox("Wireframe", initial_value=False)
        gui_show_controls = server.gui.add_checkbox("Handles", initial_value=True)

        gui_rgb.on_update(set_changed)
        gui_wireframe.on_update(set_changed)

        @gui_show_controls.on_update
        def _(_):
            # Show / hide scene gizmos (TransformControls)
            for control in transform_controls:
                control.visible = gui_show_controls.value

    # ==============================================================================================
    # 2. SHAPE TAB  — β-shape sliders (body thickness, height, …)
    # ==============================================================================================
    with tab_group.add_tab("Shape", viser.Icon.BOX):
        gui_reset_shape = server.gui.add_button("Reset Shape")
        gui_random_shape = server.gui.add_button("Random Shape")

        @gui_reset_shape.on_click
        def _(_):
            for beta in gui_betas:
                beta.value = 0.0

        @gui_random_shape.on_click
        def _(_):
            for beta in gui_betas:
                beta.value = np.random.normal(loc=0.0, scale=1.0)

        gui_betas = []
        for i in range(num_betas):
            beta = server.gui.add_slider(
                f"beta{i}", min=-5.0, max=5.0, step=0.01, initial_value=0.0
            )
            gui_betas.append(beta)
            beta.on_update(set_changed)

    # ==============================================================================================
    # 3. JOINTS TAB  — per-joint axis-angle controls
    # ==============================================================================================


    with tab_group.add_tab("Joints", viser.Icon.ANGLE):
      
        gui_reset_joints = server.gui.add_button("Reset Joints")
        gui_random_joints = server.gui.add_button("Random Joints")

        @gui_reset_joints.on_click
        def _(_):
            for joint in gui_joints:
                joint.value = (0.0, 0.0, 0.0)

        @gui_random_joints.on_click
        def _(_):
            rng = np.random.default_rng()
            for joint in gui_joints:
                joint.value = tf.SO3.sample_uniform(rng).log()

        gui_joints: list[viser.GuiInputHandle[tuple[float, float, float]]] = []
        for i in range(num_joints):
            if i == 16:
                gui_joint = server.gui.add_vector2(
                    label= f"Left Shoulder Pitch & Left Shoulder Roll",
                     initial_value=(0.0,0.0),
                     step=0.05,
                )
            elif i == 17:
              gui_joint = server.gui.add_vector2(
                    label= f"Right Shoulder Pitch & Right Shoulder Roll",
                     initial_value=(0.0,0.0),
                     step=0.05,
                )
            elif i == 18:
                gui_joint = server.gui.add_vector2(
                    label= f"Left Shoulder Yaw & Left Elbow Pitch",
                     initial_value=(0.0,0.0),
                     step=0.05,
                )
            elif i == 19:
              gui_joint = server.gui.add_vector2(
                    label= f"Right Shoulder Yaw & Right Elbow Pitch",
                     initial_value=(0.0,0.0),
                     step=0.05,
                )
            elif i == 20:
                gui_joint = server.gui.add_slider(
                      label = f"Left Elbow Yaw",
                      initial_value= 0.0,
                      step= 0.05,
                      min = deg(-123),
                      max = deg(123)
                  )
            elif i == 21:
                gui_joint = server.gui.add_slider(
                      label = f"Right Elbow Yaw",
                      initial_value= 0.0,
                      step= 0.05,
                      # min = -1.2,
                      # max = 1.2
                      min = deg(-123),
                      max = deg(123)
                  )
                
            elif i in (25,28,31,34,40,43,46,49):
                if i == 25: 
                  gui_joint = server.gui.add_slider(
                      label = f"Left Index Finger",
                      initial_value= 0.0,
                      step= 0.05,
                      # min = -1.2,
                      # max = 0.1
                      min = deg(-123),
                      max = deg(18)
                  )
                elif i == 28:
                    gui_joint = server.gui.add_slider(
                      label = f"Left Middle Finger",
                      initial_value= 0.0,
                      step= 0.05,
                      # min = -1.2,
                      # max = 0.1
                      min = deg(-132),
                      max = deg(18)
                  )
                elif i == 31:
                    gui_joint = server.gui.add_slider(
                      label = f"Left Pinkie Finger",
                      initial_value= 0.0,
                      step= 0.05,
                      # min = -1.2,
                      # max = 0.1
                      min = deg(-75),
                      max = deg(4)
                  )
                elif i == 34:
                    gui_joint = server.gui.add_slider(
                      label = f"Left Ring Finger",
                      initial_value= 0.0,
                      step= 0.05,
                      # min = -1.2,
                      # max = 0.1
                      min = deg(-136),
                      max = deg(18)
                  )
                elif i == 40: 
                  gui_joint = server.gui.add_slider(
                      label = f"Right Index Finger",
                      initial_value= 0.0,
                      step= 0.05,
                      # min = -1.2,
                      # max = 0.1
                      min = deg(-123),
                      max = deg(18)
                  )
                elif i == 43:
                    gui_joint = server.gui.add_slider(
                      label = f"Right Middle Finger",
                      initial_value= 0.0,
                      step= 0.05,
                      # min = -1.2,
                      # max = 0.1
                      min = deg(-132),
                      max = deg(18)
                  )
                elif i == 46:
                    gui_joint = server.gui.add_slider(
                      label = f"Right Pinkie Finger",
                      initial_value= 0.0,
                      step= 0.05,
                      # min = -1.2,
                      # max = 0.1
                      min = deg(-75),
                      max = deg(4)
                  )
                elif i == 49:
                    gui_joint = server.gui.add_slider(
                      label = f"Right Ring Finger",
                      initial_value= 0.0,
                      step= 0.05,
                      # min = -1.2,
                      # max = 0.1
                      min = deg(-136),
                      max = deg(18)
                  )
                    
            elif i == 37:
                left_thumb = server.gui.add_vector2(
                    label= f"Left Thumb Roll & Left Thumb Finger",
                     initial_value=(0.0,0.0),
                     step=0.05,
                     visible = False,
                )
                LeftThumb_roll = server.gui.add_slider(
                    label = "Left Thumb Roll",
                    initial_value= 0.0,
                    step = 0.05,
                    min = deg(-31),
                    max = deg(22),
                )
                LeftThumb_finger = server.gui.add_slider(
                    label = "Left Thumb Finger",
                    initial_value= 0.0,
                    step = 0.05,
                    min = deg(-44),
                    max = deg(75),
                )
                def _sync_left_thumb(_):
                    left_thumb.value = (float(LeftThumb_roll.value), float(-1.0 * LeftThumb_finger.value)) # magic number -1.0 used to match the moving direction between robot and web pose
                    print("read slider number")
                    out.changed = True
                LeftThumb_roll.on_update(_sync_left_thumb)
                LeftThumb_finger.on_update(_sync_left_thumb)
                gui_joint = left_thumb
            elif i == 52:
                right_thumb = server.gui.add_vector2(
                    label = "Right Thumb Roll & Right Thumb Finger",
                     initial_value=(0.0,0.0),
                     step=0.05,
                     visible = False,
                )
                RightThumb_roll = server.gui.add_slider(
                    label = "Right Thumb Roll",
                    initial_value= 0.0,
                    step = 0.05,
                    min = deg(-31),
                    max = deg(22),
                )
                RightThumb_finger = server.gui.add_slider(
                    label = "Right Thumb Finger",
                    initial_value= 0.0,
                    step = 0.05,
                    min = deg(-44),
                    max = deg(75),
                )
                def _sync_right_thumb(_):
                    right_thumb.value = (float(RightThumb_roll.value), float(RightThumb_finger.value))
                    out.changed = True
                RightThumb_roll.on_update(_sync_right_thumb)
                RightThumb_finger.on_update(_sync_right_thumb)
                gui_joint = right_thumb
            elif i in (26,27,29,30,32,33,35,36,38,39,41,42,44,45,47,48,50,51,53,54):
                gui_joint = server.gui.add_slider(
                    label = f"Joint {i}(1-DOF)",
                    initial_value= 0.0,
                    step= 0.05,
                    min = 0,
                    max = 4,
                    visible = False
                )
            else:
                # Each vector3 widget holds axis-angle (x,y,z) for one joint
                gui_joint = server.gui.add_vector3(
                    label=f"Joint {i}",
                    initial_value=(0.0, 0.0, 0.0),
                    step=0.05,
                )

            gui_joints.append(gui_joint)

            

            # <callback> When user drags a joint slider we update the gizmo rotation & mark dirty
            def set_callback_in_closure(i: int) -> None:
                @gui_joint.on_update
                def _(_):
                    if i == 25:
                      gui_joints[26].value = gui_joints[25].value
                      gui_joints[27].value = gui_joints[25].value
                      axis = to_axisangle(gui_joints[i].value, i)
                      # transform_controls[i].wxyz = tf.SO3.exp(axis).wxyz
                      R_gui = tf.SO3.exp(axis)
                      off = VISUAL_OFFSET.get(i, None)
                      R_vis = (tf.SO3.exp(off) @ R_gui) if off is not None else R_gui
                      transform_controls[i].wxyz = R_vis.wxyz
                      out.changed = True
                    elif i == 28:
                        gui_joints[29].value = gui_joints[28].value
                        gui_joints[30].value = gui_joints[28].value
                        axis = to_axisangle(gui_joints[i].value, i)
                        # transform_controls[i].wxyz = tf.SO3.exp(axis).wxyz
                        R_gui = tf.SO3.exp(axis)
                        off = VISUAL_OFFSET.get(i, None)
                        R_vis = (tf.SO3.exp(off) @ R_gui) if off is not None else R_gui
                        transform_controls[i].wxyz = R_vis.wxyz
                        out.changed = True
                    elif i == 31:
                        gui_joints[32].value = gui_joints[31].value
                        gui_joints[33].value = gui_joints[31].value
                        axis = to_axisangle(gui_joints[i].value, i)
                        # transform_controls[i].wxyz = tf.SO3.exp(axis).wxyz
                        R_gui = tf.SO3.exp(axis)
                        off = VISUAL_OFFSET.get(i, None)
                        R_vis = (tf.SO3.exp(off) @ R_gui) if off is not None else R_gui
                        transform_controls[i].wxyz = R_vis.wxyz
                        out.changed = True
                    elif i == 34:
                        gui_joints[35].value = gui_joints[34].value
                        gui_joints[36].value = gui_joints[34].value
                        axis = to_axisangle(gui_joints[i].value, i)
                        # transform_controls[i].wxyz = tf.SO3.exp(axis).wxyz
                        R_gui = tf.SO3.exp(axis)
                        off = VISUAL_OFFSET.get(i, None)
                        R_vis = (tf.SO3.exp(off) @ R_gui) if off is not None else R_gui
                        transform_controls[i].wxyz = R_vis.wxyz
                        out.changed = True
                    elif i == 37:
                        gui_joints[38].value = gui_joints[37].value[1]
                        gui_joints[39].value = gui_joints[37].value[1]
                        axis = to_axisangle(gui_joints[i].value, i)
                        # transform_controls[i].wxyz = tf.SO3.exp(axis).wxyz
                        R_gui = tf.SO3.exp(axis)
                        off = VISUAL_OFFSET.get(i, None)
                        R_vis = (tf.SO3.exp(off) @ R_gui) if off is not None else R_gui
                        transform_controls[i].wxyz = R_vis.wxyz
                        out.changed = True
                    elif i == 52:
                        gui_joints[53].value = gui_joints[52].value[1]
                        gui_joints[54].value = gui_joints[52].value[1]
                        axis = to_axisangle(gui_joints[i].value, i)
                        # transform_controls[i].wxyz = tf.SO3.exp(axis).wxyz
                        R_gui = tf.SO3.exp(axis)
                        off = VISUAL_OFFSET.get(i, None)
                        R_vis = (tf.SO3.exp(off) @ R_gui) if off is not None else R_gui
                        transform_controls[i].wxyz = R_vis.wxyz
                        out.changed = True
                    elif i == 40:
                      gui_joints[41].value = gui_joints[40].value
                      gui_joints[42].value = gui_joints[40].value
                      axis = to_axisangle(gui_joints[i].value, i)
                      # transform_controls[i].wxyz = tf.SO3.exp(axis).wxyz
                      R_gui = tf.SO3.exp(axis)
                      off = VISUAL_OFFSET.get(i, None)
                      R_vis = (tf.SO3.exp(off) @ R_gui) if off is not None else R_gui
                      transform_controls[i].wxyz = R_vis.wxyz
                      out.changed = True
                    elif i == 43:
                        gui_joints[44].value = gui_joints[43].value
                        gui_joints[45].value = gui_joints[43].value
                        axis = to_axisangle(gui_joints[i].value, i)
                        # transform_controls[i].wxyz = tf.SO3.exp(axis).wxyz
                        R_gui = tf.SO3.exp(axis)
                        off = VISUAL_OFFSET.get(i, None)
                        R_vis = (tf.SO3.exp(off) @ R_gui) if off is not None else R_gui
                        transform_controls[i].wxyz = R_vis.wxyz
                        out.changed = True
                    elif i == 46:
                        gui_joints[47].value = gui_joints[46].value
                        gui_joints[48].value = gui_joints[46].value
                        axis = to_axisangle(gui_joints[i].value, i)
                        # transform_controls[i].wxyz = tf.SO3.exp(axis).wxyz
                        R_gui = tf.SO3.exp(axis)
                        off = VISUAL_OFFSET.get(i, None)
                        R_vis = (tf.SO3.exp(off) @ R_gui) if off is not None else R_gui
                        transform_controls[i].wxyz = R_vis.wxyz
                        out.changed = True
                    elif i == 49:
                        gui_joints[50].value = gui_joints[49].value
                        gui_joints[51].value = gui_joints[49].value
                        axis = to_axisangle(gui_joints[i].value, i)
                        # transform_controls[i].wxyz = tf.SO3.exp(axis).wxyz
                        R_gui = tf.SO3.exp(axis)
                        off = VISUAL_OFFSET.get(i, None)
                        R_vis = (tf.SO3.exp(off) @ R_gui) if off is not None else R_gui
                        transform_controls[i].wxyz = R_vis.wxyz
                        out.changed = True
                        
                        
                    else:
                      axis = to_axisangle(gui_joints[i].value,i) 
                      # transform_controls[i].wxyz = tf.SO3.exp(axis).wxyz
                      R_gui = tf.SO3.exp(axis)
                      off = VISUAL_OFFSET.get(i, None)
                      R_vis = (tf.SO3.exp(off) @ R_gui) if off is not None else R_gui
                      transform_controls[i].wxyz = R_vis.wxyz
                      out.changed = True

                    value = tuple(to_axisangle(gui_joints[i].value,i))

                    if isinstance(value, np.ndarray):
                        value = value.tolist()
                    else:
                        value = tuple(float(x) for x in value)

                    print(value)
                    if i not in hidden_indices:
                        Sophia_control.call_remote(index = i,value = value)
                    

            set_callback_in_closure(i)
        # ------------------------------------------------------------------
        # Preset buttons (no manual dragging required)
        # Clicking a preset will:
        #   1) set the GUI widgets (so web-pose updates)
        #   2) trigger existing on_update callbacks (so robot commands are sent)
        # ------------------------------------------------------------------
        with server.gui.add_folder("Presets"):
            def _apply_preset(pose: dict[int, object]) -> None:
                for idx, val in pose.items():
                    # Setting .value triggers the existing slider callback logic,
                    # which updates transform controls, marks 'out.changed', and sends to robot.
                    gui_joints[idx].value = val
                out.changed = True

            for _name, _pose in PRESET_POSES.items():
                _btn = server.gui.add_button(f"Apply: {_name}")

                @_btn.on_click
                def _(_evt, __pose=_pose):
                    _apply_preset(__pose)


            # ------------------------------------------------------------------
            # Motion sequence player (updates GUI values => updates web + robot)
            # ------------------------------------------------------------------
            _player_lock = threading.Lock()
            _stop_event = threading.Event()
            _player_thread = {"t": None}  # mutable holder

            def _interp(a, b, alpha: float):
                # supports scalar and len-2 tuples/lists
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return (1.0 - alpha) * float(a) + alpha * float(b)
                if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
                    return tuple((1.0 - alpha) * float(x) + alpha * float(y) for x, y in zip(a, b))
                return b

            def _run_sequence(seq_name: str, fps: float, loop: bool):
                seq = MOTION_SEQUENCES[seq_name]
                dt = 1.0 / max(fps, 1e-6)

                while True:
                    for duration, target_pose in seq:
                        if _stop_event.is_set():
                            return

                        # snapshot current values for joints in this keyframe
                        start_pose = {idx: gui_joints[idx].value for idx in target_pose.keys()}
                        steps = max(1, int(duration / dt))

                        for s in range(steps):
                            if _stop_event.is_set():
                                return
                            alpha = (s + 1) / steps
                            for idx, target_val in target_pose.items():
                                gui_joints[idx].value = _interp(start_pose[idx], target_val, alpha)
                            out.changed = True
                            time.sleep(dt)

                    if not loop:
                        return

            with server.gui.add_folder("Sequences"):
                _seq_names = list(MOTION_SEQUENCES.keys())
                gui_seq = server.gui.add_dropdown("Sequence", options=_seq_names, initial_value=_seq_names[0])
                gui_fps = server.gui.add_slider("Sequence FPS", min=5, max=60, step=1, initial_value=30)
                gui_loop = server.gui.add_checkbox("Loop", initial_value=False)
                btn_play = server.gui.add_button("▶ Play sequence")
                btn_stop = server.gui.add_button("■ Stop")

                @btn_play.on_click
                def _(_evt):
                    with _player_lock:
                        _stop_event.set()
                        t = _player_thread["t"]
                        if t is not None and t.is_alive():
                            t.join(timeout=0.2)

                        _stop_event.clear()
                        t = threading.Thread(
                            target=_run_sequence,
                            args=(gui_seq.value, float(gui_fps.value), bool(gui_loop.value)),
                            daemon=True,
                        )
                        _player_thread["t"] = t
                        t.start()

                @btn_stop.on_click
                def _(_evt):
                    _stop_event.set()


    # =====================================================================
    # 4. TRANSFORM CONTROLS (Scene gizmos attached to joints)              
    # =====================================================================
    # These are the coloured draggable arrows in the 3-D viewport.
    transform_controls: list[viser.TransformControlsHandle] = []
    prefixed_joint_names = []  # e.g. "root/hip_left/knee_left/..."
    for i in range(num_joints):
        prefixed_joint_name = f"joint_{i}"
        if i > 0:
            prefixed_joint_name = (
                prefixed_joint_names[parent_idx[i]] + "/" + prefixed_joint_name
            )
        prefixed_joint_names.append(prefixed_joint_name)

        controls = server.scene.add_transform_controls(
            f"/smpl/{prefixed_joint_name}",
            depth_test=False,
            scale=0.2 * (0.75 ** prefixed_joint_name.count("/")),
            disable_axes=True,
            disable_sliders=True,
            visible=True,  # sync later
        )
        transform_controls.append(controls)

        # <callback> Scene gizmo → UI synchronisation (inverse of earlier)
        def set_callback_in_closure(i: int) -> None:
            @controls.on_update
            def _(_) -> None:
                # axisangle = tf.SO3(controls.wxyz).log()
                R_vis = tf.SO3(controls.wxyz)
                off = VISUAL_OFFSET.get(i, None)
                if off is None:
                    R_gui = R_vis
                else:
                    R_off = tf.SO3.exp(off)
                    R_gui = R_off.inv() @ R_vis
                axisangle = R_gui.log()
                if len(gui_joints[i].value) == 2:
                    gui_joints[i].value = (axisangle[1], axisangle[2])
                else:
                    gui_joints[i].value = tuple(axisangle)

        set_callback_in_closure(i)

    # Bundle everything into a convenient struct that the main loop polls
    out = GuiElements(
        gui_rgb,
        gui_wireframe,
        gui_betas,
        gui_joints,
        transform_controls=transform_controls,
        changed=True,  # Force first recompute
    )
    return out

# ——————————————————————————————————————————————————————————————————————————
# Data-holder returned by make_gui_elements(..)
# ——————————————————————————————————————————————————————————————————————————
@dataclass
class GuiElements:
    """Container for easiest passing of GUI handles & state."""

    gui_rgb: viser.GuiInputHandle[tuple[int, int, int]]
    gui_wireframe: viser.GuiInputHandle[bool]
    gui_betas: list[viser.GuiInputHandle[float]]
    gui_joints: list[viser.GuiInputHandle[tuple[float, float, float]]]
    transform_controls: list[viser.TransformControlsHandle]
    changed: bool  # set to True whenever mesh needs recompute

# Dummy helper used somewhere else

def getindexandvalue(i, value):
    return i, value


# ——————————————————————————————————————————————————————————————————————————
#  CLI Entry – allow running with `python smpl_visualizer_annotated.py --model_path model.npz`
# ——————————————————————————————————————————————————————————————————————————
if __name__ == "__main__":
    tyro.cli(main, description=__doc__)
