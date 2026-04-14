"""
Motion repository: preset joint configurations for the robot.
All angles in degrees. Unspecified joints remain where they at.
"""

import math

# All controllable joints (must match system_prompt allowed actuators)
ALL_JOINTS = [
    "RightShoulderPitch",
    "RightShoulderRoll",
    "RightShoulderYaw",
    "RightElbowPitch",
    "RightElbowYaw",
    "RightWristPitch",
    "RightWristRoll",
    "RightThumbRoll",
    "RightThumbFinger",
    "RightIndexFinger",
    "RightMiddleFinger",
    "RightRingFinger",
    "RightPinkyFinger",
    "LeftShoulderPitch",
    "LeftShoulderRoll",
    "LeftShoulderYaw",
    "LeftElbowPitch",
    "LeftElbowYaw",
    "LeftWristPitch",
    "LeftWristRoll",
    "LeftThumbRoll",
    "LeftThumbFinger",
    "LeftIndexFinger",
    "LeftMiddleFinger",
    "LeftRingFinger",
    "LeftPinkyFinger",
]

# Default: all joints at 0
DEFAULT_JOINT_ANGLES = {j: 0.0 for j in ALL_JOINTS}


def deg2rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * math.pi / 180.0


def get_motion(name: str) -> dict[str, float]:
    """
    only return the motors that are involved.
    """
    if name not in MOTIONS:
        raise KeyError(f"Unknown motion: {name}")
    return MOTIONS[name].copy()

def get_motion_rad(name: str) -> dict[str, float]:
    """Return full joint angles in radians for a motion preset."""
    return {j: deg2rad(v) for j, v in get_motion(name).items()}


# ---------------------------------------------------------------------------
# Motion presets (degrees). Only specify joints that differ from 0.
# ---------------------------------------------------------------------------

MOTIONS = { # one way for complex motion: define several mini-montions which cannot be reached by llm and wrap them into a bigger motion and make the bigger motion visible to LLM
    "standby":{
        "RightShoulderPitch": 0,
        "RightShoulderRoll": 0,
        "RightShoulderYaw": 0,
        "RightElbowPitch": 0,
        "RightElbowYaw": 0,
        "RightWristPitch": 0,
        "RightWristRoll": 0,
        "RightThumbRoll": 0,
        "RightThumbFinger": 0,
        "RightIndexFinger": 0,
        "RightMiddleFinger": 0,
        "RightRingFinger": 0,
        "RightPinkyFinger": 0,
        "LeftShoulderPitch": 0,
        "LeftShoulderRoll": 0,
        "LeftShoulderYaw": 0,
        "LeftElbowPitch": 0,
        "LeftElbowYaw": 0,
        "LeftWristPitch": 0,
        "LeftWristRoll": 0,
        "LeftThumbRoll": 0,
        "LeftThumbFinger": 0,
        "LeftIndexFinger": 0,
        "LeftMiddleFinger": 0,
        "LeftRingFinger": 0,
        "LeftPinkyFinger": 0,
    },
    "leftThumbUp": {
        "LeftShoulderPitch": 71.0,
        "LeftThumbFinger": -44.0,
        "LeftIndexFinger": 123.0,
        "LeftMiddleFinger": 132.0,
        "LeftRingFinger": 136.0,
        "LeftPinkyFinger": 75.0,
        "LeftShoulderRoll": -7.0,
        "LeftThumbRoll": 22,
    },
    "rightThumbUp": {
        "RightShoulderPitch": -71.0,
        "RightThumbFinger": 44.0,
        "RightIndexFinger": 123.0, # wrong parameter on the webend
        "RightMiddleFinger": -132.0, # wrong parameter on the webend
        "RightRingFinger": -136.0, # wrong parameter on the webend
        "RightPinkyFinger": 75.0, # wrong parameter on the webend
        "RightThumbRoll": 22,
    },
    "rightHandRaise":{
        "RightShoulderPitch": -70.0,
        "RightShoulderYaw": -13,
        "RightElbowPitch": 108,
        "RightElbowYaw": -123,
    },
    "rightHandWaveRight":{
        "RightShoulderPitch": -70.0,
        "RightShoulderYaw": -66,
        "RightElbowPitch": 108,
        "RightElbowYaw": -123,
    },
    "rightHandWaveLeft":{
        "RightShoulderPitch": -70.0,
        "RightShoulderYaw": 63,
        "RightElbowPitch": 108,
        "RightElbowYaw": -123,
    },
    "leftHandRaise":{
        "LeftShoulderPitch": 70.0,
        "LeftShoulderYaw": 13,
        "LeftElbowPitch": -108,
        "LeftElbowYaw": 123,
    },
    "leftHandWaveRight":{
        "LeftShoulderPitch": 70.0,
        "LeftShoulderYaw": 63,
        "LeftElbowPitch": -108,
        "LeftElbowYaw": 123,
    },
    "leftHandWaveLeft":{
        "LeftShoulderPitch": 70.0,
        "LeftShoulderYaw": -63,
        "LeftElbowPitch": -108,
        "LeftElbowYaw": 123,
    },
    "idea":{
        "LeftShoulderPitch": 26,
        "LeftElbowPitch": -117,
        "LeftMiddleFinger": 132,
        "LeftRingFinger": 136,
        "LeftPinkyFinger": 75
    },
    "rightHandReachOut":{
        "RightShoulderPitch": -111,
        "RightElbowPitch": -127,
    },
    "leftHandReachOut":{
        "LeftShoulderPitch": 111,
        "LeftElbowPitch": 127,
    },
    "spreadHands": {
        "RightShoulderPitch": -35,
        "RightShoulderYaw": -40,
        "RightElbowPitch": -13,
        "RightElbowYaw": 90,
        "LeftShoulderPitch": 35,
        "LeftShoulderYaw": 40,
        "LeftElbowPitch": 13,
        "LeftElbowYaw": -90,
    },
    "lookToTheLeft": {
        "NeckRotation": -40,
    },
    "lookToTheRight": {
        "NeckRotation": 40,
    },
    "peaceSign": {
        "RightShoulderPitch": -31,
        "RightShoulderYaw": -12,
        "RightElbowPitch": 119,
        "RightElbowYaw": -113,
        "RightThumbFinger": -75,
        "RightRingFinger": -136,
        "RightPinkyFinger": 75,
  
    },
    "leftHandGestureOne": {
        "LeftElbowYaw": 123,
        "LeftShoulderPitch": 68,
        "LeftElbowPitch": -119,
        "LeftThumbFinger": 75,
        "LeftIndexFinger": -18,
        "LeftMiddleFinger": 132,
        "LeftRingFinger": 136,
        "LeftPinkyFinger": 75,
    },
    "leftHandGestureTwo": {
        "LeftElbowYaw": 123,
        "LeftShoulderPitch": 68,
        "LeftElbowPitch": -119,
        "LeftThumbFinger": 75,
        "LeftIndexFinger": -18,
        "LeftMiddleFinger": -18,
        "LeftRingFinger": 136,
        "LeftPinkyFinger": 75,
    },
    "leftHandGestureThree": {
        "LeftElbowYaw": 123,
        "LeftShoulderPitch": 68,
        "LeftElbowPitch": -119,
        "LeftThumbFinger": 75,
        "LeftIndexFinger": -18,
        "LeftMiddleFinger": -18,
        "LeftRingFinger": -18,
        "LeftPinkyFinger": 75,
    },
    "leftHandGestureFour": {
        "LeftElbowYaw": 123,
        "LeftShoulderPitch": 68,
        "LeftElbowPitch": -119,
        "LeftThumbFinger": 75,
        "LeftIndexFinger": -18,
        "LeftMiddleFinger": -18,
        "LeftRingFinger": -18,
        "LeftPinkyFinger": -4,
    },
    "leftHandGestureFive": {
        "LeftElbowYaw": 123,
        "LeftShoulderPitch": 68,
        "LeftElbowPitch": -119,
        "LeftThumbFinger": -44,
        "LeftIndexFinger": -18,
        "LeftMiddleFinger": -18,
        "LeftRingFinger": -18,
        "LeftPinkyFinger": -4,
    },
    "rightArmLiftFlat":{
        "RightShoulderPitch": -145,
        "RightElbowPitch": -127,

    },
    "leftArmLiftFlat":{
        "LeftShoulderPitch": 145,
        "LeftElbowPitch": 127,

    },
    "stay":{
        
    },
    "leftShoulderYawOut":{
        "LeftShoulderYaw": 66,
    },
    "rightShoulderYawOut":{
        "RightShoulderYaw": -66,
    },
    "rightArmStretchAndRaise":{
        "RightShoulderRoll": 101,
        "RightShoulderYaw": -66,
        "RightElbowPitch": -127,
        "RightElbowYaw": 123,
    }
}