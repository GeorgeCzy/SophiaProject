"""
Motion repository: preset joint configurations for the robot.
All angles in degrees. Unspecified joints remain where they are.

MOTION_DESCRIPTIONS is intentionally written for LLM motion planning. Keep each
description physical: it should describe what the robot's body looks like in
that atomic keyframe. Semantic use is kept separately as a usage hint, so the
planner can compose primitives with A+B when a higher-level gesture needs more
than one body part.
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


MOTION_DESCRIPTIONS = {
    "standby": {
        "category": "neutral",
        "description": "Both shoulders, elbows, wrists, fingers, and gimbal joints are at the zero neutral pose.",
        "best_for": "Ending almost every gesture sequence cleanly or resetting after a compound pose.",
        "caution": "Use as the final keyframe, not as the main expressive gesture.",
    },
    "stay": {
        "category": "neutral",
        "description": "No actuator target changes; the robot keeps the pose it already has.",
        "best_for": "Brief pauses or intentionally holding a pose.",
        "caution": "Do not use as the final keyframe unless holding is explicitly desired.",
    },
    "leftThumbUp": {
        "category": "positive",
        "description": "Left upper arm lifts forward, the left elbow is slightly bent, the left thumb stays extended, and the other left fingers curl inward.",
        "best_for": "Agreement, encouragement, success, praise, or confidence. Can combine with rightThumbUp for stronger two-handed approval.",
        "caution": "Avoid for sad, serious, or factual neutral speech.",
    },
    "rightThumbUp": {
        "category": "positive",
        "description": "Right upper arm lifts forward, the right elbow is slightly bent, the right thumb stays extended, and the other right fingers curl inward.",
        "best_for": "Agreement, encouragement, success, praise, or confidence. Can combine with leftThumbUp for stronger two-handed approval.",
        "caution": "Avoid for sad, serious, or factual neutral speech.",
    },
    "peaceSign": {
        "category": "positive",
        "description": "Right arm stays near the body while the right index and middle fingers extend and the thumb, ring, and pinky fingers curl inward.",
        "best_for": "Lighthearted positivity, celebration, friendly photos, or upbeat endings.",
        "caution": "Too playful for solemn content.",
    },
    "rightHandRaise": {
        "category": "greeting",
        "description": "Right shoulder lifts the upper arm, right elbow bends, and the right forearm rises upright with the hand near or above shoulder height.",
        "best_for": "A visible raised-right-hand atom. Can start a wave or combine with leftHandRaise for both-hands-up gestures.",
        "caution": "Usually follow with rightHandWaveLeft/rightHandWaveRight or standby.",
    },
    "rightHandWaveRight": {
        "category": "greeting",
        "description": "Right arm remains raised while the upper arm and forearm shift the raised right hand toward the robot's right side.",
        "best_for": "One beat in a right-hand wave sequence.",
        "caution": "Use after rightHandRaise or between wave beats.",
    },
    "rightHandWaveLeft": {
        "category": "greeting",
        "description": "Right arm remains raised while the upper arm and forearm shift the raised right hand across toward the robot's left side.",
        "best_for": "One beat in a right-hand wave sequence.",
        "caution": "Use after rightHandRaise or between wave beats.",
    },
    "leftHandRaise": {
        "category": "greeting",
        "description": "Left shoulder lifts the upper arm, left elbow bends, and the left forearm rises upright with the hand near or above shoulder height.",
        "best_for": "A visible raised-left-hand atom. Can start a wave or combine with rightHandRaise for both-hands-up gestures.",
        "caution": "Usually follow with leftHandWaveLeft/leftHandWaveRight or standby.",
    },
    "leftHandWaveRight": {
        "category": "greeting",
        "description": "Left arm remains raised while the upper arm and forearm shift the raised left hand across toward the robot's right side.",
        "best_for": "One beat in a left-hand wave sequence.",
        "caution": "Use after leftHandRaise or between wave beats.",
    },
    "leftHandWaveLeft": {
        "category": "greeting",
        "description": "Left arm remains raised while the upper arm and forearm shift the raised left hand toward the robot's left side.",
        "best_for": "One beat in a left-hand wave sequence.",
        "caution": "Use after leftHandRaise or between wave beats.",
    },
    "idea": {
        "category": "thinking",
        "description": "Right upper arm comes forward, right elbow bends close to the torso, and the right hand closes into a compact pose near the body.",
        "best_for": "Ideas, reasoning, hesitation, checking, or 'let me think' moments.",
        "caution": "Avoid repeating too often in one sequence.",
    },
    "rightHandReachOut": {
        "category": "presenting",
        "description": "Right shoulder moves the upper arm forward, right elbow extends, and the right hand reaches outward in front of the robot.",
        "best_for": "Explaining, introducing a topic, inviting attention, or offering help. Can combine with leftHandReachOut for two-handed presenting.",
        "caution": "Keep duration moderate so it does not look frozen.",
    },
    "leftHandReachOut": {
        "category": "presenting",
        "description": "Left shoulder moves the upper arm forward, left elbow extends, and the left hand reaches outward in front of the robot.",
        "best_for": "Explaining, introducing a topic, inviting attention, or offering help. Can combine with rightHandReachOut for two-handed presenting.",
        "caution": "Keep duration moderate so it does not look frozen.",
    },
    "spreadHands": {
        "category": "presenting",
        "description": "Both shoulders open the upper arms outward, both elbows stay partly bent, and the hands separate to the left and right sides.",
        "best_for": "Overview, comparison, welcoming a group, broad explanations, or 'on the one hand'.",
        "caution": "Large gesture; use sparingly for emphasis.",
    },
    "rightArmLiftFlat": {
        "category": "presenting",
        "description": "Right shoulder lifts the arm forward and upward while the right elbow stays extended, making the right arm look straighter and flatter.",
        "best_for": "Directing attention, showing a place or option, or emphasizing a point.",
        "caution": "Can look strong; avoid for gentle small talk.",
    },
    "leftArmLiftFlat": {
        "category": "presenting",
        "description": "Left shoulder lifts the arm forward and upward while the left elbow stays extended, making the left arm look straighter and flatter.",
        "best_for": "Directing attention, showing a place or option, or emphasizing a point.",
        "caution": "Can look strong; avoid for gentle small talk.",
    },
    "rightArmStretchAndRaise": {
        "category": "presenting",
        "description": "Right shoulder rotates and raises the upper arm outward while the right elbow stays extended, creating a large stretched raised right arm.",
        "best_for": "Welcoming, showcasing, or emphasizing an important point.",
        "caution": "Very expressive; do not use for every explanation.",
    },
    "leftShoulderYawOut": {
        "category": "micro_presenting",
        "description": "Left shoulder yaws the upper arm outward a small amount while the rest of the left arm mostly keeps its current shape.",
        "best_for": "Subtle body-language variation during longer speech. Can combine with rightShoulderYawOut for balanced opening.",
        "caution": "Do not use alone as the whole response.",
    },
    "rightShoulderYawOut": {
        "category": "micro_presenting",
        "description": "Right shoulder yaws the upper arm outward a small amount while the rest of the right arm mostly keeps its current shape.",
        "best_for": "Subtle body-language variation during longer speech. Can combine with leftShoulderYawOut for balanced opening.",
        "caution": "Do not use alone as the whole response.",
    },
    "rightForearmLiftSmall": {
        "category": "micro_presenting",
        "description": "Right elbow bends a small amount so the right forearm lifts slightly while the upper arm stays mostly in place.",
        "best_for": "Natural rhythm while speaking or lightly emphasizing a phrase. Can combine with leftForearmLiftSmall for balanced emphasis.",
        "caution": "Pair with a lowering or standby motion.",
    },
    "rightForearmLiftLarge": {
        "category": "micro_presenting",
        "description": "Right elbow bends more strongly so the right forearm rises higher while the upper arm stays mostly in place.",
        "best_for": "Moderate emphasis in an explanation.",
        "caution": "Avoid for quiet or sensitive content.",
    },
    "rightForearmLowerSmall": {
        "category": "micro_presenting",
        "description": "Right elbow relaxes a small amount so the right forearm lowers slightly from a lifted pose.",
        "best_for": "Recovering from a right forearm lift.",
        "caution": "Usually follows a right forearm lift.",
    },
    "rightForearmLowerLarge": {
        "category": "micro_presenting",
        "description": "Right elbow relaxes more strongly so the right forearm lowers from a higher lifted pose.",
        "best_for": "Recovering from a stronger right forearm lift.",
        "caution": "Usually follows rightForearmLiftLarge.",
    },
    "leftForearmLiftSmall": {
        "category": "micro_presenting",
        "description": "Left elbow bends a small amount so the left forearm lifts slightly while the upper arm stays mostly in place.",
        "best_for": "Natural rhythm while speaking or lightly emphasizing a phrase. Can combine with rightForearmLiftSmall for balanced emphasis.",
        "caution": "Pair with a lowering or standby motion.",
    },
    "leftForearmLiftLarge": {
        "category": "micro_presenting",
        "description": "Left elbow bends more strongly so the left forearm rises higher while the upper arm stays mostly in place.",
        "best_for": "Moderate emphasis in an explanation.",
        "caution": "Avoid for quiet or sensitive content.",
    },
    "leftForearmLowerSmall": {
        "category": "micro_presenting",
        "description": "Left elbow relaxes a small amount so the left forearm lowers slightly from a lifted pose.",
        "best_for": "Recovering from a left forearm lift.",
        "caution": "Usually follows a left forearm lift.",
    },
    "leftForearmLowerLarge": {
        "category": "micro_presenting",
        "description": "Left elbow relaxes more strongly so the left forearm lowers from a higher lifted pose.",
        "best_for": "Recovering from a stronger left forearm lift.",
        "caution": "Usually follows leftForearmLiftLarge.",
    },
    "eyesClose": {
        "category": "attention",
        "description": "The controlled eye or gimbal actuators move into a brief closed-eye or blink-like facial pose, if mapped on the robot.",
        "best_for": "Soft pause, listening, calm acknowledgment, or reflective beat.",
        "caution": "May be subtle or unsupported by the current sender mapping.",
    },
}


SEQUENCE_PATTERNS = {
    "right_hand_wave": ["rightHandRaise", "rightHandWaveLeft", "rightHandWaveRight", "standby"],
    "left_hand_wave": ["leftHandRaise", "leftHandWaveRight", "leftHandWaveLeft", "standby"],
    "two_hand_greeting": ["leftHandRaise+rightHandRaise", "standby"],
    "surrender_or_hands_up": ["leftHandRaise+rightHandRaise", "standby"],
    "positive": ["rightThumbUp", "standby"],
    "strong_positive": ["leftThumbUp+rightThumbUp", "standby"],
    "thinking": ["idea", "standby"],
    "short_explanation": ["leftHandReachOut+rightHandReachOut", "standby"],
    "broad_explanation": ["leftHandReachOut+rightHandReachOut", "spreadHands", "standby"],
    "subtle_speaking_beats": ["rightForearmLiftSmall", "rightForearmLowerSmall", "standby"],
    "balanced_speaking_beats": ["leftForearmLiftSmall+rightForearmLiftSmall", "leftForearmLowerSmall+rightForearmLowerSmall", "standby"],
}


def motion_catalog_text() -> str:
    """Return an LLM-readable catalog of executable atomic motion keyframes."""
    lines = []
    for name in MOTIONS:
        meta = MOTION_DESCRIPTIONS.get(name, {})
        category = meta.get("category", "uncategorized")
        description = meta.get("description", "No description available.")
        best_for = meta.get("best_for", "")
        caution = meta.get("caution", "")
        line = f"- {name} [{category}]: Physical pose: {description}"
        if best_for:
            line += f" Usage hints: {best_for}"
        if caution:
            line += f" Safety/transition: {caution}"
        lines.append(line)
    return "\n".join(lines)


def compact_motion_catalog_text() -> str:
    """Return a shorter keyframe catalog for low-latency prompting."""
    lines = []
    for name in MOTIONS:
        meta = MOTION_DESCRIPTIONS.get(name, {})
        category = meta.get("category", "uncategorized")
        description = meta.get("description", "No description available.")
        lines.append(f"- {name} [{category}]: {description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Motion presets (degrees). Only specify joints that differ from 0.
# ---------------------------------------------------------------------------

MOTIONS = {  # Primitive executable keyframes. Compose simultaneous poses with A+B in the planner/sender.
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
        # "NeckRotation": 0,
        "UpperGimbalLeft": 0,
        "UpperGimbalRight": 0,
        "LowerGimbalLeft": 0,
        "LowerGimbalRight": 0,
    },
    "stay": {},
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
        "RightShoulderPitch": -26,
        "RightElbowPitch": 117,
        "RightMiddleFinger": 132,
        "RightRingFinger": 136,
        "RightPinkyFinger": -75
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
    "peaceSign": {
        "RightShoulderPitch": -31,
        "RightShoulderYaw": -12,
        "RightElbowPitch": 119,
        "RightElbowYaw": -113,
        "RightThumbFinger": -75,
        "RightRingFinger": -136,
        "RightPinkyFinger": 75,
  
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
        "RightShoulderRoll": 90,
        "RightShoulderYaw": -66,
        "RightElbowPitch": -127,
        "RightElbowYaw": 123,
    },
    "rightForearmLiftSmall":{
        "RightElbowPitch": 40,
    },
    "rightForearmLiftLarge":{
        "RightElbowPitch": 90,
    },
    "rightForearmLowerSmall":{
        "RightElbowPitch": -40,
    },
    "rightForearmLowerLarge":{
        "RightElbowPitch": -90,
    },
    "leftForearmLiftSmall":{
        "LeftElbowPitch": -40,
    },
    "leftForearmLiftLarge":{
        "LeftElbowPitch": -90,
    },
    "leftForearmLowerSmall":{
        "LeftElbowPitch": 40,
    },
    "leftForearmLowerLarge":{
        "LeftElbowPitch": 90,
    },
    "eyesClose":{
        "UpperLidLeft": -40,
        "UpperLidRight": 40,
    },
}
