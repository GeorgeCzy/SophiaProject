"""
Motion repository: preset joint configurations for the robot.
All angles in degrees. Unspecified joints remain where they are.

MOTION_DESCRIPTIONS is intentionally written for LLM motion planning. Keep it
short, physical, and semantic: the planner should understand what a keyframe
looks like and when it is appropriate, without reading joint angles.
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
        "description": "Return both arms and hands to the default neutral pose.",
        "best_for": "Ending almost every gesture sequence cleanly.",
        "caution": "Use as the final keyframe, not as the main expressive gesture.",
    },
    "stay": {
        "category": "neutral",
        "description": "Hold the current pose without moving.",
        "best_for": "Brief pauses or intentionally holding a pose.",
        "caution": "Do not use as the final keyframe unless holding is explicitly desired.",
    },
    "leftThumbUp": {
        "category": "positive",
        "description": "Left-hand thumbs-up approval gesture.",
        "best_for": "Agreement, encouragement, success, praise, or confidence.",
        "caution": "Avoid for sad, serious, or factual neutral speech.",
    },
    "rightThumbUp": {
        "category": "positive",
        "description": "Right-hand thumbs-up approval gesture.",
        "best_for": "Agreement, encouragement, success, praise, or confidence.",
        "caution": "Avoid for sad, serious, or factual neutral speech.",
    },
    "bothThumbUp": {
        "category": "positive",
        "description": "Both hands make a clear thumbs-up approval gesture.",
        "best_for": "Strong agreement, celebration, encouragement, or confident success.",
        "caution": "Very expressive; avoid for neutral, sad, or serious speech.",
    },
    "peaceSign": {
        "category": "positive",
        "description": "Right hand makes a playful peace/victory sign near the body.",
        "best_for": "Lighthearted positivity, celebration, friendly photos, or upbeat endings.",
        "caution": "Too playful for solemn content.",
    },
    "rightHandRaise": {
        "category": "greeting",
        "description": "Raise the right hand into a wave-ready pose.",
        "best_for": "Starting a right-hand hello, goodbye, or welcome wave.",
        "caution": "Usually follow with rightHandWaveLeft/rightHandWaveRight or standby.",
    },
    "rightHandWaveRight": {
        "category": "greeting",
        "description": "Right raised hand swings to the robot's right side.",
        "best_for": "One beat in a right-hand wave sequence.",
        "caution": "Use after rightHandRaise or between wave beats.",
    },
    "rightHandWaveLeft": {
        "category": "greeting",
        "description": "Right raised hand swings to the robot's left side.",
        "best_for": "One beat in a right-hand wave sequence.",
        "caution": "Use after rightHandRaise or between wave beats.",
    },
    "leftHandRaise": {
        "category": "greeting",
        "description": "Raise the left hand into a wave-ready pose.",
        "best_for": "Starting a left-hand hello, goodbye, or welcome wave.",
        "caution": "Usually follow with leftHandWaveLeft/leftHandWaveRight or standby.",
    },
    "leftHandWaveRight": {
        "category": "greeting",
        "description": "Left raised hand swings toward the robot's right side.",
        "best_for": "One beat in a left-hand wave sequence.",
        "caution": "Use after leftHandRaise or between wave beats.",
    },
    "leftHandWaveLeft": {
        "category": "greeting",
        "description": "Left raised hand swings toward the robot's left side.",
        "best_for": "One beat in a left-hand wave sequence.",
        "caution": "Use after leftHandRaise or between wave beats.",
    },
    "bothHandsRaise": {
        "category": "greeting",
        "description": "Both hands raise together into an open greeting or attention pose.",
        "best_for": "Warm welcome, greeting a group, cheerful hello, or inviting attention.",
        "caution": "Large two-arm gesture; use briefly and usually return to standby.",
    },
    "idea": {
        "category": "thinking",
        "description": "Right arm bends into a compact thinking or insight pose.",
        "best_for": "Ideas, reasoning, hesitation, checking, or 'let me think' moments.",
        "caution": "Avoid repeating too often in one sequence.",
    },
    "rightHandReachOut": {
        "category": "presenting",
        "description": "Right hand reaches forward as if offering or presenting information.",
        "best_for": "Explaining, introducing a topic, inviting attention, or offering help.",
        "caution": "Keep duration moderate so it does not look frozen.",
    },
    "leftHandReachOut": {
        "category": "presenting",
        "description": "Left hand reaches forward as if offering or presenting information.",
        "best_for": "Explaining, introducing a topic, inviting attention, or offering help.",
        "caution": "Keep duration moderate so it does not look frozen.",
    },
    "bothHandsReachOut": {
        "category": "presenting",
        "description": "Both hands reach forward together as if offering or presenting information.",
        "best_for": "Warm explanations, invitations, presenting an idea, or engaging the listener.",
        "caution": "Use briefly; it is more expressive than a single-hand reach.",
    },
    "spreadHands": {
        "category": "presenting",
        "description": "Both arms open outward in a broad explanatory gesture.",
        "best_for": "Overview, comparison, welcoming a group, broad explanations, or 'on the one hand'.",
        "caution": "Large gesture; use sparingly for emphasis.",
    },
    "rightArmLiftFlat": {
        "category": "presenting",
        "description": "Right arm extends/lifts flatter, like pointing attention to something.",
        "best_for": "Directing attention, showing a place or option, or emphasizing a point.",
        "caution": "Can look strong; avoid for gentle small talk.",
    },
    "leftArmLiftFlat": {
        "category": "presenting",
        "description": "Left arm extends/lifts flatter, like pointing attention to something.",
        "best_for": "Directing attention, showing a place or option, or emphasizing a point.",
        "caution": "Can look strong; avoid for gentle small talk.",
    },
    "rightArmStretchAndRaise": {
        "category": "presenting",
        "description": "Right arm stretches outward and raises, a large attention-guiding gesture.",
        "best_for": "Welcoming, showcasing, or emphasizing an important point.",
        "caution": "Very expressive; do not use for every explanation.",
    },
    "leftShoulderYawOut": {
        "category": "micro_presenting",
        "description": "Small left upper-arm outward adjustment.",
        "best_for": "Subtle body language during longer speech.",
        "caution": "Do not use alone as the whole response.",
    },
    "rightShoulderYawOut": {
        "category": "micro_presenting",
        "description": "Small right upper-arm outward adjustment.",
        "best_for": "Subtle body language during longer speech.",
        "caution": "Do not use alone as the whole response.",
    },
    "rightForearmLiftSmall": {
        "category": "micro_presenting",
        "description": "Small right forearm lift, like a mild conversational beat.",
        "best_for": "Natural rhythm while speaking or lightly emphasizing a phrase.",
        "caution": "Pair with a lowering or standby motion.",
    },
    "rightForearmLiftLarge": {
        "category": "micro_presenting",
        "description": "Larger right forearm lift for stronger emphasis.",
        "best_for": "Moderate emphasis in an explanation.",
        "caution": "Avoid for quiet or sensitive content.",
    },
    "rightForearmLowerSmall": {
        "category": "micro_presenting",
        "description": "Small right forearm lowering motion.",
        "best_for": "Recovering from a right forearm lift.",
        "caution": "Usually follows a right forearm lift.",
    },
    "rightForearmLowerLarge": {
        "category": "micro_presenting",
        "description": "Larger right forearm lowering motion.",
        "best_for": "Recovering from a stronger right forearm lift.",
        "caution": "Usually follows rightForearmLiftLarge.",
    },
    "leftForearmLiftSmall": {
        "category": "micro_presenting",
        "description": "Small left forearm lift, like a mild conversational beat.",
        "best_for": "Natural rhythm while speaking or lightly emphasizing a phrase.",
        "caution": "Pair with a lowering or standby motion.",
    },
    "leftForearmLiftLarge": {
        "category": "micro_presenting",
        "description": "Larger left forearm lift for stronger emphasis.",
        "best_for": "Moderate emphasis in an explanation.",
        "caution": "Avoid for quiet or sensitive content.",
    },
    "leftForearmLowerSmall": {
        "category": "micro_presenting",
        "description": "Small left forearm lowering motion.",
        "best_for": "Recovering from a left forearm lift.",
        "caution": "Usually follows a left forearm lift.",
    },
    "leftForearmLowerLarge": {
        "category": "micro_presenting",
        "description": "Larger left forearm lowering motion.",
        "best_for": "Recovering from a stronger left forearm lift.",
        "caution": "Usually follows leftForearmLiftLarge.",
    },
    "bothForearmsLiftSmall": {
        "category": "micro_presenting",
        "description": "Both forearms lift slightly together as a balanced conversational beat.",
        "best_for": "Light emphasis during explanations, acknowledgments, or short supportive phrases.",
        "caution": "Pair with bothForearmsLowerSmall or standby so the pose resolves.",
    },
    "bothForearmsLiftLarge": {
        "category": "micro_presenting",
        "description": "Both forearms lift more strongly together for a larger emphasis beat.",
        "best_for": "Important points, energetic explanation, or stronger encouragement.",
        "caution": "Avoid for calm or sensitive content.",
    },
    "bothForearmsLowerSmall": {
        "category": "micro_presenting",
        "description": "Both forearms lower slightly together, resolving a small lift.",
        "best_for": "Recovering after bothForearmsLiftSmall.",
        "caution": "Usually follows a forearm lift.",
    },
    "bothForearmsLowerLarge": {
        "category": "micro_presenting",
        "description": "Both forearms lower more strongly together, resolving a large lift.",
        "best_for": "Recovering after bothForearmsLiftLarge.",
        "caution": "Usually follows a stronger forearm lift.",
    },
    "eyesClose": {
        "category": "attention",
        "description": "Brief eye-close or blink-like pause, if supported by the robot mapping.",
        "best_for": "Soft pause, listening, calm acknowledgment, or reflective beat.",
        "caution": "May be subtle or unsupported by the current sender mapping.",
    },
}


SEQUENCE_PATTERNS = {
    "right_hand_wave": ["rightHandRaise", "rightHandWaveLeft", "rightHandWaveRight", "standby"],
    "left_hand_wave": ["leftHandRaise", "leftHandWaveRight", "leftHandWaveLeft", "standby"],
    "two_hand_greeting": ["bothHandsRaise", "standby"],
    "positive": ["rightThumbUp", "standby"],
    "strong_positive": ["bothThumbUp", "standby"],
    "thinking": ["idea", "standby"],
    "short_explanation": ["bothHandsReachOut", "standby"],
    "broad_explanation": ["bothHandsReachOut", "spreadHands", "standby"],
    "subtle_speaking_beats": ["rightForearmLiftSmall", "rightForearmLowerSmall", "standby"],
    "balanced_speaking_beats": ["bothForearmsLiftSmall", "bothForearmsLowerSmall", "standby"],
}


def motion_catalog_text() -> str:
    """Return an LLM-readable catalog of executable motion keyframes."""
    lines = []
    for name in MOTIONS:
        meta = MOTION_DESCRIPTIONS.get(name, {})
        category = meta.get("category", "uncategorized")
        description = meta.get("description", "No description available.")
        best_for = meta.get("best_for", "")
        caution = meta.get("caution", "")
        line = f"- {name} [{category}]: {description}"
        if best_for:
            line += f" Best for: {best_for}"
        if caution:
            line += f" Caution: {caution}"
        lines.append(line)
    return "\n".join(lines)


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
        # "NeckRotation": 0,
        "UpperGimbalLeft": 0,
        "UpperGimbalRight": 0,
        "LowerGimbalLeft": 0,
        "LowerGimbalRight": 0,
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
    "bothThumbUp": {
        "RightShoulderPitch": -71.0,
        "RightThumbFinger": 44.0,
        "RightIndexFinger": 123.0,
        "RightMiddleFinger": -132.0,
        "RightRingFinger": -136.0,
        "RightPinkyFinger": 75.0,
        "RightThumbRoll": 22,
        "LeftShoulderPitch": 71.0,
        "LeftThumbFinger": -44.0,
        "LeftIndexFinger": 123.0,
        "LeftMiddleFinger": 132.0,
        "LeftRingFinger": 136.0,
        "LeftPinkyFinger": 75.0,
        "LeftShoulderRoll": -7.0,
        "LeftThumbRoll": 22,
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
    "bothHandsRaise":{
        "RightShoulderPitch": -70.0,
        "RightShoulderYaw": -13,
        "RightElbowPitch": 108,
        "RightElbowYaw": -123,
        "LeftShoulderPitch": 70.0,
        "LeftShoulderYaw": 13,
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
    "bothHandsReachOut":{
        "RightShoulderPitch": -111,
        "RightElbowPitch": -127,
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
    "bothForearmsLiftSmall":{
        "RightElbowPitch": 40,
        "LeftElbowPitch": -40,
    },
    "bothForearmsLiftLarge":{
        "RightElbowPitch": 90,
        "LeftElbowPitch": -90,
    },
    "bothForearmsLowerSmall":{
        "RightElbowPitch": -40,
        "LeftElbowPitch": 40,
    },
    "bothForearmsLowerLarge":{
        "RightElbowPitch": -90,
        "LeftElbowPitch": 90,
    },
    "eyesClose":{
        "UpperLidLeft": -40,
        "UpperLidRight": 40,
    },
}
