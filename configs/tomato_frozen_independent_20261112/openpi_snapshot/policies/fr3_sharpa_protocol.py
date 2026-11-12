"""Training joint order from franka3-sharpa-pi05: measured state and commanded action, radians."""

SHARPA_JOINTS = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)
ORDER = (
    tuple(f"left_fr3_joint{i}" for i in range(1, 8))
    + tuple(f"left_{name}" for name in SHARPA_JOINTS)
    + tuple(f"right_fr3_joint{i}" for i in range(1, 8))
    + tuple(f"right_{name}" for name in SHARPA_JOINTS)
)
ACTION_DIM = len(ORDER)
# Exported LeRobot observations contain measured positions only, not 116D position/velocity pairs.
STATE_DIM = ACTION_DIM
UNIT = "rad"
