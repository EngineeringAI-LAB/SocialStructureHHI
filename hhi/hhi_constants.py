"""
hhi_constants.py
-----------------
全局常量和默认超参数。禁止放任何 runtime 逻辑或模型代码。
"""

# ── Duration Buckets ──────────────────────────────────────────────────────────
DURATION_BUCKETS = [30, 60, 90, 120, 150, 180, 240, 300]   # 合法 phase 时长集合（帧）

# ── Schema ────────────────────────────────────────────────────────────────────
MAX_SLOTS = 4                            # schema slot 数（batch collation 需要定长）

# ── Overlap / History Window ──────────────────────────────────────────────────
OVERLAP_N = 10      # phase 间 overlap 帧数；用于 prefix inpainting
W_PREV    = 10      # partner history 窗口长度；显式等于 OVERLAP_N，禁止各处独立硬编码
assert W_PREV == OVERLAP_N, "W_PREV must equal OVERLAP_N"

# ── Hysteresis Gating ─────────────────────────────────────────────────────────
ENTRY_DIST               = 1.0   # 米；穿越此距离进入 INTERACTION_MODE
EXIT_DIST                = 1.8   # 米；穿越此距离且无接触才退出 INTERACTION_MODE
EXIT_CONTACT_FREE_FRAMES = 30    # 连续无接触帧数阈值（30fps ≈ 1 秒）

# ── Contact Detection ─────────────────────────────────────────────────────────
CONTACT_DIST_THRESH   = 0.15   # 米；关节对距离低于此值视为接触
MIN_CONTACT_FRAMES    = 5      # 去抖：连续接触帧数下限
REACH_WINDUP_FRAMES   = 8      # 前摇保护：接触段起始前清零的帧数

# ── Five-Phase Segmentation ────────────────────────────────────────────────────
REACH_DIST_THRESH       = 0.8    # 米；手腕到对方上身距离低于此值 → 进入 reach 状态
RELEASE_DIST_THRESH     = 0.6    # 米；接触结束后手腕离开对方上身超过此值 → 进入 step_back
MIN_PHASE_FRAMES        = 15     # 最小 phase 帧数；更短的段合并到相邻 phase
NON_CONTACT_WRIST_SPEED = 0.3    # m/s；手腕平均速度高于此值视为有意义的非接触互动
                                 # （Wave、Chat、Imitate 等），低于此值维持 approach

# ── Y-Grounding ───────────────────────────────────────────────────────────────
GROUND_Y         = 0.0    # 世界地面 Y 坐标基准（米）
AIRBORNE_THRESH  = 0.08   # 米；双脚最低关节高于此值认为腾空
JUMP_HOLD_FRAMES = 3      # 腾空状态最小持续帧数（防止单帧误判）

# ── Foot Joints (用于 grounding 检测) ─────────────────────────────────────────
FOOT_JOINT_NAMES = ["LeftAnkle", "RightAnkle", "LeftToe", "RightToe"]

# ── Closed-loop Repair Thresholds ─────────────────────────────────────────────
REPAIR_CONTACT_MISMATCH_FRAMES = 8     # 连续接触失配帧数触发 repair
REPAIR_CONTACT_SUCCESS_RATE    = 0.6   # 接触成功率低于此值触发 repair
REPAIR_DIST_DEVIATION_M        = 0.35  # 双人距离偏差（米）持续 10 帧触发
REPAIR_DIST_HOLD_FRAMES        = 10    # 距离偏差持续帧数
REPAIR_CUMULATIVE_DRIFT_M      = 0.60  # 累计漂移量（米）触发 repair

# ── Geometric Loss Joint Pairs (SMPL 22-joint layout) ─────────────────────────
# Contact pairs: self wrist joints ↔ partner upper-body joints (same convention as
# the contact detection in hhi_dataset_builder._detect_contact_from_joints)
CONTACT_JOINT_PAIRS = [
    (20, 20), (20, 21),           # L_WRIST ↔ partner wrists
    (21, 20), (21, 21),           # R_WRIST ↔ partner wrists
    (20, 16), (20, 17),           # L_WRIST ↔ partner shoulders
    (21, 16), (21, 17),           # R_WRIST ↔ partner shoulders
    (20, 12), (21, 12),           # wrists   ↔ partner neck
]

# Bone pairs: SMPL kinematic chain (parent_idx, child_idx)
BODY_BONE_PAIRS = [
    (0,  1), (0,  2),             # pelvis → hips
    (1,  4), (2,  5),             # hip → knee
    (4,  7), (5,  8),             # knee → ankle
    (7, 10), (8, 11),             # ankle → toe
    (0,  3), (3,  6), (6,  9),   # spine chain
    (9, 12), (12, 15),            # neck → head
    (9, 13), (9, 14),             # spine3 → collars
    (13, 16), (14, 17),           # collar → shoulder
    (16, 18), (17, 19),           # shoulder → elbow
    (18, 20), (19, 21),           # elbow → wrist
]

# ── Dataset Split Paths ───────────────────────────────────────────────────────
INTERX_SPLIT_ROOT        = "/scratch3/wan451/3DBody/Inter-X/Inter-X_Dataset/splits"
INTERX_H5_PATH           = "/scratch3/wan451/3DBody/Inter-X/Inter-X_Dataset/processed/inter-x_regen.h5"
INTERX_TEXT_DIR          = "/scratch3/wan451/3DBody/Inter-X/Inter-X_Dataset/texts"
INTERX_ACTION_SETTING    = "/scratch3/wan451/3DBody/Inter-X/Inter-X_Dataset/annots/action_setting.txt"

# ── Inter-X Action Names (0-indexed, matches A000–A039 in sequence IDs) ───────
INTERX_ACTION_NAMES = [
    "Hug",                  # A000
    "Handshake",            # A001
    "Wave",                 # A002
    "Grab",                 # A003
    "Hit",                  # A004
    "Kick",                 # A005
    "Posing",               # A006
    "Push",                 # A007
    "Pull",                 # A008
    "Sit on leg",           # A009
    "Slap",                 # A010
    "Pat on back",          # A011
    "Point finger at",      # A012
    "Walk towards",         # A013
    "Knock over",           # A014
    "Step on foot",         # A015
    "High-five",            # A016
    "Chase",                # A017
    "Whisper in ear",       # A018
    "Support with hand",    # A019
    "Rock-paper-scissors",  # A020
    "Dance",                # A021
    "Link arms",            # A022
    "Shoulder to shoulder", # A023
    "Bend",                 # A024
    "Carry on back",        # A025
    "Massaging shoulder",   # A026
    "Massaging leg",        # A027
    "Hand wrestling",       # A028
    "Chat",                 # A029
    "Pat on cheek",         # A030
    "Thumb up",             # A031
    "Touch head",           # A032
    "Imitate",              # A033
    "Kiss on cheek",        # A034
    "Help up",              # A035
    "Cover mouth",          # A036
    "Look back",            # A037
    "Block",                # A038
    "Fly kiss",             # A039
]
