from pathlib import Path

# ------------------- PATH CONSTANTS -------------------
constants_path = Path(__file__)
SRC_PATH = constants_path.parent
PROJECT_PATH = SRC_PATH.parent
DATA_PATH = PROJECT_PATH / "data"
XBD_S12_PATH = DATA_PATH / "xbd_s12"
PROCESSED_PATH = DATA_PATH / "processed"
HYDRA_CONFIG_PATH = SRC_PATH / "configs"
LOGS_PATH = PROJECT_PATH / "logs"

# ------------------- SATELLITE CONSTANTS -------------------
S1_BANDS = ["VV", "VH"]
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

# ------------------- PROJECT CONSTANTS -------------------
# Event-based split from Hafner et al., 2025
TRAIN_DISASTERS = [
    "lower-puna-volcano",
    "palu-tsunami",
    "mexico-earthquake",
    "socal-fire",
    "woolsey-fire",
    "portugal-wildfire",
    "pinery-bushfire",
    "midwest-flooding",
    "moore-tornado",  # not in xBD-S12
    "joplin-tornado",  # not in xBD-S12
    "hurricane-florence",
    "hurricane-harvey",
    "hurricane-michael",
]
TEST_DISASTERS = [
    "nepal-flooding",
    "tuscaloosa-tornado",  # not in xBD-S12
    "guatemala-volcano",
    "sunda-tsunami",
    "santa-rosa-wildfire",
    "hurricane-matthew",
]
ALL_DISASTERS = sorted(TRAIN_DISASTERS + TEST_DISASTERS)


CLASSES = {
    0: "Background",
    1: "Intact",
    2: "Damaged",
    99: "Masked",  # includes unclassified and no-data
}
CLASSES_ORIGINAL = {
    0: "Background",
    1: "Intact",
    2: "Minor Damage",
    3: "Major Damage",
    4: "Destroyed",
    5: "Unclassified",
    6: "No Data",
}

# ------------------- VISUALIZATION CONSTANTS -------------------
COLORS = {
    0: "#eacfb8",  # Background, light brown
    1: "#3976af",  # Intact, blue
    2: "#c73a31",  # Damaged, red
    99: "#000000",  # No Data, black
}

COLORS_ORIGINAL = {
    0: "#eacfb8",  # Background, light brown
    1: "#3976af",  # Intact, blue
    2: "#f08535",  # Minor Damage, orange
    3: "#509d3d",  # Major Damage, green
    4: "#8B5A9B",  # Destroyed, purple
    5: "#000000",  # Unclassified, black
    6: "#000000",  # No Data, black
}
