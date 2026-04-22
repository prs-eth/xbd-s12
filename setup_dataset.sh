#!/usr/bin/env bash
# setup_dataset.sh
# Generates the xBD-S12 dataset from the original xBD dataset.
# Usage: bash setup_dataset.sh --original_xbd_path=<path> [--num_workers=<n>] [--create_hdf5]
# (Thanks Claude)

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
ORIGINAL_XBD_PATH=""
NUM_WORKERS=8
CREATE_HDF5=false

ZENODO_URL="https://zenodo.org/records/18960454/files/xbd_s12.tar.gz?download=1"
ZENODO_ARCHIVE="data/xbd_s12.tar.gz"
XBD_S12_DIR="data/xbd_s12"

# ── Argument parsing ──────────────────────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        --original_xbd_path=*)
            ORIGINAL_XBD_PATH="${arg#*=}"
            ;;
        --num_workers=*)
            NUM_WORKERS="${arg#*=}"
            ;;
        --create_hdf5)
            CREATE_HDF5=true
            ;;
        -h|--help)
            echo "Usage: bash setup_dataset.sh --original_xbd_path=<path> [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --original_xbd_path=<path>  Path to the original xBD dataset (required)."
            echo "                              Expected to contain: hold/, test/, tier1/, tier3/"
            echo "  --num_workers=<n>           Number of parallel workers (default: 4)"
            echo "  --create_hdf5               Also create the HDF5 file after VRT and mask creation."
            echo ""
            echo "The Sentinel data (xBD-S12) is downloaded automatically from Zenodo"
            echo "if data/xbd_s12 is not found."
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $arg"
            echo "Run with --help for usage."
            exit 1
            ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────

# original_xbd_path is always required
if [[ -z "$ORIGINAL_XBD_PATH" ]]; then
    echo "Error: --original_xbd_path is required."
    echo "Run with --help for usage."
    exit 1
fi

if [[ ! -d "$ORIGINAL_XBD_PATH" ]]; then
    echo "Error: Directory not found: $ORIGINAL_XBD_PATH"
    exit 1
fi

for split in hold test tier1 tier3; do
    if [[ ! -d "$ORIGINAL_XBD_PATH/$split" ]]; then
        echo "Error: Expected split directory not found: $ORIGINAL_XBD_PATH/$split"
        exit 1
    fi
done

# ── Auto-detect whether download is needed ────────────────────────────────────
if [[ ! -d "$XBD_S12_DIR" ]]; then
    echo "[$XBD_S12_DIR] not found — will download from Zenodo automatically."
    DOWNLOAD=true
else
    echo "[$XBD_S12_DIR] found — skipping download."
    DOWNLOAD=false
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo " xBD-S12 Dataset Setup"
echo "============================================"
echo "  download           : $DOWNLOAD"
echo "  original_xbd_path  : $ORIGINAL_XBD_PATH"
echo "  num_workers        : $NUM_WORKERS"
echo "  create_hdf5        : $CREATE_HDF5"
echo "============================================"
echo ""

# ── Step 1: Download from Zenodo (if needed) ──────────────────────────────────
if [[ "$DOWNLOAD" == true ]]; then
    echo "[1/4] Downloading xBD-S12 archive from Zenodo..."

    mkdir -p "$(dirname "$ZENODO_ARCHIVE")"

    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$ZENODO_ARCHIVE" "$ZENODO_URL"
    elif command -v wget &>/dev/null; then
        wget -q --show-progress -O "$ZENODO_ARCHIVE" "$ZENODO_URL"
    else
        echo "Error: neither curl nor wget found. Please install one and retry."
        exit 1
    fi

    echo "[1/4] Extracting archive..."
    tar -xzf "$ZENODO_ARCHIVE" -C "$(dirname "$ZENODO_ARCHIVE")"
    rm "$ZENODO_ARCHIVE"
    echo "[1/4] Done. Dataset extracted to $XBD_S12_DIR"
else
    echo "[1/4] Skipping download — $XBD_S12_DIR already exists."
fi
echo ""

# ── Step 2: Create aligned VRT files ─────────────────────────────────────────
echo "[2/4] Creating aligned VRT files..."
python src/data/create_aligned_vrt.py \
    --original_xbd_path="$ORIGINAL_XBD_PATH" \
    --num_workers="$NUM_WORKERS"
echo "[2/4] Done."
echo ""

# ── Step 3: Create masks ──────────────────────────────────────────────────────
echo "[3/4] Creating masks..."
python src/data/create_masks.py \
    --original_xbd_path="$ORIGINAL_XBD_PATH"
echo "[3/4] Done."
echo ""

# ── Step 4 (optional): Create HDF5 ───────────────────────────────────────────
if [[ "$CREATE_HDF5" == true ]]; then
    echo "[4/4] Creating HDF5 file (this may take a while)..."
    python src/data/create_hdf5.py \
        --num_workers="$NUM_WORKERS"
    echo "[4/4] Done."
else
    echo "[4/4] Skipping HDF5 creation (pass --create_hdf5 to enable)."
fi

echo ""
echo "============================================"
echo " Dataset setup complete."
echo "============================================"