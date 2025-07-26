#!/bin/bash
# Copy the latest WASM build from the submodule to the extension build directory
# Usage: ./sync_wasm_from_submodule.sh

set -e

# Path to submodule WASM build
SUBMODULE_WASM="candle-whisper-wasm/pkg/candle_whisper_wasm_bg.wasm"
# Path to extension build directory
EXT_BUILD_DIR="extension/build"

if [ ! -f "$SUBMODULE_WASM" ]; then
  echo "Error: WASM file not found at $SUBMODULE_WASM"
  exit 1
fi

mkdir -p "$EXT_BUILD_DIR"
cp "$SUBMODULE_WASM" "$EXT_BUILD_DIR/candle_whisper_wasm_bg.wasm"
echo "Copied $SUBMODULE_WASM to $EXT_BUILD_DIR/candle_whisper_wasm_bg.wasm"
ls -lh "$EXT_BUILD_DIR/candle_whisper_wasm_bg.wasm"