#!/bin/bash
# Build Candle WASM Whisper library for Chrome extension
set -e
cd ..
cargo build --target wasm32-unknown-unknown --release
wasm-bindgen --target web --out-dir extension/build target/wasm32-unknown-unknown/release/candle_whisper_wasm.wasm
