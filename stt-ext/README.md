# Candle Whisper WASM Chrome Extension

This extension provides on-device speech-to-text using Candle (Rust ML) + Whisper (distil/tiny) running in WebAssembly.

## Structure
- `manifest.json`: Chrome extension manifest (MV3)
- `popup.html`, `popup.js`: UI for recording and displaying transcription. Sends audio to background via message passing.
- `background.js`: Service worker that orchestrates WASM inference. Instantiates `whisperWorker.js`, handles messages from popup, forwards audio, and relays results.
- `whisperWorker.js`: WebWorker that loads the WASM/JS bundle and runs inference.
- `build/`: Place Candle WASM/JS output here (see below)

## Build WASM/JS Bundle
1. Build the Candle WASM Whisper library (from the Candle repo):
   ```sh
   sh build-lib.sh
   # or use wasm-pack build --target web
   # Output should be placed in ./build (e.g., m.js, m_bg.wasm)
   ```
2. Copy the output files (`m.js`, `m_bg.wasm`, etc.) into the `extension/build/` directory.

## Model Assets
- By default, the worker expects model weights, tokenizer, config, and mel filters to be loaded at runtime (see TODOs in `whisperWorker.js`).
- For demo, use the smallest model (e.g., `tiny.en` or quantized) and fetch from Hugging Face.


## Development
- Load the `extension/` directory as an unpacked extension in Chrome.
- Click the extension icon, record audio, and view transcription.

## Architecture (2025 Refactor)

**Message Flow:**
1. `popup.js` records audio, sends `{type: 'INFER', audioBuffer, modelID, requestId}` to background via `chrome.runtime.sendMessage`.
2. `background.js` receives message, instantiates `whisperWorker.js` if needed, forwards audio and requestId to worker.
3. `whisperWorker.js` runs inference, posts result (or error) back to background with requestId.
4. `background.js` relays result to popup via `sendResponse`.
5. `popup.js` updates UI with result or error.

**Message Protocol:**
- Request: `{ type: 'INFER', audioBuffer, modelID, requestId }`
- Response: `{ status: 'complete', output, requestId }` or `{ error, requestId }`
- Status updates: `{ status, message, requestId }`

**Concurrency:**
- Each request includes a unique `requestId` to match responses.

**Troubleshooting:**
- If you see 'Unknown response' or no result, check the background service worker logs in Chrome's Extensions panel.
- If model assets fail to load, ensure URLs are correct and files are listed in `web_accessible_resources` in `manifest.json`.
- If the service worker is suspended, Chrome may restart it; the worker and model will be re-initialized as needed.

**worker-proxy.js is now obsolete and removed.**


---

## Step-by-Step: Integrating Candle Whisper WASM in a Chrome Extension

### 1. Set Up the Rust WASM Project
- Ensure your Rust project is set up for WASM:
  - In `Cargo.toml`:
    ```toml
    [lib]
    crate-type = ["cdylib"]

    [dependencies]
    wasm-bindgen = "0.2"
    ```
- Your `src/lib.rs` can be minimal or export test functions.

### 2. Build the WASM Library
- Install the WASM target:
  ```sh
  rustup target add wasm32-unknown-unknown
  ```
- Build the project:
  ```sh
  cargo build --target wasm32-unknown-unknown --release
  ```
- Locate the `.wasm` output, typically in:
  ```
  target/wasm32-unknown-unknown/release/deps/candle_whisper_wasm.wasm
  ```

### 3. Generate JS Bindings with wasm-bindgen
- Install wasm-bindgen CLI if not already:
  ```sh
  cargo install wasm-bindgen-cli
  ```
- Run wasm-bindgen to generate JS/TS glue code:
  ```sh
  wasm-bindgen --target web --out-dir <output_dir> <path_to_wasm>
  ```
  Example:
  ```sh
  wasm-bindgen --target web --out-dir ../extension/build target/wasm32-unknown-unknown/release/deps/candle_whisper_wasm.wasm
  ```

### 4. Prepare Model Assets
- Download model weights, tokenizer, config, and mel filters as needed.
- Store them in a location accessible to your extension (e.g., via URLs or local files).

### 5. Integrate WASM in Your Extension Worker
- In your `whisperWorker.js`:
  - Import the generated JS:
    ```js
    import init, { Decoder } from './build/candle_whisper_wasm.js';
    ```
  - Load model assets (fetch as Uint8Array).
  - Construct the `Decoder`:
    ```js
    decoder = new Decoder(
      weights, tokenizer, melFilters, config,
      quantized, is_multilingual, timestamps, task, language
    );
    ```
  - On message, pass the raw audio buffer to `decode`:
    ```js
    const wavInput = new Uint8Array(audioBuffer);
    const jsonResult = decoder.decode(wavInput);
    const segments = JSON.parse(jsonResult);
    postMessage({ status: 'complete', output: segments });
    ```

### 6. Connect the Worker to Your UI
- In your popup or main UI JS:
  - Send the recorded audio buffer to the worker.
  - Display status updates and the final transcription result.

### 7. Test End-to-End
- Build the WASM and JS bundle.
- Load your extension in Chrome (enable developer mode).
- Record audio and verify transcription output.

---

**Tips:**
- Always match the WASM/JS API to the Candle example (`Decoder` class, `decode` method).
- If you change the Rust code, rebuild and re-run wasm-bindgen.
- Check the browser console for errors if something fails.
