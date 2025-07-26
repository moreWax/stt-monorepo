# Candle Whisper WASM Chrome Extension: Testing & Verification Guide

This guide walks you through testing and verifying your Chrome extension for on-device speech-to-text using Candle + Whisper in WASM. Follow these steps after implementing or refactoring the extension.

---


## 1. Prerequisites
- Ensure you have built the WASM/JS bundle and placed all model assets in `extension/build/`.
- **Run the verification script to check required files:**
  ```sh
  python3 verify_extension_files.py
  ```
- Confirm the following files are up to date with the latest architecture (all are in the `extension/` folder):
  - `manifest.json` (`extension/manifest.json`)
  - `popup.js` (`extension/popup.js`)
  - `background.js` (`extension/background.js`)
  - `whisperWorker.js` (`extension/whisperWorker.js`)

---

## 2. Load the Extension in Chrome
1. Open Chrome and go to `chrome://extensions`.
2. Enable **Developer Mode** (toggle in the top right).
3. Click **Load unpacked** and select the `extension/` directory.

---

## 3. Test the Popup UI
1. Click the extension icon in the Chrome toolbar to open the popup.
2. Click **Start Recording** and speak into your microphone.
3. Click the button again to stop recording.
4. Wait for the transcription to appear in the popup.

---

## 4. Verify Message Passing & Inference
- The popup should send audio to the background service worker.
- The background should instantiate `whisperWorker.js`, forward the audio, and relay the result back.
- The popup should display the transcription or an error message.

---

## 5. Debugging & Troubleshooting
- If you see **no output** or an **error**:
  - Open the Chrome Extensions panel (`chrome://extensions`), click **Service Worker** or **Inspect views** for your extension, and check the console for errors.
  - Common issues:
    - Asset loading errors (check `web_accessible_resources` in `manifest.json`)
    - Service worker suspension (Chrome may restart it; model will re-initialize)
    - Permission errors (ensure `microphone` and `storage` permissions are set)
- If you see 'Unknown response', check that the message protocol matches between popup and background.

---

## 6. Success Criteria
- The extension records audio, sends it to the background, runs WASM inference, and displays the transcription in the popup.
- No errors appear in the popup or background console.

---

## 7. Next Steps
- Once verified, you can proceed to add new features, optimize performance, or prepare for deployment.
- Keep this guide for future reference and onboarding.

---

**Tip:** If you encounter issues, document the error and steps to reproduce. This will help with debugging and future improvements.
