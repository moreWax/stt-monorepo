// This will import the WASM/JS bundle and run inference
import init, { Decoder } from './build/candle_whisper_wasm.js';
import { MODELS, MEL_FILTERS_URL } from './model-config.js';
import { decodeAudioToPCM } from './audio-utils.js';


let decoder;
let ready = false;
let modelAssets = {};

async function fetchAsset(url) {
  const resp = await fetch(url);
  return new Uint8Array(await resp.arrayBuffer());
}

async function ensureReady(modelID = 'tiny_en') {
  if (!ready) {
    await init();
    const model = MODELS[modelID];
    // Download model assets
    const [weights, tokenizer, config, melFilters] = await Promise.all([
      fetchAsset(model.base_url + model.model),
      fetchAsset(model.base_url + model.tokenizer),
      fetchAsset(model.base_url + model.config),
      fetchAsset(MEL_FILTERS_URL)
    ]);
    modelAssets = { weights, tokenizer, config, melFilters };
    // Determine quantized, multilingual, etc.
    const quantized = modelID.includes('quantized');
    const is_multilingual = modelID.includes('multilingual');
    const timestamps = true;
    const task = null;
    const language = null;
    decoder = new Decoder(
      weights,
      tokenizer,
      melFilters,
      config,
      quantized,
      is_multilingual,
      timestamps,
      task,
      language
    );
    ready = true;
  }
}

onmessage = async (e) => {
  const { audioBuffer, modelID, requestId } = e.data;
  try {
    postMessage({ status: 'decoding', message: 'Decoding audio...', requestId });
    await ensureReady(modelID);
    // Decode audio to PCM Float32Array (16kHz mono)
    const pcm = await decodeAudioToPCM(audioBuffer, 16000);
    // Convert Float32Array PCM to Int16 WAV bytes for the model
    const wavBytes = floatTo16BitPCM(pcm);
    postMessage({ status: 'inference', message: 'Running inference...', requestId });
    const jsonResult = decoder.decode(wavBytes);
    const segments = JSON.parse(jsonResult);
    postMessage({ status: 'complete', output: segments, requestId });
  } catch (err) {
    postMessage({ error: err.message || String(err), requestId });
  }
};

// Helper: Convert Float32Array PCM to Int16 WAV bytes (raw PCM, not full WAV header)
function floatTo16BitPCM(float32Array) {
  const int16Array = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return new Uint8Array(int16Array.buffer);
}
