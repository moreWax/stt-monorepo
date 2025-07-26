// Model asset URLs for Whisper (tiny.en, quantized, distil-medium)
export const MODELS = {
  tiny_en: {
    base_url: "https://huggingface.co/openai/whisper-tiny.en/resolve/main/",
    model: "model.safetensors",
    tokenizer: "tokenizer.json",
    config: "config.json",
    size: "151 MB"
  },
  tiny_en_quantized_q80: {
    base_url: "https://huggingface.co/lmz/candle-whisper/resolve/main/",
    model: "model-tiny-en-q80.gguf",
    tokenizer: "tokenizer-tiny-en.json",
    config: "config-tiny-en.json",
    size: "41.8 MB"
  },
  distil_medium_en: {
    base_url: "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/",
    model: "model.safetensors",
    tokenizer: "tokenizer.json",
    config: "config.json",
    size: "789 MB"
  }
};
export const MEL_FILTERS_URL = "https://huggingface.co/spaces/lmz/candle-whisper/resolve/main/mel_filters.safetensors";
