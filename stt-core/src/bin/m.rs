use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Decoder {}

#[wasm_bindgen]
impl Decoder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Decoder {
        Decoder {}
    }

    #[wasm_bindgen]
    pub fn decode(&self, _wav_input: Vec<u8>) -> String {
        // Placeholder: return dummy JSON
        "{\"output\": [{\"dr\": {\"text\": \"Hello from WASM!\"}}]}".to_string()
    }
}

#[wasm_bindgen]
pub fn greet() -> String {
    "Candle WASM Whisper loaded".to_string()
}

fn main() {}
