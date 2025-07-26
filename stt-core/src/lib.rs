
// lib.rs can be empty or re-export greet for test

pub fn greet() -> &'static str {
    "Candle WASM Whisper loaded"
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_greet() {
        assert_eq!(crate::greet(), "Candle WASM Whisper loaded");
    }
}
