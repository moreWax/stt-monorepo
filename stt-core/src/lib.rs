
// lib.rs can be empty or re-export greet for test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet() {
        assert_eq!(crate::greet(), "Candle WASM Whisper loaded");
    }
}
