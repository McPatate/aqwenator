//! Stop strings over streamed text.

/// Finds the first stop string in text that arrives in pieces. Text is released as soon as
/// it can no longer be part of a stop string, so nothing at or past a stop is ever emitted.
pub struct StopText {
    stops: Vec<String>,
    held: String,
}

impl StopText {
    pub fn new(stops: Vec<String>) -> Self {
        Self { stops: stops.into_iter().filter(|s| !s.is_empty()).collect(), held: String::new() }
    }

    /// Add a piece. Returns the text that may be emitted now and the stop string that was
    /// reached, if one was (the text before it is returned, the rest dropped).
    pub fn push(&mut self, piece: &str) -> (String, Option<String>) {
        if self.stops.is_empty() {
            return (piece.to_string(), None);
        }
        self.held.push_str(piece);
        // Earliest match; at the same position, the stop listed first.
        if let Some((at, stop)) = self.stops.iter().filter_map(|s| self.held.find(s.as_str()).map(|at| (at, s))).min_by_key(|x| x.0) {
            let stop = stop.clone();
            self.held.truncate(at);
            return (std::mem::take(&mut self.held), Some(stop));
        }
        // Keep back the longest tail that a stop string starts with.
        let keep = self
            .held
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| self.stops.iter().any(|s| s.starts_with(&self.held[i..])))
            .unwrap_or(self.held.len());
        let tail = self.held.split_off(keep);
        (std::mem::replace(&mut self.held, tail), None)
    }

    /// The generation ended without a stop string: release what was kept back.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], pieces: &[&str]) -> (String, Option<String>) {
        let mut s = StopText::new(stops.iter().map(|x| x.to_string()).collect());
        let mut out = String::new();
        for p in pieces {
            let (text, hit) = s.push(p);
            out.push_str(&text);
            if hit.is_some() {
                return (out, hit);
            }
        }
        out.push_str(&s.flush());
        (out, None)
    }

    #[test]
    fn stop_inside_one_piece() {
        assert_eq!(run(&["6"], &["1, 2, ", "5, 6, 7"]), ("1, 2, 5, ".into(), Some("6".into())));
    }

    #[test]
    fn stop_across_pieces_is_never_emitted() {
        // "END" arrives as "E", "N", "D": the "E" and "N" must be held, not streamed.
        let mut s = StopText::new(vec!["END".into()]);
        assert_eq!(s.push("abc E"), ("abc ".into(), None));
        assert_eq!(s.push("N"), ("".into(), None));
        assert_eq!(s.push("D more"), ("".into(), Some("END".into())));
    }

    #[test]
    fn a_false_start_is_released() {
        assert_eq!(run(&["END"], &["an E", "Nx", " tail"]), ("an ENx tail".into(), None));
        assert_eq!(run(&["END"], &["ends with EN"]), ("ends with EN".into(), None));
    }

    #[test]
    fn earliest_of_several_and_multibyte_text() {
        assert_eq!(run(&["\n\n", "é!"], &["caf", "é", "!", "\n\n"]), ("caf".into(), Some("é!".into())));
        assert_eq!(run(&["b", "ab"], &["xab"]), ("x".into(), Some("ab".into())));
        assert_eq!(run(&[], &["a", "b"]), ("ab".into(), None));
    }
}
