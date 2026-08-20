//! A pass-through filter that reports progress to the user.
//!
//! It is not a filter in the signal processing sense: every sample is handed
//! on untouched.  What it does is count the samples that flow through it and
//! print a line like
//!
//! ```text
//! Progress:  42% ETA: 2m 10s
//! ```
//!
//! updating it in place with backspace characters, so that the user can see
//! that the program is alive and roughly how long it still has to run.
//!
//! Placed in front of the de-esser, it measures how much of the *input* has
//! been read.  The de-esser buffers about a second of look-ahead, so the
//! reported position runs a fraction of a second ahead of what has actually
//! been written - irrelevant for a progress indicator, and it keeps the
//! reporting independent of what the rest of the chain does.

use std::io::{self, IsTerminal, Write};
use std::time::{Duration, Instant};

/// How often the wall clock is consulted, in samples.  Reading the clock for
/// every sample would cost more than the filter itself; 4096 samples is about
/// 90 ms of audio, far finer than the update interval below.
const CLOCK_CHECK_INTERVAL: u64 = 4096;

/// Minimum time between two redraws.  Fast enough to look alive, slow enough
/// not to flood a slow terminal.
const UPDATE_INTERVAL: Duration = Duration::from_millis(200);

pub struct ProgressReport<I> {
    inner: I,
    /// Total number of samples expected, known at construction time.
    total: u64,
    /// Samples seen so far.
    done: u64,
    started: Instant,
    last_update: Instant,
    /// Number of characters currently on screen, so they can be backspaced
    /// over on the next update.
    printed: usize,
    /// False when there is nothing to report to (output is redirected), or
    /// when the caller asked for silence.
    enabled: bool,
}

impl<I> ProgressReport<I> {
    /// `total_samples` is what the report is measured against; pass the length
    /// from the file header.  `quiet` suppresses the output entirely.
    ///
    /// Progress goes to stderr, and only if stderr is a terminal: a redirected
    /// stream would otherwise collect a stream of backspaces, and stdout stays
    /// free for the result summary.
    pub fn new(inner: I, total_samples: u64, quiet: bool) -> Self {
        let now = Instant::now();
        ProgressReport {
            inner,
            total: total_samples,
            done: 0,
            started: now,
            last_update: now,
            printed: 0,
            enabled: !quiet && total_samples > 0 && io::stderr().is_terminal(),
        }
    }

    /// Redraws the line if enough time has passed since the last redraw.
    fn maybe_redraw(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_update) < UPDATE_INTERVAL {
            return;
        }
        self.last_update = now;

        let percent = (100 * self.done / self.total).min(100);
        let elapsed = now.duration_since(self.started).as_secs_f64();
        // Assume the rest of the file takes as long per sample as the part
        // already done, which is a good model here: the work per sample does
        // not depend on the content.
        let remaining = (self.total - self.done) as f64;
        let eta = if self.done == 0 {
            None
        } else {
            Some((elapsed * remaining / self.done as f64).round() as u64)
        };

        let text = match eta {
            Some(seconds) => format!("Progress: {percent:3}% ETA: {}", format_duration(seconds)),
            None => format!("Progress: {percent:3}%"),
        };
        self.draw(&text);
    }

    /// Overwrites the previous line with `text`, using backspaces.
    ///
    /// Failures are ignored on purpose: a progress indicator must never be the
    /// reason a conversion fails.
    fn draw(&mut self, text: &str) {
        let mut stderr = io::stderr().lock();
        let _ = write_backspaces(&mut stderr, self.printed);
        let _ = stderr.write_all(text.as_bytes());
        // If the new text is shorter than the old one, wipe the leftovers and
        // step back over them.
        if self.printed > text.len() {
            let leftover = self.printed - text.len();
            let _ = write_repeated(&mut stderr, b' ', leftover);
            let _ = write_backspaces(&mut stderr, leftover);
        }
        let _ = stderr.flush();
        self.printed = text.len();
    }

    /// Removes the progress line, so that whatever is printed next starts on a
    /// clean line.
    fn clear(&mut self) {
        if self.printed == 0 {
            return;
        }
        let mut stderr = io::stderr().lock();
        let _ = write_backspaces(&mut stderr, self.printed);
        let _ = write_repeated(&mut stderr, b' ', self.printed);
        let _ = write_backspaces(&mut stderr, self.printed);
        let _ = stderr.flush();
        self.printed = 0;
    }
}

impl<I: Iterator<Item = f32>> Iterator for ProgressReport<I> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        match self.inner.next() {
            Some(sample) => {
                self.done += 1;
                if self.enabled && self.done.is_multiple_of(CLOCK_CHECK_INTERVAL) {
                    self.maybe_redraw();
                }
                Some(sample)
            }
            None => {
                // End of the stream: take the line down again.
                if self.enabled {
                    self.clear();
                }
                None
            }
        }
    }
}

/// Formats a number of seconds the way a person reads it: `45s`, `2m 10s`,
/// `1h 05m`.
fn format_duration(seconds: u64) -> String {
    if seconds >= 3600 {
        format!("{}h {:02}m", seconds / 3600, (seconds % 3600) / 60)
    } else if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

fn write_backspaces(out: &mut impl Write, count: usize) -> io::Result<()> {
    write_repeated(out, 0x08, count)
}

/// Writes the same byte `count` times, in chunks, without allocating.
fn write_repeated(out: &mut impl Write, byte: u8, count: usize) -> io::Result<()> {
    let chunk = [byte; 64];
    let mut left = count;
    while left > 0 {
        let now = left.min(chunk.len());
        out.write_all(&chunk[..now])?;
        left -= now;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_pass_through_untouched() {
        let input: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        // `quiet` keeps the test output clean; the sample path is the same.
        let output: Vec<f32> = ProgressReport::new(input.iter().copied(), 1000, true).collect();
        assert_eq!(input, output);
    }

    #[test]
    fn durations_are_formatted_for_humans() {
        assert_eq!(format_duration(0), "0s");
        assert_eq!(format_duration(45), "45s");
        assert_eq!(format_duration(60), "1m 00s");
        assert_eq!(format_duration(130), "2m 10s");
        assert_eq!(format_duration(3600), "1h 00m");
        assert_eq!(format_duration(3900), "1h 05m");
    }

    #[test]
    fn an_empty_stream_is_handled() {
        let empty: Vec<f32> = Vec::new();
        let output: Vec<f32> = ProgressReport::new(empty.into_iter(), 0, false).collect();
        assert!(output.is_empty());
    }
}
