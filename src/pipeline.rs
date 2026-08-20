//! The place where filters are hooked into the sample stream.
//!
//! Everything between the source and the sink is an ordinary Rust iterator
//! over `f32` samples in the range -1.0 ..= +1.0.  A filter is therefore just
//! an `Iterator<Item = f32>` that wraps another one, and chaining filters is
//! chaining iterators:
//!
//! ```ignore
//! let pipeline = source.progress(total, quiet).deesser();
//! ```
//!
//! The chain stays lazy: no sample is read from the file until the sink asks
//! for one, and each filter only ever holds the state it needs itself.
//!
//! # Adding a filter
//!
//! 1. Write a struct that owns the upstream iterator and implements
//!    `Iterator<Item = f32>`.
//! 2. Give it a method on [`FilterExt`] below.
//!
//! Nothing else in the program needs to change - not the source, not the sink
//! and not the other filters.

use crate::deesser::DeEsser;
use crate::progress::ProgressReport;

/// Extension trait implemented for every stream of samples.
pub trait FilterExt: Iterator<Item = f32> + Sized {
    /// Removes sibilance; see [`crate::deesser`].
    fn deesser(self) -> DeEsser<Self> {
        DeEsser::new(self)
    }

    /// Reports how far the stream has advanced; passes samples through
    /// unchanged.  `total_samples` is the length the progress is measured
    /// against, `quiet` turns the reporting off.
    fn progress(self, total_samples: u64, quiet: bool) -> ProgressReport<Self> {
        ProgressReport::new(self, total_samples, quiet)
    }
}

impl<I: Iterator<Item = f32>> FilterExt for I {}
