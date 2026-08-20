//! A fixed size ring buffer of samples addressed by *absolute* sample index.
//!
//! The de-esser needs to look back a little (it repairs audio that it has
//! already seen) and to hold audio until it is certain that nothing will
//! modify it any more.  Addressing by absolute index keeps the bookkeeping
//! readable: positions of clicks, events and groups are all plain sample
//! counters from the beginning of the file, and this buffer simply refuses to
//! answer questions about samples that have already been dropped.
//!
//! The capacity is fixed at construction, so memory use is independent of the
//! length of the input.

pub struct SampleRing {
    data: Vec<f32>,
    /// Absolute index that the next pushed sample will get.
    write_index: u64,
}

impl SampleRing {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        SampleRing {
            data: vec![0.0; capacity],
            write_index: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Absolute index of the sample that will be written next, i.e. one past
    /// the newest sample currently stored.
    pub fn write_index(&self) -> u64 {
        self.write_index
    }

    /// Absolute index of the oldest sample still available.
    pub fn oldest_index(&self) -> u64 {
        self.write_index
            .saturating_sub(self.data.len() as u64)
    }

    /// True if `index` is still stored.
    pub fn contains(&self, index: u64) -> bool {
        index >= self.oldest_index() && index < self.write_index
    }

    /// Appends a sample, overwriting the oldest one.
    ///
    /// The caller is responsible for having consumed anything it still needs;
    /// see `DeEsser::next`, which checks this before every push.
    pub fn push(&mut self, sample: f32) {
        let slot = self.slot(self.write_index);
        self.data[slot] = sample;
        self.write_index += 1;
    }

    pub fn get(&self, index: u64) -> f32 {
        debug_assert!(self.contains(index), "sample {index} is no longer buffered");
        self.data[self.slot(index)]
    }

    /// Adds `delta` to a sample that is still buffered.  This is how the
    /// repair stage applies its correction.
    pub fn add(&mut self, index: u64, delta: f32) {
        debug_assert!(self.contains(index), "sample {index} is no longer buffered");
        let slot = self.slot(index);
        self.data[slot] += delta;
    }

    fn slot(&self, index: u64) -> usize {
        (index % self.data.len() as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_most_recent_samples() {
        let mut ring = SampleRing::new(4);
        for i in 0..6 {
            ring.push(i as f32);
        }
        assert_eq!(ring.write_index(), 6);
        assert_eq!(ring.oldest_index(), 2);
        assert!(!ring.contains(1));
        assert_eq!(ring.get(2), 2.0);
        assert_eq!(ring.get(5), 5.0);
    }

    #[test]
    fn add_modifies_in_place() {
        let mut ring = SampleRing::new(4);
        ring.push(1.0);
        ring.add(0, 0.5);
        assert_eq!(ring.get(0), 1.5);
    }
}
