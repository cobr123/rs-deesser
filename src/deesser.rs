//! The de-esser itself: an `Iterator<Item = f32>` adapter that sits between
//! the file source and the file sink.
//!
//! # What the algorithm does
//!
//! This is not a compressor with attack and release times.  It works in two
//! stages, "detect events, then repair them":
//!
//! 1. **Detection.**  The band 2500..8000 Hz is split into ten logarithmically
//!    spaced sub bands.  Each sub band is isolated with a band-pass FIR whose
//!    gain at the centre frequency is exactly 1, and the peak magnitude of the
//!    filtered signal is measured over blocks of 10 ms ("steps").  A run of
//!    consecutive steps whose peak exceeds -20 dBFS is a *click*.
//!
//! 2. **Repair.**  Clicks that start in the same step form an *event*; events
//!    that overlap (after both ends are widened by the 5 ms crossfade) form a
//!    *group*.  For every event, the corresponding excerpt of the input is
//!    passed through a cascade of peaking EQ filters - one per sub band that
//!    fired - each cutting exactly as many dB as that sub band exceeded the
//!    threshold.  The difference between the filtered and the original excerpt
//!    is faded in and out with a trapezoid and added to the output.
//!
//! Because every cut is exactly the measured excess, a sibilant is pushed down
//! to the threshold and nothing else is touched.  Two caveats worth knowing:
//! the cut is exact only for material at the centre of a band (a tone halfway
//! between two centres is measured through the skirts of two detectors and
//! repaired by the skirts of two bells, so it stays a few dB above the
//! threshold), and where two events are close enough for their crossfades to
//! overlap the attenuation dips by about a dB.  Both are properties of the
//! original algorithm, and both are covered by tests at the end of this file.
//!
//! # How it stays within a fixed amount of memory
//!
//! Samples flow through a ring buffer.  The detector runs ahead of the output,
//! and a sample is only handed to the sink once no future correction can reach
//! it any more, which is decided by [`DeEsser::safe_limit`].  Two safety
//! limits that the original Nyquist plug-in did not have keep the look-ahead
//! bounded on dense material: a single run is cut after `MAX_RUN_STEPS`, and a
//! group is closed after `MAX_GROUP_SAMPLES`.  Nothing here scales with the
//! length of the input: measured peak memory of the whole program is about
//! 3 MB for a ten second file and for a ten minute file alike.

use std::collections::VecDeque;

use crate::dsp::{design_band_pass, BandPassKernel, Fir, PeakingDesign};
use crate::ring::SampleRing;

// ---------------------------------------------------------------------------
// Fixed configuration
// ---------------------------------------------------------------------------

/// The only sample rate this program supports.
pub const SAMPLE_RATE: f64 = 44_100.0;

/// Lower edge of the analysed frequency range, in Hz.
const DETECT_LOW_HZ: f64 = 2500.0;
/// Upper edge of the analysed frequency range, in Hz.
const DETECT_HIGH_HZ: f64 = 8000.0;
/// Number of sub bands the range is split into.
const BAND_COUNT: usize = 10;
/// Absolute detection threshold, in dBFS.
const THRESHOLD_DB: f64 = -20.0;
/// Detector block length, in milliseconds.
const STEP_MS: f64 = 10.0;
/// Length of the fade in and fade out of every correction, in milliseconds.
const CROSSFADE_MS: f64 = 5.0;

// --- limits that bound look-ahead, memory and CPU (not in the original) ----

/// Longest run of consecutive loud steps that is treated as one click (1 s).
/// A longer stretch is simply split into several clicks, which also makes the
/// gain follow the signal a little more closely.
const MAX_RUN_STEPS: u32 = 100;
/// Longest chain of overlapping events that is repaired as one unit (2 s).
const MAX_GROUP_SAMPLES: u64 = 2 * 44_100;
/// Ring buffer capacity (4 s).  It must exceed `MAX_GROUP_SAMPLES` plus the
/// detector look-ahead by a comfortable margin; see the assertion in `next`.
const RING_CAPACITY: usize = 4 * 44_100;

// ---------------------------------------------------------------------------
// Filter bank layout
// ---------------------------------------------------------------------------

/// Everything that is decided about one sub band before any audio is seen.
pub struct BandDesign {
    pub low_hz: f64,
    pub high_hz: f64,
    /// Band-pass filter used to detect activity in this band.
    pub kernel: BandPassKernel,
    /// Centre frequency of the peaking filter used to repair this band.
    ///
    /// The detector uses the arithmetic centre of the band, the repair uses
    /// the geometric one; that is how the original plug-in does it, and the
    /// difference (a few Hz) is inaudible.
    pub eq_center_hz: f64,
    /// Width of the peaking filter, in octaves.  It equals the width of the
    /// band, so cascading neighbouring bands adds up to a smooth curve.
    pub eq_bandwidth_octaves: f64,
}

/// Designs the whole filter bank: `BAND_COUNT` logarithmically spaced bands
/// between `DETECT_LOW_HZ` and `DETECT_HIGH_HZ`.
///
/// The `N+1` band edges are `low * ratio^i` with `ratio = (high/low)^(1/N)`,
/// so every band spans the same number of octaves.
pub fn design_bands() -> Vec<BandDesign> {
    let ratio = (DETECT_HIGH_HZ / DETECT_LOW_HZ).powf(1.0 / BAND_COUNT as f64);
    let edges: Vec<f64> = (0..=BAND_COUNT)
        .map(|i| DETECT_LOW_HZ * ratio.powi(i as i32))
        .collect();

    (0..BAND_COUNT)
        .map(|i| {
            let (low_hz, high_hz) = (edges[i], edges[i + 1]);
            BandDesign {
                low_hz,
                high_hz,
                kernel: design_band_pass(SAMPLE_RATE, low_hz, high_hz),
                eq_center_hz: (low_hz * high_hz).sqrt(),
                eq_bandwidth_octaves: (high_hz / low_hz).log2(),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Detector state
// ---------------------------------------------------------------------------

/// A run of consecutive steps above the threshold, in one sub band.
#[derive(Clone, Copy)]
struct Run {
    start_step: u64,
    steps: u32,
    /// Largest step peak seen so far, as a linear amplitude.
    peak: f32,
}

/// One detected event: all the clicks that start in the same step.
///
/// The per band results are kept in a fixed size array instead of a list, so
/// an event needs no allocation and the bands are naturally ordered by
/// increasing frequency - the order in which the repair filters are applied.
#[derive(Clone, Copy)]
struct Event {
    start_step: u64,
    /// Length of the event in steps: the longest of the runs that formed it.
    steps: u32,
    /// `peak / threshold` per band, or 0.0 for bands that did not fire.
    ratios: [f32; BAND_COUNT],
}

/// Everything that belongs to one sub band.
struct Band {
    fir: Fir,
    /// Group delay of `fir`, in samples.
    latency: usize,
    /// Peaking EQ parameters used to repair this band.
    eq: PeakingDesign,

    // --- detection state ---
    /// Filtered samples still to be discarded: the first `latency` outputs of
    /// the FIR describe input positions before the start of the file.
    warmup: usize,
    /// Peak magnitude within the step currently being accumulated.
    peak: f32,
    /// Samples accumulated into the current step.
    filled: u64,
    /// Number of steps completed by this band so far.
    steps_done: u64,
    /// Run of loud steps currently open, if any.
    run: Option<Run>,
}

/// Result of feeding one filtered sample to a band.
struct BandUpdate {
    /// A step boundary was crossed, so `steps_done` changed.
    step_closed: bool,
    /// A run ended with this step.
    finished_run: Option<Run>,
}

impl Band {
    /// Consumes one output sample of the band-pass filter.
    ///
    /// `sample` is the FIR output for input position `n`; because the kernel is
    /// symmetric it actually describes input position `n - latency`, which is
    /// why the first `latency` outputs are dropped.  From then on every output
    /// corresponds to exactly one input position, so counting them is the same
    /// as tracking the position in the file.
    fn accept(&mut self, sample: f32, step: u64, threshold: f32) -> BandUpdate {
        if self.warmup > 0 {
            self.warmup -= 1;
            return BandUpdate {
                step_closed: false,
                finished_run: None,
            };
        }

        self.peak = self.peak.max(sample.abs());
        self.filled += 1;
        if self.filled < step {
            return BandUpdate {
                step_closed: false,
                finished_run: None,
            };
        }

        // The step is complete: this is one envelope value.
        let envelope = self.peak;
        self.peak = 0.0;
        self.filled = 0;
        let index = self.steps_done;
        self.steps_done += 1;

        BandUpdate {
            step_closed: true,
            finished_run: self.close_step(index, envelope, threshold),
        }
    }

    /// Threshold comparison and run bookkeeping for one envelope value.
    fn close_step(&mut self, index: u64, envelope: f32, threshold: f32) -> Option<Run> {
        if envelope <= threshold {
            // Quiet step: whatever run was open ends here.
            return self.run.take();
        }

        match &mut self.run {
            None => {
                self.run = Some(Run {
                    start_step: index,
                    steps: 1,
                    peak: envelope,
                })
            }
            Some(run) => {
                run.steps += 1;
                run.peak = run.peak.max(envelope);
            }
        }

        // Enforce the length limit.  Note that the current step has already
        // been counted, so the next loud step simply opens a new run - no step
        // is skipped and no time is lost.
        if self.run.is_some_and(|run| run.steps >= MAX_RUN_STEPS) {
            return self.run.take();
        }
        None
    }
}

// ---------------------------------------------------------------------------
// The filter
// ---------------------------------------------------------------------------

/// De-esser as an iterator adapter.  See the module documentation.
pub struct DeEsser<I> {
    /// Upstream source of samples.
    inner: I,

    bands: Vec<Band>,
    /// Detection threshold as a linear amplitude (10^(-20/20) = 0.1).
    threshold: f32,
    /// Detector block length in samples (441).
    step: u64,
    /// Crossfade length in samples (220).
    crossfade: u64,

    /// Audio that has been read but not yet handed to the sink.  The repair
    /// stage writes its corrections directly into this buffer.
    ring: SampleRing,
    /// Absolute index of the next sample to be returned by `next`.
    emit_index: u64,

    /// Detected events that are not part of a closed group yet, ordered by
    /// `start_step`.
    events: VecDeque<Event>,
    /// Events of the group currently being collected.
    group: Vec<Event>,
    /// Sample range covered by `group`, including crossfades.
    group_start: u64,
    group_end: u64,

    /// Scratch buffers for the repair stage, allocated once.
    excerpt: Vec<f32>,
    filtered: Vec<f32>,

    /// The upstream iterator is exhausted and the tail has been flushed.
    input_done: bool,
}

impl<I: Iterator<Item = f32>> DeEsser<I> {
    pub fn new(inner: I) -> Self {
        let step = (SAMPLE_RATE * STEP_MS / 1000.0).floor() as u64;
        let crossfade = (SAMPLE_RATE * CROSSFADE_MS / 1000.0).round() as u64;
        let threshold = 10f64.powf(THRESHOLD_DB / 20.0) as f32;

        let bands = design_bands()
            .into_iter()
            .map(|design| {
                let latency = design.kernel.latency;
                Band {
                    fir: Fir::new(design.kernel.taps),
                    latency,
                    eq: PeakingDesign::new(
                        SAMPLE_RATE,
                        design.eq_center_hz,
                        design.eq_bandwidth_octaves,
                    ),
                    warmup: latency,
                    peak: 0.0,
                    filled: 0,
                    steps_done: 0,
                    run: None,
                }
            })
            .collect();

        // Longest excerpt the repair stage can ever be asked to filter.
        let max_excerpt = (MAX_RUN_STEPS as u64 * step + 2 * crossfade) as usize;

        DeEsser {
            inner,
            bands,
            threshold,
            step,
            crossfade,
            ring: SampleRing::new(RING_CAPACITY),
            emit_index: 0,
            events: VecDeque::new(),
            group: Vec::new(),
            group_start: 0,
            group_end: 0,
            excerpt: Vec::with_capacity(max_excerpt),
            filtered: Vec::with_capacity(max_excerpt),
            input_done: false,
        }
    }

    // -- detection ---------------------------------------------------------

    /// Stores one input sample and runs it through all band detectors.
    fn push_input(&mut self, sample: f32) {
        self.ring.push(sample);

        let (step, threshold) = (self.step, self.threshold);
        // A run can only finish at a step boundary, so at most one run per
        // band can finish per input sample.
        let mut finished: [Option<Run>; BAND_COUNT] = [None; BAND_COUNT];
        let mut any_step_closed = false;

        for (index, band) in self.bands.iter_mut().enumerate() {
            let filtered = band.fir.process(sample);
            let update = band.accept(filtered, step, threshold);
            any_step_closed |= update.step_closed;
            finished[index] = update.finished_run;
        }

        for (band_index, run) in finished.iter().enumerate() {
            if let Some(run) = run {
                self.record_click(band_index, *run);
            }
        }

        // Nothing downstream can change unless a step boundary was crossed.
        if any_step_closed {
            self.advance_grouping();
        }
    }

    /// Merges a finished run into the event that starts at the same step.
    fn record_click(&mut self, band_index: usize, run: Run) {
        let ratio = run.peak / self.threshold;

        // Runs finish out of order (bands have different filter delays and
        // different run lengths), so the queue is kept sorted by start step.
        // Searching from the back is cheap because a new click almost always
        // belongs to one of the most recent events.
        let mut position = self.events.len();
        while position > 0 && self.events[position - 1].start_step > run.start_step {
            position -= 1;
        }

        if position > 0 && self.events[position - 1].start_step == run.start_step {
            let event = &mut self.events[position - 1];
            event.steps = event.steps.max(run.steps);
            event.ratios[band_index] = event.ratios[band_index].max(ratio);
        } else {
            let mut event = Event {
                start_step: run.start_step,
                steps: run.steps,
                ratios: [0.0; BAND_COUNT],
            };
            event.ratios[band_index] = ratio;
            self.events.insert(position, event);
        }
    }

    /// Number of steps that *every* band has finished.  Bands with a longer
    /// filter lag behind, so this is the position of the slowest one.
    fn detector_steps_done(&self) -> u64 {
        self.bands
            .iter()
            .map(|band| band.steps_done)
            .min()
            .unwrap_or(0)
    }

    /// Earliest sample that a correction which does not exist yet could still
    /// touch.  Everything before it is final.
    fn earliest_future_correction(&self) -> u64 {
        // A brand new run could start at the first step nobody has processed.
        let mut earliest = self.detector_steps_done() * self.step;

        // Runs that are already open start earlier than that.
        for band in &self.bands {
            if let Some(run) = band.run {
                earliest = earliest.min(run.start_step * self.step);
            }
        }

        // So do events that are waiting to be finalised (the queue is sorted,
        // so the front is the earliest).
        if let Some(event) = self.events.front() {
            earliest = earliest.min(event.start_step * self.step);
        }

        earliest.saturating_sub(self.crossfade)
    }

    /// An event is final when no band can still add a click to it: all bands
    /// have processed its starting step, and none of them has a run open that
    /// started there.
    fn event_is_final(&self, event: &Event) -> bool {
        if self.detector_steps_done() <= event.start_step {
            return false;
        }
        !self
            .bands
            .iter()
            .any(|band| band.run.is_some_and(|run| run.start_step == event.start_step))
    }

    /// Sample range covered by an event, including the crossfades.
    fn event_span(&self, event: &Event) -> (u64, u64) {
        let start_sample = event.start_step * self.step;
        let start = start_sample.saturating_sub(self.crossfade);
        let end = start_sample + event.steps as u64 * self.step + self.crossfade;
        (start, end)
    }

    // -- grouping ----------------------------------------------------------

    /// Moves finalised events into the current group and repairs the group as
    /// soon as it is certain that nothing else can join it.
    fn advance_grouping(&mut self) {
        while let Some(&event) = self.events.front() {
            if !self.event_is_final(&event) {
                break;
            }
            self.events.pop_front();
            self.add_to_group(event);
        }

        if !self.group.is_empty() && self.earliest_future_correction() >= self.group_end {
            self.repair_group();
        }
    }

    fn add_to_group(&mut self, event: Event) {
        let (start, end) = self.event_span(&event);

        if self.group.is_empty() {
            self.group_start = start;
            self.group_end = end;
            self.group.push(event);
            return;
        }

        let overlaps = start < self.group_end;
        let extended_end = self.group_end.max(end);
        if overlaps && extended_end - self.group_start <= MAX_GROUP_SAMPLES {
            self.group_end = extended_end;
            self.group.push(event);
        } else {
            // Either there is a gap, or the chain has grown too long to be
            // repaired as one unit.  Close it and start a new one.
            self.repair_group();
            self.group_start = start;
            self.group_end = end;
            self.group.push(event);
        }
    }

    // -- repair ------------------------------------------------------------

    /// Applies the corrections of all events of the current group to the ring
    /// buffer, then clears the group.
    ///
    /// Events are processed in time order and the correction is written back
    /// into the ring, so an event that overlaps an earlier one automatically
    /// sees the already corrected signal - which is what the original does.
    /// The detector has long since read these samples, so modifying them does
    /// not feed back into detection.
    fn repair_group(&mut self) {
        // Destructuring gives us disjoint borrows of the fields we touch.
        let DeEsser {
            ring,
            bands,
            group,
            excerpt,
            filtered,
            step,
            crossfade,
            ..
        } = self;
        let (step, crossfade) = (*step, *crossfade);

        for event in group.iter() {
            let start_sample = event.start_step * step;
            let start = start_sample
                .saturating_sub(crossfade)
                .max(ring.oldest_index());
            // The tail of the last event can reach past the end of the file.
            let end = (start_sample + event.steps as u64 * step + crossfade)
                .min(ring.write_index());
            if end <= start {
                continue;
            }
            let length = (end - start) as usize;

            excerpt.clear();
            filtered.clear();
            for offset in 0..length as u64 {
                let value = ring.get(start + offset);
                excerpt.push(value);
                filtered.push(value);
            }

            // Cascade of peaking filters, from the lowest band upwards.
            for (band_index, band) in bands.iter().enumerate() {
                let ratio = event.ratios[band_index];
                if ratio <= 0.0 {
                    continue; // this band did not fire
                }
                // Cut exactly the excess: a peak that was 6 dB above the
                // threshold gets -6 dB.
                let gain_db = -20.0 * (ratio as f64).log10();
                band.eq.biquad(gain_db).process_in_place(filtered);
            }

            // Fade the correction in and out so that it cannot click.
            for offset in 0..length {
                let fade = trapezoid(offset, length, crossfade as usize);
                let delta = (filtered[offset] - excerpt[offset]) * fade;
                ring.add(start + offset as u64, delta);
            }
        }

        group.clear();
    }

    // -- output ------------------------------------------------------------

    /// Absolute index up to which samples are final and may be handed to the
    /// sink.
    fn safe_limit(&self) -> u64 {
        if self.input_done {
            // Everything has been detected and repaired already.
            return self.ring.write_index();
        }

        let mut limit = self.earliest_future_correction();
        if !self.group.is_empty() {
            // A group that is still collecting events will be rewritten from
            // its start, so nothing from there on may leave yet.
            limit = limit.min(self.group_start);
        }
        limit.min(self.ring.write_index())
    }

    /// Called once the upstream iterator is exhausted.
    fn finish_stream(&mut self) {
        // The band-pass filters still hold `latency` samples worth of signal.
        // Feeding zeros flushes them, which analyses the tail of the file
        // exactly as if the signal continued with silence.
        let (step, threshold) = (self.step, self.threshold);
        for index in 0..self.bands.len() {
            for _ in 0..self.bands[index].latency {
                let value = self.bands[index].fir.process(0.0);
                let update = self.bands[index].accept(value, step, threshold);
                if let Some(run) = update.finished_run {
                    self.record_click(index, run);
                }
            }
        }

        // Close runs that are still open at the end of the file.
        for index in 0..self.bands.len() {
            if let Some(run) = self.bands[index].run.take() {
                self.record_click(index, run);
            }
        }

        // No more clicks can appear, so every queued event is final.
        while let Some(event) = self.events.pop_front() {
            self.add_to_group(event);
        }
        if !self.group.is_empty() {
            self.repair_group();
        }

        self.input_done = true;
    }
}

impl<I: Iterator<Item = f32>> Iterator for DeEsser<I> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        loop {
            // 1. Hand out a sample that can no longer be modified.
            if self.emit_index < self.safe_limit() {
                let sample = self.ring.get(self.emit_index);
                self.emit_index += 1;
                return Some(sample);
            }

            // 2. Nothing is ready: pull more input, or finish up.
            if self.input_done {
                return None;
            }
            match self.inner.next() {
                Some(sample) => {
                    // The ring must never overwrite a sample that has not been
                    // emitted yet.  The distance between the write position
                    // and the output position is bounded by the detector
                    // look-ahead plus MAX_GROUP_SAMPLES, which is well below
                    // RING_CAPACITY; if this ever fires, those limits and the
                    // capacity have drifted apart.
                    assert!(
                        self.ring.write_index() - self.emit_index < self.ring.capacity() as u64,
                        "ring buffer overflow: look-ahead exceeded RING_CAPACITY"
                    );
                    self.push_input(sample);
                }
                None => self.finish_stream(),
            }
        }
    }
}

/// Trapezoidal fade used to blend a correction in and out.
///
/// Rises linearly from 0 to 1 over the first `ramp` samples, stays at 1, and
/// falls back to 0 over the last `ramp` samples.  With `ramp == 0` it is a
/// plain rectangle.
fn trapezoid(offset: usize, length: usize, ramp: usize) -> f32 {
    if ramp == 0 || length <= 2 * ramp {
        return 1.0;
    }
    if offset <= ramp {
        offset as f32 / ramp as f32
    } else if offset >= length - ramp {
        (length - offset) as f32 / ramp as f32
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::FilterExt;
    use std::f64::consts::PI;

    fn sine(freq: f64, amplitude: f64, length: usize) -> Vec<f32> {
        (0..length)
            .map(|n| (amplitude * (2.0 * PI * freq * n as f64 / SAMPLE_RATE).sin()) as f32)
            .collect()
    }

    fn peak(samples: &[f32]) -> f32 {
        samples.iter().fold(0.0f32, |acc, s| acc.max(s.abs()))
    }

    /// Applies a raised cosine fade to both ends of a buffer.
    ///
    /// A tone that starts abruptly is a click: its energy is spread over all
    /// bands, so neighbouring detectors fire as well and add their own cuts.
    /// That is correct behaviour, but it gets in the way when the point of the
    /// test is the steady state, hence the smooth edges.
    fn fade_edges(samples: &mut [f32], ramp: usize) {
        let length = samples.len();
        for index in 0..ramp.min(length / 2) {
            let gain = 0.5 - 0.5 * (PI * index as f64 / ramp as f64).cos();
            samples[index] *= gain as f32;
            samples[length - 1 - index] *= gain as f32;
        }
    }

    #[test]
    fn quiet_signal_passes_through_unchanged() {
        // Well below the -20 dB threshold: nothing should be detected, so the
        // output must be bit identical to the input.
        let input = sine(4000.0, 0.02, 44_100);
        let output: Vec<f32> = input.iter().copied().deesser().collect();
        assert_eq!(input, output);
    }

    #[test]
    fn signal_outside_the_detection_range_passes_through_unchanged() {
        // 400 Hz at full scale: loud, but nowhere near the analysed band.
        let input = sine(400.0, 0.9, 44_100);
        let output: Vec<f32> = input.iter().copied().deesser().collect();
        assert_eq!(input, output);
    }

    #[test]
    fn length_is_preserved() {
        // Whatever happens inside, the filter must return exactly as many
        // samples as it was given - including for inputs shorter than one
        // detector step or one filter kernel.
        for length in [0usize, 1, 100, 441, 1000, 44_100, 100_000] {
            let count = sine(4000.0, 0.5, length).into_iter().deesser().count();
            assert_eq!(count, length, "input of {length} samples");
        }
    }

    #[test]
    fn band_layout_matches_the_specification() {
        // Cross-check against the table in DEESSER.md section 13.1.
        let bands = design_bands();
        assert_eq!(bands.len(), 10);

        let first = &bands[0];
        assert!((first.low_hz - 2500.00).abs() < 0.01);
        assert!((first.high_hz - 2808.37).abs() < 0.01);
        assert_eq!(first.kernel.periods, 27);
        assert_eq!(first.kernel.taps.len(), 448);
        assert_eq!(first.kernel.latency, 224);
        assert!((first.eq_center_hz - 2649.70).abs() < 0.01);
        assert!((first.eq_bandwidth_octaves - 0.16781).abs() < 0.0001);

        let last = &bands[9];
        assert!((last.low_hz - 7121.56).abs() < 0.01);
        assert!((last.high_hz - 8000.00).abs() < 0.01);
        assert_eq!(last.kernel.periods, 27);
        assert_eq!(last.kernel.taps.len(), 157);
        assert_eq!(last.kernel.latency, 78);
        assert!((last.eq_center_hz - 7548.01).abs() < 0.01);

        // Every band must be designed for the same number of cycles: with
        // logarithmic spacing the filter bank has constant Q.
        assert!(bands.iter().all(|band| band.kernel.periods == 27));
    }

    /// Detection centre of band 5 (3981.07 .. 4472.14 Hz).  A tone here is
    /// measured by exactly one band, at that band's unity gain point.
    const BAND5_CENTER_HZ: f64 = 4226.60;

    #[test]
    fn loud_tone_at_a_band_centre_is_pushed_down_to_the_threshold() {
        // Half a second of tone: short enough to stay one single run, so the
        // whole thing is corrected by one event.  0.5 is about -6 dBFS, i.e.
        // 14 dB above the -20 dB threshold, and the correction should land the
        // steady state exactly on the threshold.
        let mut input = vec![0.0f32; 44_100];
        let mut tone = sine(BAND5_CENTER_HZ, 0.5, 22_050);
        fade_edges(&mut tone, 2205);
        input[11_000..11_000 + tone.len()].copy_from_slice(&tone);

        let output: Vec<f32> = input.iter().copied().deesser().collect();
        assert_eq!(output.len(), input.len());

        // Skip the fades at both ends of the correction.
        let measured_db = 20.0 * (peak(&output[13_000..31_000]) as f64).log10();
        assert!(
            (measured_db - THRESHOLD_DB).abs() < 0.5,
            "expected about {THRESHOLD_DB} dB, measured {measured_db} dB"
        );
    }

    #[test]
    fn sustained_tone_stays_close_to_the_threshold() {
        // A tone that never stops is split into several events, because a run
        // is cut after MAX_RUN_STEPS.  Neighbouring events overlap by two
        // crossfades and the later one is computed on top of the correction of
        // the earlier one, so the attenuation dips slightly at every seam.
        // The same happens in the original plug-in whenever two events are
        // close together; it stays around a dB and is not audible on speech,
        // but it means "exactly the threshold" only holds per event.
        let input = sine(BAND5_CENTER_HZ, 0.5, 3 * 44_100);
        let output: Vec<f32> = input.iter().copied().deesser().collect();

        let measured_db = 20.0 * (peak(&output[44_100..2 * 44_100]) as f64).log10();
        assert!(
            (measured_db - THRESHOLD_DB).abs() < 2.0,
            "expected roughly {THRESHOLD_DB} dB, measured {measured_db} dB"
        );
    }

    #[test]
    fn loud_tone_between_two_band_centres_is_only_partly_reduced() {
        // This documents a property of the algorithm rather than a bug: the
        // cut is exact only at the centre of a band.  A tone sitting between
        // two centres is measured through the skirts of two detectors (so the
        // measured excess is smaller than the real one) and repaired by the
        // skirts of two bells, so it ends up above the threshold.
        let input = sine(4000.0, 0.5, 3 * 44_100);
        let output: Vec<f32> = input.iter().copied().deesser().collect();

        let measured_db = 20.0 * (peak(&output[44_100..2 * 44_100]) as f64).log10();
        assert!(
            measured_db < -14.0,
            "expected a clear reduction, measured {measured_db} dB"
        );
        assert!(
            measured_db > THRESHOLD_DB,
            "cannot go below the threshold, measured {measured_db} dB"
        );
    }

    #[test]
    fn a_short_burst_is_attenuated_and_its_surroundings_are_not() {
        // 100 ms of a loud tone in the middle of an otherwise silent file.
        let mut input = vec![0.0f32; 44_100];
        let burst = sine(BAND5_CENTER_HZ, 0.6, 4410);
        input[20_000..20_000 + burst.len()].copy_from_slice(&burst);

        let output: Vec<f32> = input.iter().copied().deesser().collect();
        assert_eq!(output.len(), input.len());

        // The body of the burst is pushed down to the threshold.  The first
        // and last few hundred samples are skipped: that is where the
        // correction fades in and out.
        let body = 21_000..23_400;
        let before = peak(&input[body.clone()]);
        let after = peak(&output[body]);
        assert!(
            after < before * 0.25,
            "burst was not attenuated: {before} -> {after}"
        );

        // ... and the silence around it is untouched.
        assert_eq!(&output[..19_000], &input[..19_000]);
        assert_eq!(&output[26_000..], &input[26_000..]);
    }
}
