//! Signal processing building blocks: a band-pass FIR used by the detector and
//! a peaking (bell) IIR filter used by the repair stage.
//!
//! Everything here is plain scalar floating point arithmetic.  No SIMD, no
//! FFT, no lookup tables: the intent is that a reader who knows what an FIR
//! and a biquad are can follow the code line by line.  Filter design is done
//! in `f64` (it happens once, at start-up), the sample-by-sample work is done
//! in `f32` for the FIR and in `f64` for the biquad, whose feedback path is
//! more sensitive to rounding.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Window function
// ---------------------------------------------------------------------------

/// Coefficients of the SFT3F window: `w(u) = a0 + a1*cos(2*pi*u) + a2*cos(4*pi*u)`.
///
/// SFT3F is a three term "flat top" window (Heinzel et al., "Spectrum and
/// spectral density estimation by the DFT", Appendix D).  It was chosen by the
/// original plug-in because it has the highest ratio of -3 dB width to first
/// zero among the windows considered, i.e. the steepest main lobe for a given
/// width, which is what you want when you isolate a narrow frequency band.
const SFT3F: [f64; 3] = [0.26526, -0.5, 0.23474];

/// Width of the SFT3F main lobe at -3 dB, expressed in DFT bins (a "bin" being
/// the reciprocal of the window duration).  Used to match the filter bandwidth
/// to the width of a frequency band.
const SFT3F_3DB_WIDTH_BINS: f64 = 3.1502;

/// Evaluates the window at position `u`, where `u` runs from 0 at the first
/// sample to 1 at the (nominal) end of the window.
///
/// Note that `a0 + a1 + a2 == 0`, so the window starts and ends at zero and
/// peaks at 1.0 in the middle.
fn sft3f_window(u: f64) -> f64 {
    SFT3F[0] + SFT3F[1] * (2.0 * PI * u).cos() + SFT3F[2] * (4.0 * PI * u).cos()
}

// ---------------------------------------------------------------------------
// Band-pass FIR design
// ---------------------------------------------------------------------------

/// A designed band-pass kernel, ready to be handed to [`Fir::new`].
pub struct BandPassKernel {
    /// Impulse response, `taps[0]` multiplies the newest sample.
    pub taps: Vec<f32>,
    /// Group delay in samples (`taps.len() / 2`).  The kernel is symmetric, so
    /// the output at time `n` describes the input at time `n - latency`.
    pub latency: usize,
    /// Centre frequency the kernel was designed for, in Hz.
    pub center_hz: f64,
    /// Number of whole cycles of `center_hz` that fit in the kernel.
    pub periods: u32,
}

/// Designs a band-pass FIR for the band `band_lo .. band_hi`.
///
/// The kernel is a cosine at the centre frequency multiplied by a window:
///
/// ```text
///     k[n] = w(u[n]) * cos(2*pi * center * n / fs)
/// ```
///
/// A windowed cosine is a band-pass filter whose passband shape is the
/// spectrum of the window shifted up to the centre frequency, and whose phase
/// is linear (the kernel is symmetric).  Two design decisions matter:
///
/// 1. **The window must contain a whole number `P` of cycles.**  Otherwise the
///    kernel does not start and end symmetrically around the centre and the
///    passband is skewed.  `P` is therefore rounded down, which can only make
///    the window shorter and the passband wider.
///
/// 2. **The bandwidth is matched to the band.**  The -3 dB width of the main
///    lobe is `SFT3F_3DB_WIDTH_BINS / T` Hz for a window of duration `T`, so
///    requiring that this equals the width of the band fixes `T`, and with it
///    `P = center * T` and the kernel length `L = fs * T`.
///
/// Finally the kernel is scaled so that a sinusoid at the centre frequency
/// passes through with **unchanged amplitude**.  This is what makes the
/// absolute detection threshold meaningful: the detector compares the filtered
/// signal against a plain amplitude value.
pub fn design_band_pass(sample_rate: f64, band_lo: f64, band_hi: f64) -> BandPassKernel {
    assert!(band_lo > 0.0 && band_hi > band_lo);

    let center_hz = 0.5 * (band_lo + band_hi);
    let band_width_hz = band_hi - band_lo;

    // Widest bin (= shortest window) that still resolves this band ...
    let min_bin_hz = band_width_hz / SFT3F_3DB_WIDTH_BINS;
    // ... turned into a whole number of cycles, at least one.
    let periods = (center_hz / min_bin_hz).floor().max(1.0) as u32;

    // Exact window duration in samples.  It is generally not an integer; the
    // kernel is truncated to the samples that fit, and `length_exact` stays as
    // the reference for the window shape so that the window really spans the
    // whole `periods` cycles.
    let length_exact = sample_rate * periods as f64 / center_hz;
    let length = length_exact.floor() as usize;

    let mut taps = vec![0.0f64; length];
    // Response of the (still unnormalised) kernel to a cosine at the centre
    // frequency, accumulated while the taps are generated.
    let mut gain_at_center = 0.0f64;
    for (n, tap) in taps.iter_mut().enumerate() {
        let u = n as f64 / length_exact;
        let phase = 2.0 * PI * center_hz * n as f64 / sample_rate;
        let value = sft3f_window(u) * phase.cos();
        gain_at_center += value * phase.cos();
        *tap = value;
    }

    // A symmetric kernel of length L delays the signal by L/2 samples, which
    // at the centre frequency is a phase shift of pi*P.  For odd `P` that
    // inverts the band-passed waveform.  Flipping the sign undoes it; this is
    // cosmetic for a detector that looks at magnitudes, but it keeps the
    // filtered signal in phase with the input if anyone ever plots it.
    let sign = if periods % 2 == 1 { -1.0 } else { 1.0 };
    let scale = sign / gain_at_center;

    BandPassKernel {
        taps: taps.iter().map(|t| (t * scale) as f32).collect(),
        latency: length / 2,
        center_hz,
        periods,
    }
}

// ---------------------------------------------------------------------------
// FIR filter
// ---------------------------------------------------------------------------

/// A direct form FIR filter: `y[n] = sum(taps[m] * x[n-m])`.
///
/// The history is kept in a circular buffer, so no samples are ever moved
/// around.  Cost is one multiply-add per tap per sample; for this program that
/// is about 2800 multiply-adds per input sample across all ten bands.
pub struct Fir {
    taps: Vec<f32>,
    history: Vec<f32>,
    /// Index in `history` where the newest sample is stored.
    newest: usize,
}

impl Fir {
    pub fn new(taps: Vec<f32>) -> Self {
        assert!(!taps.is_empty());
        let history = vec![0.0; taps.len()];
        Fir {
            taps,
            history,
            newest: 0,
        }
    }

    /// Feeds one sample and returns one output sample.
    pub fn process(&mut self, sample: f32) -> f32 {
        let len = self.history.len();
        self.history[self.newest] = sample;

        // Walk the taps forwards and the history backwards in time.
        let mut acc = 0.0f32;
        let mut index = self.newest;
        for &tap in &self.taps {
            acc += tap * self.history[index];
            index = if index == 0 { len - 1 } else { index - 1 };
        }

        self.newest = if self.newest + 1 == len { 0 } else { self.newest + 1 };
        acc
    }
}

// ---------------------------------------------------------------------------
// Peaking (bell) EQ
// ---------------------------------------------------------------------------

/// The parts of a peaking EQ that do not depend on the gain.
///
/// The repair stage builds a new filter for every event, but always at the
/// same centre frequency and bandwidth, so these two numbers are computed once
/// per band at start-up.
#[derive(Clone, Copy)]
pub struct PeakingDesign {
    cos_w0: f64,
    alpha: f64,
}

impl PeakingDesign {
    /// `bandwidth_octaves` is the distance between the two frequencies at
    /// which the response reaches half of the peak gain **in dB** - the same
    /// definition Nyquist's `eq-band` and the RBJ Audio EQ Cookbook use.
    pub fn new(sample_rate: f64, center_hz: f64, bandwidth_octaves: f64) -> Self {
        let w0 = 2.0 * PI * center_hz / sample_rate;
        let sin_w0 = w0.sin();
        // alpha = sin(w0) * sinh( ln(2)/2 * BW * w0/sin(w0) )
        let alpha = sin_w0 * (0.5 * 2f64.ln() * bandwidth_octaves * w0 / sin_w0).sinh();
        PeakingDesign {
            cos_w0: w0.cos(),
            alpha,
        }
    }

    /// Instantiates a filter with the given gain (negative = cut).
    pub fn biquad(&self, gain_db: f64) -> Biquad {
        // A is the square root of the linear gain: the peaking design applies
        // A to the numerator and 1/A to the denominator, so the peak ends up
        // at A^2 = 10^(gain_db/20).
        let a = 10f64.powf(gain_db / 40.0);
        let a0 = 1.0 + self.alpha / a;
        Biquad {
            b0: (1.0 + self.alpha * a) / a0,
            b1: (-2.0 * self.cos_w0) / a0,
            b2: (1.0 - self.alpha * a) / a0,
            a1: (-2.0 * self.cos_w0) / a0,
            a2: (1.0 - self.alpha / a) / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }
}

/// A second order IIR section in direct form I:
///
/// ```text
///     y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
/// ```
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    /// Filters one sample.
    pub fn process(&mut self, sample: f32) -> f32 {
        let x0 = sample as f64;
        let y0 = self.b0 * x0 + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x0;
        self.y2 = self.y1;
        self.y1 = y0;
        y0 as f32
    }

    /// Filters a whole slice in place, starting from a zeroed state.
    ///
    /// The repair stage always filters short, independent excerpts, so each
    /// one starts with an empty filter memory - exactly like the original
    /// plug-in, where every excerpt is a fresh sound.  The resulting start-up
    /// transient is hidden by the crossfade that is applied afterwards.
    pub fn process_in_place(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FS: f64 = 44_100.0;

    /// Feeds a sinusoid through a filter and returns the peak amplitude of the
    /// steady state part of the output.
    fn measure_amplitude(mut filter: impl FnMut(f32) -> f32, freq: f64, samples: usize) -> f64 {
        let mut peak = 0.0f64;
        for n in 0..samples {
            let x = (2.0 * PI * freq * n as f64 / FS).sin() as f32;
            let y = filter(x);
            // Ignore the start-up transient (first half of the run).
            if n >= samples / 2 {
                peak = peak.max(y.abs() as f64);
            }
        }
        peak
    }

    #[test]
    fn band_pass_has_unity_gain_at_center() {
        let kernel = design_band_pass(FS, 2500.0, 2808.37);
        let mut fir = Fir::new(kernel.taps.clone());
        let amplitude = measure_amplitude(|x| fir.process(x), kernel.center_hz, 8192);
        assert!(
            (amplitude - 1.0).abs() < 0.01,
            "expected unity gain, measured {amplitude}"
        );
    }

    #[test]
    fn band_pass_rejects_far_away_frequencies() {
        let kernel = design_band_pass(FS, 2500.0, 2808.37);
        let mut fir = Fir::new(kernel.taps.clone());
        // One octave below the band: should be far down the stop band.
        let amplitude = measure_amplitude(|x| fir.process(x), 1300.0, 8192);
        assert!(amplitude < 0.02, "expected strong rejection, measured {amplitude}");
    }

    #[test]
    fn band_pass_geometry_matches_the_specification() {
        // The worked example from DEESSER.md, band 1 of the default layout.
        let kernel = design_band_pass(FS, 2500.0, 2808.37);
        assert_eq!(kernel.periods, 27);
        assert_eq!(kernel.taps.len(), 448);
        assert_eq!(kernel.latency, 224);
    }

    #[test]
    fn peaking_eq_reaches_the_requested_gain() {
        let design = PeakingDesign::new(FS, 2649.70, 0.16781);
        let mut biquad = design.biquad(-6.0);
        let amplitude = measure_amplitude(|x| biquad.process(x), 2649.70, 8192);
        let measured_db = 20.0 * amplitude.log10();
        assert!(
            (measured_db + 6.0).abs() < 0.1,
            "expected -6 dB at the centre, measured {measured_db} dB"
        );
    }

    #[test]
    fn peaking_eq_leaves_distant_frequencies_alone() {
        let design = PeakingDesign::new(FS, 2649.70, 0.16781);
        let mut biquad = design.biquad(-12.0);
        let amplitude = measure_amplitude(|x| biquad.process(x), 500.0, 8192);
        let measured_db = 20.0 * amplitude.log10();
        assert!(
            measured_db.abs() < 0.2,
            "expected no change at 500 Hz, measured {measured_db} dB"
        );
    }
}
