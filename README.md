# rs-deesser

A de-esser for 16-bit mono 44100 Hz WAV files, implemented as a streaming
iterator pipeline with no third-party crates.

```sh
cargo build --release
target/release/rs-deesser input.wav output.wav
```

Options:

| Option | Meaning |
|---|---|
| `--bypass` | connect the source straight to the sink: copies the file |
| `--bands` | print the designed filter bank and exit |
| `-q`, `--quiet` | do not draw the progress indicator |
| `-h`, `--help` | usage |

While it runs, a progress line is drawn on stderr and updated in place:

```text
Progress:  42% ETA: 2m 10s
```

It is removed again when the program finishes, and it is skipped entirely when
stderr is not a terminal, so redirected output stays clean.

Input files that are not 16-bit PCM, mono and 44100 Hz are rejected with a
message; nothing is converted or resampled.

## Pipeline

```text
  WavReader  ->  ProgressReport  ->  DeEsser  ->  write_wav
  (source)       (filter)            (filter)     (sink)
```

Every stage is an ordinary Rust iterator over `f32` samples in the range
-1.0 ..= +1.0, so filters compose:

```rust
let pipeline = source.progress(total_samples, quiet).deesser();
wav::write_wav(&output_path, pipeline)?;
```

`ProgressReport` shows what a filter that only observes the stream looks like:
it passes every sample through untouched and only counts them.

Nothing is computed until the sink pulls a sample, and no stage stores an
amount of audio that depends on the length of the file.

To add another filter, write an `Iterator<Item = f32>` adapter and give it a
method on the `FilterExt` trait in `src/pipeline.rs`; the source, the sink and
the existing filters do not change.

## Source layout

| File | Contents |
|---|---|
| `src/main.rs` | command line, wiring of source, filters and sink |
| `src/pipeline.rs` | `FilterExt`: where filters are hooked into the stream |
| `src/wav.rs` | RIFF/WAVE parsing and writing, `i16 <-> f32` conversion |
| `src/dsp.rs` | window function, band-pass FIR design, FIR and biquad filters |
| `src/ring.rs` | ring buffer addressed by absolute sample index |
| `src/deesser.rs` | the de-esser: detector, event grouping, repair |
| `src/progress.rs` | pass-through filter that reports progress and an ETA |

## The algorithm in one paragraph

The range 2500..8000 Hz is split into ten logarithmically spaced bands. Each
band is isolated by a band-pass FIR (a windowed cosine, 27 cycles long,
normalised to unity gain at its centre frequency) and the peak magnitude of the
filtered signal is measured over 10 ms blocks. A run of blocks above -20 dBFS
is an event; the corresponding excerpt of the audio is passed through a cascade
of peaking EQ filters, one per band that fired, each cutting exactly the number
of dB by which that band exceeded the threshold. The difference between the
filtered and the original excerpt is faded in and out over 5 ms and added back
into the signal, so only the detected regions and only the offending bands are
touched.

The full specification, including the design formulas and a table of all
derived constants, is in `../DEESSER.md`. The program is a port of the
`DeEsser.ny` Nyquist plug-in for Audacity with a fixed parameter set.

## Fixed parameters

| Parameter | Value |
|---|---|
| analysed range | 2500 .. 8000 Hz, 10 bands |
| threshold | -20 dBFS |
| detector step | 10 ms (441 samples) |
| crossfade | 5 ms (220 samples) |
| action | apply changes |

Two limits that the original plug-in does not have keep the look-ahead bounded:
a run of loud blocks is cut after 1 s, and a chain of overlapping events is
repaired in units of at most 2 s. They are the constants `MAX_RUN_STEPS` and
`MAX_GROUP_SAMPLES` in `src/deesser.rs`.

## Static build

For copying the binary to another machine (same CPU architecture, any
distribution):

```sh
cargo build-static          # alias for --release --target x86_64-unknown-linux-musl
scp target/x86_64-unknown-linux-musl/release/rs-deesser other-host:
```

The musl target links the C library into the executable, so there is no dynamic
loader and no libc version to match:

```console
$ file target/x86_64-unknown-linux-musl/release/rs-deesser
ELF 64-bit LSB pie executable, x86-64, static-pie linked, stripped
$ ldd target/x86_64-unknown-linux-musl/release/rs-deesser
        statically linked
```

The target has to be installed once: `rustup target add x86_64-unknown-linux-musl`.
Settings live in `.cargo/config.toml`; `cargo build` and `cargo test` are not
affected by them and keep producing an ordinary dynamically linked binary for
the host. `cargo test-static` runs the test suite against the musl build.

Note that the build deliberately does *not* pass `-C target-cpu=native`: the
default is the baseline x86-64 instruction set, so the binary runs on every
x86-64 CPU rather than only on machines as new as the one that built it.

## Cost

Measured on a ten minute file (`--release`):

* peak resident memory: **3 MB**, the same as for a ten second file
  (the static musl build uses about 1 MB);
* run time: about 47 s, i.e. roughly 13x faster than real time;
* the musl and the glibc build produce bit identical output.

The detector is the expensive part: ten FIRs of 448 .. 157 taps, about 2800
multiply-accumulates per input sample. It is written as a plain scalar loop for
readability; an FFT based convolution would be much faster.

## Tests

```sh
cargo test --release
```

The tests cover the exactness of the `i16 -> f32 -> i16` round trip, the ring
buffer, the measured gain of the designed filters, the geometry of the filter
bank against the specification, the behaviour of the de-esser itself (length
preservation, quiet and out-of-band material passing through unchanged, a loud
tone being pushed down to the threshold) and the progress filter (samples pass
through untouched, durations are formatted as expected).
