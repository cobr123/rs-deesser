//! A de-esser for 16-bit mono 44100 Hz WAV files.
//!
//! ```text
//! rs-deesser [--bypass] <input.wav> <output.wav>
//! ```
//!
//! # Structure
//!
//! The program is a pipeline built out of plain Rust iterators:
//!
//! ```text
//!   WavReader        ProgressReport      DeEsser         write_wav
//!   (source)   -->   (filter)      -->  (filter)   -->  (sink)
//! ```
//!
//! * [`wav::WavReader`] turns a file into a stream of samples,
//! * [`progress::ProgressReport`] counts them and draws a progress line,
//! * [`deesser::DeEsser`] removes sibilance,
//! * [`wav::write_wav`] pulls the stream and writes it back to a file.
//!
//! Filters are added and removed by editing one expression in `run` below;
//! see [`pipeline::FilterExt`].
//!
//! Every stage is lazy: nothing is computed until the sink asks for the next
//! sample, and no stage holds more than a fixed amount of audio, so files of
//! any length are processed in about a megabyte of memory.
//!
//! Samples are `f32` in the range -1.0 ..= +1.0 everywhere between the source
//! and the sink; that is the only contract a new filter has to honour.  With
//! no filter in between (`--bypass`) the pipeline degenerates into a plain
//! file copy, which is a useful sanity check of the two ends.
//!
//! The algorithm implemented by the filter is described in `DEESSER.md`; it is
//! a port of the `DeEsser.ny` Nyquist plug-in for Audacity, with the fixed
//! parameter set: threshold -20 dB, step 10 ms, 2500..8000 Hz, 10 bands,
//! crossfade 5 ms, "apply changes".

mod deesser;
mod dsp;
mod pipeline;
mod progress;
mod ring;
mod wav;

use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use pipeline::FilterExt;
use wav::WavReader;

const USAGE: &str = "\
usage: rs-deesser [--bypass] <input.wav> <output.wav>
       rs-deesser --bands

Applies a de-esser to a 16-bit mono PCM WAV file sampled at 44100 Hz.

options:
  --bypass     copy the samples without filtering (source -> sink only)
  --bands      print the design of the filter bank and exit
  -q, --quiet  do not draw the progress indicator
  -h, --help   show this message";

fn main() -> ExitCode {
    let options = match parse_arguments() {
        Ok(Command::Process(options)) => options,
        // --help / --bands: nothing to process, and not an error.
        Ok(Command::Help) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Ok(Command::ShowBands) => {
            print_band_table();
            return ExitCode::SUCCESS;
        }
        Err(message) => {
            eprintln!("rs-deesser: {message}\n\n{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    match run(&options) {
        Ok(samples) => {
            let seconds = samples as f64 / deesser::SAMPLE_RATE;
            println!(
                "{} -> {}: {samples} samples ({seconds:.2} s){}",
                options.input.display(),
                options.output.display(),
                if options.bypass { ", bypassed" } else { "" }
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("rs-deesser: {error}");
            ExitCode::FAILURE
        }
    }
}

struct Options {
    input: PathBuf,
    output: PathBuf,
    bypass: bool,
    quiet: bool,
}

enum Command {
    Process(Options),
    ShowBands,
    Help,
}

/// Hand written argument parsing - the command line is small enough that a
/// dependency would cost more than it saves.
fn parse_arguments() -> Result<Command, String> {
    let mut positional: Vec<PathBuf> = Vec::new();
    let mut bypass = false;
    let mut quiet = false;

    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "-h" | "--help" => return Ok(Command::Help),
            "--bands" => return Ok(Command::ShowBands),
            "--bypass" => bypass = true,
            "-q" | "--quiet" => quiet = true,
            other if other.starts_with('-') => {
                return Err(format!("unknown option '{other}'"));
            }
            other => positional.push(PathBuf::from(other)),
        }
    }

    match positional.len() {
        2 => Ok(Command::Process(Options {
            input: positional[0].clone(),
            output: positional[1].clone(),
            bypass,
            quiet,
        })),
        0 | 1 => Err("expected an input and an output file".to_string()),
        _ => Err("expected exactly two file names".to_string()),
    }
}

/// Prints the designed filter bank: useful when comparing the implementation
/// against the algorithm description, and a compact summary of what the
/// detector actually looks at.
fn print_band_table() {
    println!(
        "{:>2} {:>9} {:>9} {:>9} {:>4} {:>5} {:>5} {:>9} {:>8}",
        "i", "low Hz", "high Hz", "centre", "per", "taps", "lat", "eq Hz", "eq oct"
    );
    for (index, band) in deesser::design_bands().iter().enumerate() {
        println!(
            "{:>2} {:>9.2} {:>9.2} {:>9.2} {:>4} {:>5} {:>5} {:>9.2} {:>8.5}",
            index + 1,
            band.low_hz,
            band.high_hz,
            band.kernel.center_hz,
            band.kernel.periods,
            band.kernel.taps.len(),
            band.kernel.latency,
            band.eq_center_hz,
            band.eq_bandwidth_octaves
        );
    }
}

/// Builds the pipeline and runs it.  Returns the number of samples written.
fn run(options: &Options) -> io::Result<u64> {
    // Opening the source also validates the format: mono, 44100 Hz, 16 bit.
    let mut source = WavReader::open(&options.input)?;

    let input_samples = source.len();

    // The progress indicator is a filter like any other: it passes samples
    // through unchanged and only watches them go by.  It sits in front of the
    // de-esser, so it reports how much of the input has been read.
    let written = if options.bypass {
        // Source into sink, with only the report in between: a copy.
        write_pipeline(
            &options.output,
            (&mut source).progress(input_samples, options.quiet),
        )?
    } else {
        // The same stream with the de-esser spliced in.  Further filters
        // would simply be chained here: `.deesser().some_other_filter()`.
        write_pipeline(
            &options.output,
            (&mut source)
                .progress(input_samples, options.quiet)
                .deesser(),
        )?
    };

    // `Iterator::next` cannot report failures, so a read error shows up as an
    // early end of stream.  Check for it before declaring success.
    if let Some(error) = source.take_error() {
        return Err(error);
    }

    // The de-esser is length preserving, so a mismatch means the input was
    // truncated in a way that still parsed.
    if written != input_samples {
        eprintln!(
            "rs-deesser: warning: header announced {input_samples} samples, wrote {written}"
        );
    }

    Ok(written)
}

fn write_pipeline(path: &Path, samples: impl Iterator<Item = f32>) -> io::Result<u64> {
    wav::write_wav(path, samples)
}
