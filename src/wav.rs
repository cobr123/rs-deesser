//! Minimal WAV (RIFF/WAVE) reading and writing, implemented from scratch.
//!
//! This module provides the two ends of the processing pipeline:
//!
//! * [`WavReader`] - the *source*.  It is an `Iterator<Item = f32>`, so it can
//!   be plugged into any chain of iterator adapters.
//! * [`write_wav`] - the *sink*.  It pulls samples out of any
//!   `Iterator<Item = f32>` and writes them to a file.
//!
//! Connecting the source directly to the sink copies the file: samples are
//! converted `i16 -> f32 -> i16`, and that round trip is exact (see
//! [`sample_from_i16`] / [`sample_to_i16`]).
//!
//! Only the exact format this program is specified for is accepted:
//! 16-bit signed PCM, one channel, 44100 Hz.  Anything else is rejected with a
//! descriptive error instead of being converted, resampled or guessed at.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// The one and only sample rate this program accepts.
pub const REQUIRED_SAMPLE_RATE: u32 = 44_100;
/// The one and only channel count this program accepts.
pub const REQUIRED_CHANNELS: u16 = 1;
/// The one and only sample format this program accepts.
pub const REQUIRED_BITS_PER_SAMPLE: u16 = 16;

/// `WAVE_FORMAT_PCM`, the only format tag we accept.
const FORMAT_TAG_PCM: u16 = 1;

/// Samples travel through the pipeline as `f32` in the range -1.0 ..= +1.0.
///
/// 16-bit PCM is scaled by 32768 (not 32767): the scale factor is a power of
/// two, so every `i16` maps to an exactly representable `f32` and the value
/// can be recovered without loss.
#[inline]
pub fn sample_from_i16(raw: i16) -> f32 {
    raw as f32 / 32768.0
}

/// Inverse of [`sample_from_i16`], with clipping.
///
/// `+1.0` would map to 32768, which does not fit into an `i16`, so the
/// positive side is clamped to 32767.  Rounding is to nearest, which makes the
/// round trip `i16 -> f32 -> i16` exact for every input value.
#[inline]
pub fn sample_to_i16(sample: f32) -> i16 {
    let scaled = (sample * 32768.0).round();
    if scaled >= 32767.0 {
        i16::MAX
    } else if scaled <= -32768.0 {
        i16::MIN
    } else {
        scaled as i16
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

// ---------------------------------------------------------------------------
// Source
// ---------------------------------------------------------------------------

/// Streaming reader for 16-bit mono PCM WAV files.
///
/// The header is parsed and validated by [`WavReader::open`]; the sample data
/// is then read lazily, a couple of bytes at a time, through a `BufReader`.
/// Memory use does not depend on the length of the file.
///
/// `Iterator::next` cannot report errors, so a read error stops the iteration
/// (`next` returns `None`) and is stored inside the reader.  Callers must ask
/// for it with [`WavReader::take_error`] once the pipeline has finished,
/// otherwise a truncated file would silently look like a short file.
pub struct WavReader<R: Read> {
    reader: R,
    /// Number of samples not yet returned.
    remaining: u64,
    /// Total number of samples in the `data` chunk.
    total: u64,
    error: Option<io::Error>,
}

impl WavReader<BufReader<File>> {
    /// Opens a file and validates that it is 16-bit mono PCM at 44100 Hz.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        WavReader::new(BufReader::new(file))
    }
}

impl<R: Read> WavReader<R> {
    /// Parses and validates the RIFF header, leaving the reader positioned at
    /// the first sample of the `data` chunk.
    pub fn new(mut reader: R) -> io::Result<Self> {
        let mut fourcc = [0u8; 4];

        // ---- RIFF container header: "RIFF" <size> "WAVE" ----
        reader.read_exact(&mut fourcc)?;
        if &fourcc != b"RIFF" {
            return Err(invalid_data("not a RIFF file (missing 'RIFF' magic)"));
        }
        read_u32(&mut reader)?; // total size, not needed for streaming
        reader.read_exact(&mut fourcc)?;
        if &fourcc != b"WAVE" {
            return Err(invalid_data("not a WAVE file (missing 'WAVE' magic)"));
        }

        // ---- Walk the chunks until "data" is found ----
        // A conforming file has "fmt " before "data"; other chunks (LIST,
        // fact, ...) are skipped.
        let mut format_seen = false;
        loop {
            if let Err(e) = reader.read_exact(&mut fourcc) {
                return if e.kind() == io::ErrorKind::UnexpectedEof {
                    Err(invalid_data("no 'data' chunk found"))
                } else {
                    Err(e)
                };
            }
            let chunk_size = read_u32(&mut reader)?;

            match &fourcc {
                b"fmt " => {
                    if chunk_size < 16 {
                        return Err(invalid_data("'fmt ' chunk is too short"));
                    }
                    let format_tag = read_u16(&mut reader)?;
                    let channels = read_u16(&mut reader)?;
                    let sample_rate = read_u32(&mut reader)?;
                    let _byte_rate = read_u32(&mut reader)?;
                    let _block_align = read_u16(&mut reader)?;
                    let bits_per_sample = read_u16(&mut reader)?;
                    skip(&mut reader, chunk_size as u64 - 16)?;

                    if format_tag != FORMAT_TAG_PCM {
                        return Err(invalid_data(format!(
                            "unsupported WAV format tag {format_tag} (only uncompressed PCM = 1 is supported)"
                        )));
                    }
                    if channels != REQUIRED_CHANNELS {
                        return Err(invalid_data(format!(
                            "expected {REQUIRED_CHANNELS} channel, found {channels}"
                        )));
                    }
                    if sample_rate != REQUIRED_SAMPLE_RATE {
                        return Err(invalid_data(format!(
                            "expected {REQUIRED_SAMPLE_RATE} Hz, found {sample_rate} Hz"
                        )));
                    }
                    if bits_per_sample != REQUIRED_BITS_PER_SAMPLE {
                        return Err(invalid_data(format!(
                            "expected {REQUIRED_BITS_PER_SAMPLE} bits per sample, found {bits_per_sample}"
                        )));
                    }
                    format_seen = true;
                }
                b"data" => {
                    if !format_seen {
                        return Err(invalid_data("'data' chunk appears before 'fmt '"));
                    }
                    let samples = chunk_size as u64 / 2;
                    return Ok(WavReader {
                        reader,
                        remaining: samples,
                        total: samples,
                        error: None,
                    });
                }
                _ => {
                    // Unknown chunk: skip its payload, plus the pad byte that
                    // RIFF inserts after odd-sized chunks.
                    skip(&mut reader, chunk_size as u64 + (chunk_size as u64 & 1))?;
                }
            }
        }
    }

    /// Total number of samples in the file.
    pub fn len(&self) -> u64 {
        self.total
    }

    /// Returns the read error that stopped the iteration, if any.
    pub fn take_error(&mut self) -> Option<io::Error> {
        self.error.take()
    }
}

impl<R: Read> Iterator for WavReader<R> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.remaining == 0 || self.error.is_some() {
            return None;
        }
        let mut bytes = [0u8; 2];
        match self.reader.read_exact(&mut bytes) {
            Ok(()) => {
                self.remaining -= 1;
                Some(sample_from_i16(i16::from_le_bytes(bytes)))
            }
            Err(e) => {
                // Truncated or unreadable file: remember why we stopped.
                self.error = Some(e);
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sink
// ---------------------------------------------------------------------------

/// Streaming writer for 16-bit mono PCM WAV files.
///
/// The header contains two length fields that are only known once all samples
/// have been written, so a placeholder header is written first and patched in
/// [`WavWriter::finish`].
pub struct WavWriter {
    writer: BufWriter<File>,
    samples_written: u64,
}

/// Size of the fixed header this writer emits: 12 bytes of RIFF/WAVE, a
/// 24-byte `fmt ` chunk and an 8-byte `data` chunk header.
const HEADER_LEN: u64 = 44;

impl WavWriter {
    /// Creates the file and writes a placeholder header.
    pub fn create(path: &Path) -> io::Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        write_header(&mut writer, 0)?;
        Ok(WavWriter {
            writer,
            samples_written: 0,
        })
    }

    /// Appends one sample.
    pub fn write_sample(&mut self, sample: f32) -> io::Result<()> {
        self.writer
            .write_all(&sample_to_i16(sample).to_le_bytes())?;
        self.samples_written += 1;
        Ok(())
    }

    /// Flushes the data and patches the two size fields in the header.
    pub fn finish(mut self) -> io::Result<u64> {
        self.writer.flush()?;
        self.writer.seek(SeekFrom::Start(0))?;
        write_header(&mut self.writer, self.samples_written)?;
        self.writer.flush()?;
        Ok(self.samples_written)
    }
}

/// The sink of the pipeline: drains an iterator of samples into a WAV file.
///
/// This is what actually drives the whole chain - every filter upstream is
/// lazy and computes nothing until the sink asks for the next sample.
pub fn write_wav<I>(path: &Path, samples: I) -> io::Result<u64>
where
    I: Iterator<Item = f32>,
{
    let mut writer = WavWriter::create(path)?;
    for sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finish()
}

fn write_header(writer: &mut impl Write, samples: u64) -> io::Result<()> {
    let data_bytes = (samples * 2) as u32;
    let byte_rate = REQUIRED_SAMPLE_RATE * REQUIRED_CHANNELS as u32 * (REQUIRED_BITS_PER_SAMPLE as u32 / 8);
    let block_align = REQUIRED_CHANNELS * (REQUIRED_BITS_PER_SAMPLE / 8);

    writer.write_all(b"RIFF")?;
    // Size of everything after this field.
    writer.write_all(&((HEADER_LEN as u32 - 8) + data_bytes).to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?; // chunk size of a plain PCM fmt
    writer.write_all(&FORMAT_TAG_PCM.to_le_bytes())?;
    writer.write_all(&REQUIRED_CHANNELS.to_le_bytes())?;
    writer.write_all(&REQUIRED_SAMPLE_RATE.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&REQUIRED_BITS_PER_SAMPLE.to_le_bytes())?;

    writer.write_all(b"data")?;
    writer.write_all(&data_bytes.to_le_bytes())?;
    Ok(())
}

fn read_u16(reader: &mut impl Read) -> io::Result<u16> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

/// Skips `count` bytes without requiring `Seek`, so that the reader also works
/// on non-seekable streams.
fn skip(reader: &mut impl Read, mut count: u64) -> io::Result<()> {
    let mut scratch = [0u8; 512];
    while count > 0 {
        let chunk = count.min(scratch.len() as u64) as usize;
        reader.read_exact(&mut scratch[..chunk])?;
        count -= chunk as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i16_round_trip_is_exact() {
        // Every 16-bit value must survive i16 -> f32 -> i16 unchanged,
        // otherwise "source connected straight to sink" would not be a copy.
        for raw in i16::MIN..=i16::MAX {
            assert_eq!(sample_to_i16(sample_from_i16(raw)), raw);
        }
    }

    #[test]
    fn out_of_range_samples_are_clipped() {
        assert_eq!(sample_to_i16(2.0), i16::MAX);
        assert_eq!(sample_to_i16(-2.0), i16::MIN);
    }
}
