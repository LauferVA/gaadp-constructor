# Project Specification: WAV File Synthesizer

## 1. Objective
Create a Python script `synth.py` that generates a valid `.wav` file from scratch without using `wave`, `scipy`, or `audio` libraries. You must construct the header bytes manually.

## 2. Functional Requirements
* **Input:** A frequency (e.g., 440Hz) and duration (e.g., 2 seconds).
* **Signal Generation:**
    * Sample Rate: 44100 Hz.
    * Bit Depth: 16-bit (Signed Short).
    * Formula: Amplitude × sin(2π × Frequency × Time)
* **Output:** A file named `output.wav`.

## 3. The Core Challenge (The Header)
A WAV file consists of a 44-byte header followed by raw PCM data.
You must use `struct.pack` to write:
1.  `RIFF` (4 bytes)
2.  File Size (4 bytes integer)
3.  `WAVE` (4 bytes)
4.  `fmt ` (4 bytes)
5.  ...and so on (Channels, Sample Rate, Byte Rate, Block Align, Bits Per Sample).
6.  `data` (4 bytes)
7.  Data Size (4 bytes)

## 4. Acceptance Criteria
* **Verification:** The generated file must play a clear tone in a standard media player (VLC/QuickTime).
* **Fidelity:** No clicking or static (implies correct handling of 16-bit signed integer wrapping).
* **Math:** 440Hz must sound like A4.

## 5. Research Instructions
1.  Search "WAV file header specification" (canonical layout).
2.  Research how to map a sine wave float value (-1.0 to 1.0) to a 16-bit signed integer (-32767 to 32767).
3.  Little Endian vs Big Endian: WAV is typically Little Endian. Ensure `struct.pack` uses `<`.
