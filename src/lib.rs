use wasm_bindgen::prelude::*;
use std::f32::consts::PI;

const NUM_NOTES: usize = 8;
const REST_NOTE: usize = 999; // Special value to represent silence/rest

// External function to get random seed from JavaScript
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = Math, js_name = random)]
    fn js_random() -> f64;
}

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum InstrumentType {
    // Why the inconsistency? Because I'm stupid - and didn't realize that a
    // web browser transformer obviously can't play an actual piano or guitar :)
    // Well, too late now - so we swapped Piano to "80s" and Guitar to "Old Nokia"
    // The names remain for code clarity.
    Robo = 0,
    Piano = 1,
    Guitar = 2,
}

// Simple music transformer that generates melodic sequences
#[wasm_bindgen]
pub struct MusicTransformer {
    // Note probabilities (C, D, E, F, G, A, B, C')
    note_frequencies: [f32; NUM_NOTES],
    // Current sequence state
    current_sequence: Vec<usize>,
    // Tempo in BPM
    tempo: f32,
    // Melodic mode flag
    melodic_mode: bool,
    // Random spacing flag (adds rests between phrases)
    random_spacing: bool,
    // Instrument type
    instrument: InstrumentType,
    // Sample rate (Hz)
    sample_rate: f32,
    // Target duration (seconds)
    target_duration: f32,
}

#[wasm_bindgen]
impl MusicTransformer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> MusicTransformer {
        // Initialize with pentatonic scale frequencies (C major pentatonic)
        let base_freq = 261.63; // Middle C
        let note_frequencies = [
            base_freq,           // C
            base_freq * 1.125,   // D
            base_freq * 1.260,   // E
            base_freq * 1.498,   // G
            base_freq * 1.682,   // A
            base_freq * 2.0,     // C (octave)
            base_freq * 2.245,   // D (octave)
            base_freq * 2.520,   // E (octave)
        ];

        MusicTransformer {
            note_frequencies,
            current_sequence: Vec::new(),
            tempo: 120.0,
            melodic_mode: false,
            random_spacing: true,
            instrument: InstrumentType::Robo,
            sample_rate: 44100.0,
            target_duration: 12.0,
        }
    }

    /// Calculate sequence length based on tempo to achieve target duration
    fn get_sequence_length(&self) -> usize {
        // beat_duration = 60.0 / tempo (seconds per beat)
        // note_duration = beat_duration / 2.0 (eighth notes)
        // sequence_length = target_duration / note_duration
        let beat_duration = 60.0 / self.tempo;
        let note_duration = beat_duration / 2.0;
        let length = (self.target_duration / note_duration).ceil() as usize;
        length.max(16) // Minimum 16 notes
    }

    /// Generate attention weights for a given sequence length
    fn generate_attention_weights(&self, seq_length: usize) -> Vec<Vec<f32>> {
        let mut attention_weights = Vec::new();
        for i in 0..seq_length {
            let mut weights = Vec::new();
            for j in 0..seq_length {
                // Closer positions have higher attention
                let distance = ((i as f32 - j as f32).abs() + 1.0).recip();
                weights.push(distance);
            }
            attention_weights.push(weights);
        }
        attention_weights
    }

    /// Set melodic mode
    #[wasm_bindgen]
    pub fn set_melodic(&mut self, melodic: bool) {
        self.melodic_mode = melodic;
    }

    /// Set random spacing mode
    #[wasm_bindgen]
    pub fn set_random_spacing(&mut self, random_spacing: bool) {
        self.random_spacing = random_spacing;
    }

    /// Set the instrument type
    #[wasm_bindgen]
    pub fn set_instrument(&mut self, instrument: InstrumentType) {
        self.instrument = instrument;
    }

    /// Set the tempo in beats per minute
    #[wasm_bindgen]
    pub fn set_tempo(&mut self, bpm: f32) {
        self.tempo = bpm.max(60.0).min(240.0);
    }

    /// Set the sample rate
    #[wasm_bindgen]
    pub fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate.max(22050.0).min(48000.0);
    }

    /// Set the target duration
    #[wasm_bindgen]
    pub fn set_target_duration(&mut self, duration: f32) {
        self.target_duration = duration.max(5.0).min(30.0);
    }

    /// Get the sample rate
    #[wasm_bindgen]
    pub fn get_sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Get the target duration
    #[wasm_bindgen]
    pub fn get_target_duration(&self) -> f32 {
        self.target_duration
    }

    /// Generate a new musical sequence using transformer-like attention
    #[wasm_bindgen]
    pub fn generate_sequence(&mut self) -> Vec<usize> {
        let mut sequence = Vec::new();
        
        // Calculate sequence length based on tempo to maintain 12 second duration
        let sequence_length = self.get_sequence_length();
        
        // Generate attention weights for this sequence length
        let attention_weights = self.generate_attention_weights(sequence_length);
        
        // Start with a random note - use JS random for true randomness in browser
        let start_note = (js_random() * NUM_NOTES as f64) as usize % NUM_NOTES;
        sequence.push(start_note);

        // Track phrase position for adding rests
        let mut notes_since_rest = 1;
        let phrase_length = 4 + (js_random() * 4.0) as usize; // Random phrase length 4-7 notes

        // Generate rest of sequence using attention mechanism
        for pos in 1..sequence_length {
            // Add rests to create phrases (every 4-8 notes) - only if random_spacing is enabled
            if self.random_spacing && notes_since_rest >= phrase_length && js_random() < 0.6 {
                // Add 1-2 rest notes
                let rest_count = if js_random() < 0.7 { 1 } else { 2 };
                for _ in 0..rest_count {
                    sequence.push(REST_NOTE);
                }
                notes_since_rest = 0;
                continue;
            }

            let mut note_probs = vec![0.0; NUM_NOTES];
            
            // Apply attention to previous notes (skip rests)
            for (prev_pos, &prev_note) in sequence.iter().enumerate() {
                if prev_note == REST_NOTE {
                    continue; // Skip rest notes in attention calculation
                }
                
                let attention = attention_weights[pos][prev_pos];
                
                // Favor consonant intervals
                for note in 0..NUM_NOTES {
                    let interval = (note as i32 - prev_note as i32).abs();
                    
                    let consonance = if self.melodic_mode {
                        // Melodic mode: stronger preference for smooth stepwise motion
                        match interval {
                            0 => 0.8,      // Unison (less repetition)
                            1 | 2 => 1.0,  // Steps (highly preferred)
                            3 => 0.85,     // Thirds (good)
                            4 => 0.7,      // Fourth
                            5 => 0.6,      // Fifth
                            7 => 0.5,      // Octave
                            _ => 0.3,      // Discourage large leaps
                        }
                    } else {
                        // Non-melodic: more harmonic/rhythmic variety
                        match interval {
                            0 => 1.0,      // Unison
                            2 | 3 => 0.9,  // Thirds
                            4 => 0.95,     // Fourth
                            5 => 0.8,      // Fifth
                            7 => 0.95,     // Octave
                            _ => 0.5,
                        }
                    };
                    note_probs[note] += attention * consonance;
                }
            }

            // Normalize probabilities
            let sum: f32 = note_probs.iter().sum();
            if sum > 0.0 {
                for prob in note_probs.iter_mut() {
                    *prob /= sum;
                }
            }

            // Sample from distribution - use JS random for true randomness
            let rand_val = js_random() as f32;
            let mut cumulative = 0.0;
            let mut selected_note = 0;
            for (note, &prob) in note_probs.iter().enumerate() {
                cumulative += prob;
                if rand_val <= cumulative {
                    selected_note = note;
                    break;
                }
            }
            
            // Add slight random variation to ensure uniqueness - use JS random
            if js_random() < 0.15 {
                selected_note = (js_random() * NUM_NOTES as f64) as usize % NUM_NOTES;
            }

            sequence.push(selected_note);
            notes_since_rest += 1;
        }

        self.current_sequence = sequence.clone();
        sequence
    }

    /// Generate audio samples for the current sequence
    #[wasm_bindgen]
    pub fn generate_audio(&self) -> Vec<f32> {
        if self.current_sequence.is_empty() {
            return vec![0.0; (self.sample_rate * 0.1) as usize];
        }

        let beat_duration = 60.0 / self.tempo;
        let note_duration = beat_duration / 2.0; // Eighth notes (changed from 16th for longer duration)
        let samples_per_note = (self.sample_rate * note_duration) as usize;
        
        // Calculate actual duration based on sequence length and tempo
        let total_samples = self.current_sequence.len() * samples_per_note;
        let mut samples = vec![0.0; total_samples];

        for (seq_idx, &note_idx) in self.current_sequence.iter().enumerate() {
            let start_sample = seq_idx * samples_per_note;
            if start_sample >= total_samples {
                break;
            }

            // Skip if this is a rest note
            if note_idx == REST_NOTE {
                continue; // Leave samples as 0.0 (silence)
            }

            let frequency = self.note_frequencies[note_idx];
            let end_sample = (start_sample + samples_per_note).min(total_samples);
            
            // Add slight frequency variation for uniqueness
            let freq_variation = 1.0 + ((seq_idx as f32 * 0.001).sin() * 0.002);
            let varied_frequency = frequency * freq_variation;

            // Generate note with ADSR envelope
            for i in start_sample..end_sample {
                let t = (i - start_sample) as f32 / self.sample_rate;
                let note_progress = (i - start_sample) as f32 / samples_per_note as f32;
                
                // ADSR envelope (instrument-specific)
                let envelope = match self.instrument {
                    InstrumentType::Piano => {
                        // Piano: fast attack, medium decay, quick release
                        if note_progress < 0.05 {
                            note_progress / 0.05
                        } else if note_progress < 0.2 {
                            1.0 - (note_progress - 0.05) * 0.4 / 0.15
                        } else if note_progress < 0.85 {
                            0.6 * (1.0 - (note_progress - 0.2) * 0.3 / 0.65)
                        } else {
                            0.42 * (1.0 - (note_progress - 0.85) / 0.15).powf(2.0)
                        }
                    },
                    InstrumentType::Guitar => {
                        // Guitar: medium attack, sustained, gentle release
                        if note_progress < 0.08 {
                            note_progress / 0.08
                        } else if note_progress < 0.15 {
                            1.0 - (note_progress - 0.08) * 0.2 / 0.07
                        } else if note_progress < 0.88 {
                            0.8
                        } else {
                            0.8 * (1.0 - (note_progress - 0.88) / 0.12).powf(1.5)
                        }
                    },
                    InstrumentType::Robo => {
                        // Robo/Synth: original envelope
                        if note_progress < 0.1 {
                            note_progress / 0.1
                        } else if note_progress < 0.3 {
                            1.0 - (note_progress - 0.1) * 0.3 / 0.2
                        } else if note_progress < 0.9 {
                            0.7
                        } else {
                            0.7 * (1.0 - (note_progress - 0.9) / 0.1)
                        }
                    }
                };

                // Generate waveform based on instrument
                let phase = 2.0 * PI * varied_frequency * t;
                let sample = match self.instrument {
                    InstrumentType::Piano => {
                        // Piano: rich harmonics with decreasing amplitudes
                        let fundamental = phase.sin() * 0.5;
                        let harmonic2 = (phase * 2.0).sin() * 0.25;
                        let harmonic3 = (phase * 3.0).sin() * 0.15;
                        let harmonic4 = (phase * 4.0).sin() * 0.08;
                        let harmonic5 = (phase * 5.0).sin() * 0.04;
                        (fundamental + harmonic2 + harmonic3 + harmonic4 + harmonic5) * envelope * 0.35
                    },
                    InstrumentType::Guitar => {
                        // Guitar: prominent odd harmonics, slight vibrato
                        let vibrato = (t * 6.0).sin() * 0.003;
                        let vibrato_phase = 2.0 * PI * varied_frequency * (1.0 + vibrato) * t;
                        let fundamental = vibrato_phase.sin() * 0.6;
                        let harmonic2 = (vibrato_phase * 2.0).sin() * 0.15;
                        let harmonic3 = (vibrato_phase * 3.0).sin() * 0.25;
                        let harmonic5 = (vibrato_phase * 5.0).sin() * 0.12;
                        (fundamental + harmonic2 + harmonic3 + harmonic5) * envelope * 0.4
                    },
                    InstrumentType::Robo => {
                        // Robo/Synth: original sound
                        (phase.sin() * 0.6 
                            + (phase * 2.0).sin() * 0.2 
                            + (phase * 3.0).sin() * 0.1) * envelope * 0.3
                    }
                };
                
                samples[i] += sample;
            }
        }

        // Normalize
        let max_amplitude = samples.iter().cloned().fold(0.0f32, f32::max);
        if max_amplitude > 0.0 {
            for sample in samples.iter_mut() {
                *sample /= max_amplitude;
                *sample *= 0.8; // Leave headroom
            }
        }

        samples
    }
    
    /// Get the duration of the current sequence in seconds
    #[wasm_bindgen]
    pub fn get_duration(&self) -> f32 {
        if self.current_sequence.is_empty() {
            return 0.0;
        }
        let beat_duration = 60.0 / self.tempo;
        let note_duration = beat_duration / 2.0; // Eighth notes (same as in generate_audio)
        self.current_sequence.len() as f32 * note_duration
    }

    /// Get the current sequence
    #[wasm_bindgen]
    pub fn get_sequence(&self) -> Vec<usize> {
        self.current_sequence.clone()
    }
}
