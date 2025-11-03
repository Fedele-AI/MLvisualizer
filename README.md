# ML Visualizer

A collection of interactive demonstrations of AI and machine learning architectures.

## 🎮 Usage

1. Launch the visualizer by opening `index.html`, or open [fedele-ai.github.io/MLvisualizer](https://fedele-ai.github.io/MLvisualizer/)
2. Select a demo from the homepage
3. Interact with the controls to adjust parameters
4. Watch the neural network learn and adapt in real-time

## 📚 Learn More

Each demo includes:
- 📊 Real-time visualizations
- 🎛️ Interactive controls
- 📈 Performance metrics
- ℹ️ Educational descriptions

## 🛠️ Tech Stack

- HTML5
- CSS3
- Vanilla JavaScript
- Rust (compiled to WebAssembly)

## Neural Music Transformer Demo 

This project includes a real, working transformer-like sequence model implemented in Rust and compiled to WebAssembly for the browser. It uses an attention mechanism over previously generated notes to produce short melodic phrases, and then synthesizes audio samples on the fly.

Important context:

- It is a minimal, hand-crafted model for education and fun. There’s no training, no large parameter matrices, and no text tokens.
- It does use attention over a sequence, so it’s “a real transformer” in spirit, but it’s tiny and domain-specific (8-note pentatonic scale + rests), nothing like the multi-billion-parameter models powering systems like ChatGPT.
- Output is generated deterministically with small randomness from the browser’s `Math.random()`.

### What it does

- Generates a musical note sequence with simple attention: recent positions get higher weight, and consonant intervals are biased.
- Optionally inserts rests to create phrases and supports different envelope “instruments”.
- Converts the sequence to audio samples (Float32) you can play with the Web Audio API.

### Quick start (browser, ESM)

```html
<script type="module">
	import init, { MusicTransformer, InstrumentType } from './pkg/music_transformer.js';

	async function main() {
		// Load the WASM module
		await init();

		// Create transformer and configure
		const mt = new MusicTransformer();
		mt.set_melodic(true);                 // smoother stepwise motion
		mt.set_random_spacing(true);          // insert rests between phrases
		mt.set_instrument(InstrumentType.Piano);
		mt.set_tempo(110);                    // BPM, clamped to [60, 240]
		mt.set_target_duration(12);           // seconds, clamped to [5, 30]

		// Use your AudioContext’s sample rate for perfect playback
		const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
		mt.set_sample_rate(audioCtx.sampleRate);

		// Generate music
		const sequence = mt.generate_sequence();
		// sequence is a Uint32Array of note indices (0..7) and rest markers (999)

		const samples = mt.generate_audio(); // Float32Array mono samples

		// Play via Web Audio API
		const buffer = audioCtx.createBuffer(1, samples.length, audioCtx.sampleRate);
		buffer.getChannelData(0).set(samples);
		const src = audioCtx.createBufferSource();
		src.buffer = buffer;
		src.connect(audioCtx.destination);
		src.start();
	}

	main();
	// Note: Browsers may require a user gesture before starting AudioContext
</script>
```

### Rust core (excerpt)

The Rust core exposes a `MusicTransformer` with a tiny attention-based generator and a simple synthesizer with ADSR envelopes per instrument:

```rust
#[wasm_bindgen]
pub struct MusicTransformer { /* ... */ }

#[wasm_bindgen]
impl MusicTransformer {
		#[wasm_bindgen(constructor)]
		pub fn new() -> MusicTransformer { /* init scale, defaults */ }

		pub fn set_melodic(&mut self, melodic: bool);
		pub fn set_random_spacing(&mut self, on: bool);
		pub fn set_instrument(&mut self, instrument: InstrumentType);
		pub fn set_tempo(&mut self, bpm: f32);
		pub fn set_sample_rate(&mut self, rate: f32);
		pub fn set_target_duration(&mut self, seconds: f32);

		// Generates a sequence of note indices (0..7) and rest marker 999
		pub fn generate_sequence(&mut self) -> Vec<usize>;

		// Renders mono audio samples for the current sequence
		pub fn generate_audio(&self) -> Vec<f32>;
}
```

Notes:

- Sequence values of `999` represent rests (silence).
- Instruments are simple envelopes/harmonics: `Robo` (synth), `Piano`, `Guitar`.
- Attention weights emphasize nearby positions; “melodic” mode increases preference for small intervals.

### How it differs from large LLMs

- Scale: This is a few functions and tiny arrays running in your browser. LLMs like ChatGPT use billions of parameters on GPU/TPU clusters.
- Training: This model is not trained; it’s rule-guided. LLMs are trained on massive datasets.
- Modality: This model outputs notes and synthesized waveforms; LLMs operate on text tokens (and sometimes images/audio) using very large vocabularies.

This demo is designed to help you peek inside the mechanics—attention, sequencing, and synthesis—without the complexity of production-grade models.

## 📄 License

GNU General Public License v3.0. See `LICENSE.MD` for details.