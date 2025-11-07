# ML Visualizer 🤖

<div align="center">

[![License](https://img.shields.io/badge/📄_License-GPL--3.0-blue?style=for-the-badge)](LICENSE.md)
[![Made with Rust](https://img.shields.io/badge/🦀_Made_with-Rust-orange?style=for-the-badge)](https://www.rust-lang.org/)
[![WebAssembly](https://img.shields.io/badge/⚡_WebAssembly-654FF0?style=for-the-badge&logo=webassembly&logoColor=white)](https://webassembly.org/)

**Interactive demonstrations of AI and machine learning architectures**

[🚀 Launch Demo](https://mlvisualizer.org/) • [📖 Documentation](#learn-more) • [🎵 Music Transformer](https://mlvisualizer.org/neural-music.html)

</div>

---

## Usage

1. 🌐 **Launch** the visualizer by opening `index.html`, or visit **[mlvisualizer.org](https://mlvisualizer.org/)**
2. 🎯 **Select** a demo from the homepage
3. 🎛️ **Interact** with the controls to adjust parameters
4. 👀 **Watch** the neural network learn and adapt in real-time

## Learn More

Each demo includes:
- 📊 **Real-time visualizations** - Watch neural networks in action
- 🎛️ **Interactive controls** - Adjust parameters and see immediate results
- 📈 **Performance metrics** - Track learning progress and accuracy
- 💡 **Educational descriptions** - Understand the theory behind each model

## Tech Stack

| Technology | Purpose |
|------------|---------|
| 🌐 **HTML5** | Structure & Markup |
| 🎨 **CSS3** | Styling & Animations |
| ⚡ **Vanilla JavaScript** | Interactive Visualizations |
| 🦀 **Rust + WebAssembly** | High-Performance AI Audio Generation |

---

## File structure

Repository tree (top-level). Use this as a quick reference:

```
MLvisualizer/
├── .gitignore               # Files and patterns ignored by Git
├── .nojekyll                # Disable Jekyll processing on GitHub Pages
├── .github/                 # GitHub actions, issue templates, and CI configs
├── Cargo.toml               # Rust crate manifest (dependencies and metadata)
├── Cargo.lock               # Locked dependency versions for reproducible builds
├── LICENSE.md               # Project license (GPL-3.0)
├── sitemap.xml              # Project sitemap for search engines
├── sitemap.xsl              # Stylesheet for sitemap.xml
├── README.md                # Project overview and documentation (this file)
├── index.html               # Main landing page for the demo
├── CNAME                    # Custom domain name for GitHub Pages
├── robots.txt               # Search engine crawling instructions
├── css/                     # Stylesheets
│   ├── styles.css           # Global site styles
│   └── neural-music.css     # Styles for the music demo
├── html/                    # Standalone HTML demo pages and fragments
│   ├── neural-music.html    # Music Transformer demo page (audio + visualization)
│   └── jslicense.html       # License fragment included in HTML pages
├── js/                      # Front-end JavaScript (visualizers and site scripts)
│   ├── core.js              # Core visualization helpers and utilities
│   ├── script.js            # Site initialization and glue code
│   ├── neural-music.js      # Music demo UI and integration with wasm pkg
│   ├── visualizers-basic.js # Basic visualization implementations
│   └── visualizers-advanced.js # Advanced visualization implementations
├── pkg/                     # Generated WebAssembly artifacts and JS wrappers
│   ├── music_transformer.js # Browser import wrapper for the wasm module
│   ├── music_transformer.d.ts # TypeScript definitions for the wrapper
│   ├── music_transformer_bg.wasm # Compiled wasm binary
│   ├── music_transformer_bg.wasm.d.ts # Wasm type declarations
│   └── package.json         # pkg metadata for the generated package
├── src/                     # Rust source code for the Music Transformer
│   └── lib.rs               # Core Rust implementation and wasm bindings
├── target/                  # Cargo build artifacts and compiled outputs
└── tools/                   # Build and maintenance utilities
	├── build.sh             # Local build/packaging helper script
	└── CODE_OF_CONDUCT.md   # Contribution guidelines and conduct policy
```

Notes:

- `pkg/` and `target/` are generated build outputs. Do not edit files in these folders directly; change the source files under `src/`, `js/`, `css/`, and `html/` and regenerate artifacts via the build scripts.
- `pkg/` contains the WebAssembly wrapper and related artifacts used by the browser; it includes generated binaries, JS wrappers, and TypeScript definitions. The `bark.wav` file is an easter-egg and is not needed for normal development or packaging.

## Neural Music Transformer Demo 

This project includes a real, working transformer-like sequence model implemented in Rust and compiled to WebAssembly for the browser. It uses an attention mechanism over previously generated notes to produce short melodic phrases, and then synthesizes audio samples on the fly.

Important context:

- **Educational Model**: A minimal, hand-crafted model for education and fun. There's no training, no large parameter matrices, and no text tokens.
- **Real Transformer**: It does use attention over a sequence, so it's "a real transformer" in spirit, but it's tiny and domain-specific (8-note pentatonic scale + rests), nothing like the multi-billion-parameter models powering systems like ChatGPT.
- **Deterministic Output**: Generated deterministically with small randomness from the browser's `Math.random()`.

### What it does

- Generates a musical note sequence with simple attention: recent positions get higher weight, and consonant intervals are biased
- Optionally inserts rests to create phrases and supports different envelope "instruments"
- Converts the sequence to audio samples (Float32) you can play with the Web Audio API

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

		// Configuration setters
		pub fn set_melodic(&mut self, melodic: bool);
		pub fn set_random_spacing(&mut self, on: bool);
		pub fn set_instrument(&mut self, instrument: InstrumentType);
		pub fn set_tempo(&mut self, bpm: f32);              // Clamped to [60, 240]
		pub fn set_sample_rate(&mut self, rate: f32);       // Clamped to [22050, 48000]
		pub fn set_target_duration(&mut self, seconds: f32); // Clamped to [5, 30]

		// Configuration getters
		pub fn get_sample_rate(&self) -> f32;
		pub fn get_target_duration(&self) -> f32;
		pub fn get_sequence(&self) -> Vec<usize>;           // Returns current sequence
		pub fn get_duration(&self) -> f32;                  // Returns actual duration in seconds

		// Generation methods
		pub fn generate_sequence(&mut self) -> Vec<usize>;  // Returns note indices (0..7) and rest marker 999
		pub fn generate_audio(&self) -> Vec<f32>;           // Returns mono audio samples for current sequence
}
```

**Notes:**

- Sequence values of `999` represent rests (silence)
- Instruments are simple envelopes/harmonics: `Robo` (synth), `Piano`, `Guitar`
- Attention weights emphasize nearby positions; "melodic" mode increases preference for small intervals

### 🔬 How It Differs from Large LLMs

| Aspect | This Demo | Large LLMs (e.g., ChatGPT) |
|--------|-----------|----------------------------|
| 📏 **Scale** | A few functions and tiny arrays running in your browser | Billions of parameters on GPU/TPU clusters |
| 🎓 **Training** | Not trained; rule-guided | Trained on massive datasets |
| 🎭 **Modality** | Outputs notes and synthesized waveforms | Operates on text tokens (and sometimes images/audio) using very large vocabularies |

This demo is designed to help you peek inside the mechanics - **attention**, **sequencing**, and **synthesis** - without the complexity of production-grade models.

---

## 📄 License

GNU General Public License v3.0 - See [`LICENSE.MD`](LICENSE.MD) for details.

---

<div align="center">

**Built with ❤️ for CEE 4803 at Georgia Tech**

🌟 [Star this repo](https://github.com/Fedele-AI/MLvisualizer) if you find it helpful!

</div>