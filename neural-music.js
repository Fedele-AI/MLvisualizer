import init, { MusicTransformer, InstrumentType } from './pkg/music_transformer.js';

let transformer = null;
let audioContext = null;
let audioUnlocked = false;
let isGenerating = false;
let isInitialized = false;
let isInitializing = false;
let initPromise = null; // Store the initialization promise for reuse
let isChangingInstrument = false; // Flag to prevent generation during instrument change
let wasmInitialized = false; // Track if WASM module itself is loaded
let wasmInitPromise = null; // Store WASM init promise to prevent double-init
let currentSource = null;
let currentBuffer = null;
let startTime = 0;
let pausedAt = 0;
let isPlaying = false;
let animationFrame = null;
let isDragging = false;
let neuralViz = null;
let currentInstrument = 0; // 0 = Robo, 1 = Piano, 2 = Guitar
let currentSampleRate = 44100;
let currentTargetDuration = 12;

// Ensure an unlocked/running AudioContext exists
async function ensureAudioContext() {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!audioContext) {
        audioContext = new Ctx();
    }
    try {
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }
    } catch (_) { /* no-op */ }

    // Play one-frame silent buffer once to fully unlock
    if (!audioUnlocked) {
        try {
            const silent = audioContext.createBuffer(1, 1, audioContext.sampleRate);
            const src = audioContext.createBufferSource();
            src.buffer = silent;
            src.connect(audioContext.destination);
            src.start(0);
            audioUnlocked = true;
        } catch (_) { /* ignore */ }
    }
    return audioContext;
}

// Linear resampler to match generated audio to the AudioContext sample rate
function resampleFloat32(input, fromRate, toRate) {
    if (!input || fromRate === toRate) return input;
    const ratio = toRate / fromRate;
    const outLen = Math.max(1, Math.round(input.length * ratio));
    const output = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
        const pos = i / ratio;
        const i0 = Math.floor(pos);
        const i1 = Math.min(i0 + 1, input.length - 1);
        const frac = pos - i0;
        output[i] = input[i0] * (1 - frac) + input[i1] * frac;
    }
    return output;
}

// Draw waveform visualization
function drawWaveform(buffer) {
    const canvas = document.getElementById('waveform-canvas');
    if (!canvas || !buffer) return;
    
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    // Set canvas size
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    const width = rect.width;
    const height = rect.height;
    const channelData = buffer.getChannelData(0);
    const step = Math.ceil(channelData.length / width);
    const amp = height / 2;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw gradient background
    const bgGradient = ctx.createLinearGradient(0, 0, 0, height);
    bgGradient.addColorStop(0, 'rgba(102, 126, 234, 0.05)');
    bgGradient.addColorStop(0.5, 'rgba(118, 75, 162, 0.08)');
    bgGradient.addColorStop(1, 'rgba(102, 126, 234, 0.05)');
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, width, height);
    
    // Draw center line
    ctx.strokeStyle = 'rgba(102, 126, 234, 0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, amp);
    ctx.lineTo(width, amp);
    ctx.stroke();
    
    // Draw waveform with gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, 'rgba(102, 126, 234, 0.8)');
    gradient.addColorStop(0.5, 'rgba(118, 75, 162, 0.9)');
    gradient.addColorStop(1, 'rgba(102, 126, 234, 0.8)');
    
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    ctx.beginPath();
    for (let i = 0; i < width; i++) {
        let min = 1.0;
        let max = -1.0;
        
        for (let j = 0; j < step; j++) {
            const datum = channelData[(i * step) + j];
            if (datum < min) min = datum;
            if (datum > max) max = datum;
        }
        
        const x = i;
        const y1 = (1 + min) * amp;
        const y2 = (1 + max) * amp;
        
        if (i === 0) {
            ctx.moveTo(x, amp);
        } else {
            ctx.lineTo(x, y1);
            ctx.lineTo(x, y2);
        }
    }
    ctx.stroke();
    
    // Draw filled area under waveform
    ctx.globalAlpha = 0.3;
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.moveTo(0, amp);
    for (let i = 0; i < width; i++) {
        let max = -1.0;
        for (let j = 0; j < step; j++) {
            const datum = channelData[(i * step) + j];
            if (datum > max) max = datum;
        }
        const y = (1 + max) * amp;
        ctx.lineTo(i, y);
    }
    ctx.lineTo(width, amp);
    ctx.closePath();
    ctx.fill();
    
    ctx.globalAlpha = 1.0;
}

// Draw playhead on waveform
function drawWaveformPlayhead(progress) {
    const canvas = document.getElementById('waveform-canvas');
    if (!canvas || !currentBuffer) return;
    
    // Redraw the waveform
    drawWaveform(currentBuffer);
    
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    
    // Draw playhead position
    const x = (progress / 100) * width;
    
    // Playhead line
    ctx.strokeStyle = 'rgba(236, 72, 153, 0.8)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
    
    // Playhead indicator at top
    ctx.fillStyle = 'rgba(236, 72, 153, 0.9)';
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x - 6, 12);
    ctx.lineTo(x + 6, 12);
    ctx.closePath();
    ctx.fill();
}

class NeuralNetworkVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.setupCanvas();
        
        // 4-layer transformer architecture
        this.layers = [
            { neurons: 8, x: 0.15, label: 'Input', color: [102, 126, 234] },
            { neurons: 16, x: 0.35, label: 'Attention 1', color: [139, 92, 246] },
            { neurons: 16, x: 0.65, label: 'Attention 2', color: [236, 72, 153] },
            { neurons: 8, x: 0.85, label: 'Output', color: [59, 130, 246] }
        ];
        
        this.neurons = [];
        this.connections = [];
        this.activeConnections = new Set();
        this.firingNeurons = new Set();
        this.isGenerating = false;
        this.currentSequenceIndex = 0;
        this.particles = [];
        this.attentionFlows = [];
        
        this.initializeNetwork();
        this.startIdleAnimation();
    }
    
    setupCanvas() {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        this.width = rect.width;
        this.height = rect.height;
    }
    
    initializeNetwork() {
        // Create neurons
        this.layers.forEach((layer, layerIdx) => {
            const spacing = this.height / (layer.neurons + 1);
            for (let i = 0; i < layer.neurons; i++) {
                this.neurons.push({
                    x: layer.x * this.width,
                    y: spacing * (i + 1),
                    layer: layerIdx,
                    index: i,
                    activation: 0,
                    targetActivation: 0,
                    pulsePhase: Math.random() * Math.PI * 2,
                    color: layer.color
                });
            }
        });
        
        // Create connections between layers
        for (let l = 0; l < this.layers.length - 1; l++) {
            const currentLayer = this.neurons.filter(n => n.layer === l);
            const nextLayer = this.neurons.filter(n => n.layer === l + 1);
            
            currentLayer.forEach(n1 => {
                nextLayer.forEach(n2 => {
                    this.connections.push({
                        from: n1,
                        to: n2,
                        weight: Math.random() * 0.5 + 0.5,
                        pulse: 0,
                        targetPulse: 0,
                        flowProgress: 0
                    });
                });
            });
        }
    }
    
    startIdleAnimation() {
        const animate = () => {
            if (!this.isGenerating) {
                this.neurons.forEach((neuron, i) => {
                    const phase = Date.now() * 0.0015 + i * 0.3;
                    const wave = Math.sin(phase) * 0.12 + 0.15;
                    neuron.targetActivation = wave;
                    neuron.pulsePhase += 0.02;
                });
                this.updateInfo(0, 0);
            }
            this.render();
            requestAnimationFrame(animate);
        };
        animate();
    }
    
    updateInfo(sequenceIndex, activeNeurons, totalLength) {
        totalLength = totalLength || 96; // Default fallback
        document.getElementById('info-sequence').textContent = `${sequenceIndex}/${totalLength}`;
        document.getElementById('info-active').textContent = activeNeurons;
    }
    
    createParticle(x, y, color) {
        this.particles.push({
            x, y,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2,
            life: 1.0,
            color,
            size: Math.random() * 3 + 1
        });
    }
    
    async generateSequence(sequence) {
        this.isGenerating = true;
        this.currentSequenceIndex = 0;
        const totalLength = sequence.length;
        const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));
        
        // Process each note
        for (let i = 0; i < sequence.length; i++) {
            this.currentSequenceIndex = i + 1;
            const note = sequence[i];
            
            // Fire input neuron with emphasis
            const inputLayer = this.neurons.filter(n => n.layer === 0);
            const inputNeuron = inputLayer[note % inputLayer.length];
            inputNeuron.targetActivation = 1.0;
            this.firingNeurons.add(inputNeuron);
            
            // Create particles at input
            for (let p = 0; p < 3; p++) {
                this.createParticle(inputNeuron.x, inputNeuron.y, inputNeuron.color);
            }
            
            await delay(20);
            
            // Propagate through layers with attention mechanism
            for (let layer = 0; layer < this.layers.length - 1; layer++) {
                const currentNeurons = this.neurons.filter(n => n.layer === layer);
                const nextNeurons = this.neurons.filter(n => n.layer === layer + 1);
                
                // Calculate attention weights
                const attentionWeights = new Map();
                currentNeurons.forEach(from => {
                    if (from.activation > 0.2) {
                        this.connections.forEach(conn => {
                            if (conn.from === from && nextNeurons.includes(conn.to)) {
                                const attention = Math.random() * 0.6 + 0.4;
                                conn.targetPulse = 1.0;
                                conn.flowProgress = 0;
                                this.activeConnections.add(conn);
                                
                                // Create attention flow
                                this.attentionFlows.push({
                                    from: conn.from,
                                    to: conn.to,
                                    progress: 0,
                                    speed: 0.08 + Math.random() * 0.04,
                                    intensity: attention,
                                    color: from.color
                                });
                                
                                const currentWeight = attentionWeights.get(conn.to) || 0;
                                attentionWeights.set(conn.to, currentWeight + attention);
                                this.firingNeurons.add(conn.to);
                            }
                        });
                    }
                });
                
                // Apply attention weights to next layer
                attentionWeights.forEach((weight, neuron) => {
                    neuron.targetActivation = Math.min(1.0, weight);
                    // Create particles at highly activated neurons
                    if (weight > 0.7) {
                        this.createParticle(neuron.x, neuron.y, neuron.color);
                    }
                });
                
                await delay(35);
            }
            
            this.updateInfo(this.currentSequenceIndex, this.firingNeurons.size, totalLength);
            
            // Smooth decay
            await delay(15);
            this.neurons.forEach(n => {
                n.targetActivation *= 0.75;
                if (n.targetActivation < 0.1) this.firingNeurons.delete(n);
            });
            
            this.connections.forEach(conn => {
                conn.targetPulse *= 0.7;
                if (conn.targetPulse < 0.1) this.activeConnections.delete(conn);
            });
        }
        
        // Final smooth decay
        await delay(400);
        this.neurons.forEach(n => n.targetActivation = 0.15);
        this.firingNeurons.clear();
        this.activeConnections.clear();
        this.attentionFlows = [];
        this.isGenerating = false;
        this.updateInfo(totalLength, 0, totalLength);
    }
    
    render() {
        // Dark background with subtle gradient
        const bgGradient = this.ctx.createLinearGradient(0, 0, this.width, this.height);
        bgGradient.addColorStop(0, '#0a0e27');
        bgGradient.addColorStop(0.5, '#0d1135');
        bgGradient.addColorStop(1, '#0a0e27');
        this.ctx.fillStyle = bgGradient;
        this.ctx.fillRect(0, 0, this.width, this.height);
        
        // Smooth neuron activation transitions
        this.neurons.forEach(n => {
            n.activation += (n.targetActivation - n.activation) * 0.2;
        });
        
        // Smooth connection pulse transitions
        this.connections.forEach(conn => {
            conn.pulse += (conn.targetPulse - conn.pulse) * 0.25;
        });
        
        // Update and render particles
        this.particles = this.particles.filter(particle => {
            particle.x += particle.vx;
            particle.y += particle.vy;
            particle.life -= 0.02;
            particle.vx *= 0.98;
            particle.vy *= 0.98;
            
            if (particle.life > 0) {
                const alpha = particle.life * 0.8;
                const gradient = this.ctx.createRadialGradient(
                    particle.x, particle.y, 0,
                    particle.x, particle.y, particle.size * 2
                );
                const [r, g, b] = particle.color;
                gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${alpha})`);
                gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
                this.ctx.fillStyle = gradient;
                this.ctx.beginPath();
                this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
                this.ctx.fill();
                return true;
            }
            return false;
        });
        
        // Update and render attention flows
        this.attentionFlows = this.attentionFlows.filter(flow => {
            flow.progress += flow.speed;
            
            if (flow.progress < 1.0) {
                const x = flow.from.x + (flow.to.x - flow.from.x) * flow.progress;
                const y = flow.from.y + (flow.to.y - flow.from.y) * flow.progress;
                
                // Draw flowing particle
                const size = 3 + flow.intensity * 2;
                const alpha = flow.intensity * (1 - Math.abs(flow.progress - 0.5) * 0.5);
                const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, size * 2);
                const [r, g, b] = flow.color;
                gradient.addColorStop(0, `rgba(255, 255, 255, ${alpha})`);
                gradient.addColorStop(0.5, `rgba(${r}, ${g}, ${b}, ${alpha * 0.8})`);
                gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);
                this.ctx.fillStyle = gradient;
                this.ctx.beginPath();
                this.ctx.arc(x, y, size, 0, Math.PI * 2);
                this.ctx.fill();
                
                return true;
            }
            return false;
        });
        
        // Draw connections with improved styling
        this.connections.forEach(conn => {
            if (conn.pulse < 0.02) return;
            
            const alpha = conn.pulse * 0.5 + 0.03;
            const gradient = this.ctx.createLinearGradient(
                conn.from.x, conn.from.y, conn.to.x, conn.to.y
            );
            
            if (conn.pulse > 0.15) {
                const [r1, g1, b1] = conn.from.color;
                const [r2, g2, b2] = conn.to.color;
                gradient.addColorStop(0, `rgba(${r1}, ${g1}, ${b1}, ${alpha})`);
                gradient.addColorStop(0.5, `rgba(167, 139, 250, ${alpha * 1.5})`);
                gradient.addColorStop(1, `rgba(${r2}, ${g2}, ${b2}, ${alpha})`);
                this.ctx.strokeStyle = gradient;
                this.ctx.lineWidth = 0.8 + conn.pulse * 2.5;
                
                // Add glow for strong connections
                this.ctx.shadowBlur = 8 * conn.pulse;
                this.ctx.shadowColor = `rgba(167, 139, 250, ${conn.pulse * 0.5})`;
            } else {
                const [r, g, b] = conn.from.color;
                this.ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha * 0.5})`;
                this.ctx.lineWidth = 0.5;
                this.ctx.shadowBlur = 0;
            }
            
            this.ctx.beginPath();
            this.ctx.moveTo(conn.from.x, conn.from.y);
            
            // Add slight curve to connections for visual interest
            const midX = (conn.from.x + conn.to.x) / 2;
            const midY = (conn.from.y + conn.to.y) / 2;
            const offset = (Math.random() - 0.5) * 10 * conn.pulse;
            this.ctx.quadraticCurveTo(midX, midY + offset, conn.to.x, conn.to.y);
            this.ctx.stroke();
            
            this.ctx.shadowBlur = 0;
        });
        
        // Draw neurons with enhanced visuals
        this.neurons.forEach(neuron => {
            const baseSize = 3;
            const pulse = Math.sin(neuron.pulsePhase) * 0.15 + 1;
            const size = baseSize + neuron.activation * 9 * pulse;
            const alpha = 0.25 + neuron.activation * 0.75;
            
            // Outer glow for highly activated neurons
            if (neuron.activation > 0.3) {
                const glowSize = size * 3;
                const glowGradient = this.ctx.createRadialGradient(
                    neuron.x, neuron.y, 0, neuron.x, neuron.y, glowSize
                );
                const [r, g, b] = neuron.color;
                glowGradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${neuron.activation * 0.4})`);
                glowGradient.addColorStop(0.5, `rgba(${r}, ${g}, ${b}, ${neuron.activation * 0.2})`);
                glowGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
                this.ctx.fillStyle = glowGradient;
                this.ctx.beginPath();
                this.ctx.arc(neuron.x, neuron.y, glowSize, 0, Math.PI * 2);
                this.ctx.fill();
            }
            
            // Main neuron body with gradient
            const gradient = this.ctx.createRadialGradient(
                neuron.x - size * 0.3, neuron.y - size * 0.3, 0,
                neuron.x, neuron.y, size
            );
            const [r, g, b] = neuron.color;
            gradient.addColorStop(0, `rgba(255, 255, 255, ${alpha})`);
            gradient.addColorStop(0.3, `rgba(${r + 50}, ${g + 50}, ${b + 50}, ${alpha})`);
            gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, ${alpha * 0.9})`);
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(neuron.x, neuron.y, size, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Firing indicator ring
            if (this.firingNeurons.has(neuron) && neuron.activation > 0.5) {
                const ringAlpha = neuron.activation * 0.9;
                this.ctx.strokeStyle = `rgba(255, 255, 255, ${ringAlpha})`;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.arc(neuron.x, neuron.y, size + 3, 0, Math.PI * 2);
                this.ctx.stroke();
                
                // Inner bright core
                this.ctx.fillStyle = `rgba(255, 255, 255, ${ringAlpha * 0.6})`;
                this.ctx.beginPath();
                this.ctx.arc(neuron.x, neuron.y, size * 0.4, 0, Math.PI * 2);
                this.ctx.fill();
            }
        });
        
        // Draw layer labels with better styling
        this.ctx.font = 'bold 11px -apple-system, sans-serif';
        this.ctx.textAlign = 'center';
        this.layers.forEach(layer => {
            const [r, g, b] = layer.color;
            this.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.7)`;
            this.ctx.fillText(layer.label, layer.x * this.width, this.height - 5);
        });
    }
}

// Ensure WASM module is initialized (can be called multiple times safely)
async function ensureWasmInit() {
    // If already initialized, return immediately
    if (wasmInitialized) {
        return;
    }
    
    // If currently initializing, wait for the existing promise
    if (wasmInitPromise) {
        return wasmInitPromise;
    }
    
    // Create new initialization promise
    wasmInitPromise = (async () => {
        try {
            console.log('🚀 Initializing WASM module...');
            await init();
            wasmInitialized = true;
            console.log('✅ WASM module loaded');
        } catch (error) {
            wasmInitialized = false;
            wasmInitPromise = null; // Allow retry
            throw error;
        }
    })();
    
    return wasmInitPromise;
}

async function initApp() {
    // If already initialized, just return
    if (isInitialized) {
        return true;
    }
    
    // If currently initializing, return the existing promise
    if (isInitializing && initPromise) {
        return initPromise;
    }
    
    const status = document.getElementById('status');
    const generateBtn = document.getElementById('generate-btn');
    const loadingOverlay = document.getElementById('wasm-loading-overlay');
    const progressFill = document.getElementById('wasm-progress-fill');
    
    // Set initializing state
    isInitializing = true;
    
    // Show loading overlay
    if (loadingOverlay) {
        loadingOverlay.classList.add('active');
    }
    
    // Disable controls during initialization
    generateBtn.disabled = true;
    status.textContent = 'Loading neural network...';
    status.classList.add('generating');
    
    // Simulate progress bar (since we can't track WASM loading progress exactly)
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress = Math.min(progress + Math.random() * 15, 90);
        if (progressFill) {
            progressFill.style.width = `${progress}%`;
        }
    }, 200);
    
    // Create the initialization promise
    initPromise = (async () => {
        let retryCount = 0;
        const maxRetries = 5;
        
        while (retryCount < maxRetries) {
            try {
                // Initialize WASM module with timeout (only if not already initialized)
                console.log(`WASM initialization attempt ${retryCount + 1}/${maxRetries}`);
                
                // Use the shared WASM init function with timeout
                const wasmInitWithTimeout = Promise.race([
                    ensureWasmInit(),
                    new Promise((_, reject) => 
                        setTimeout(() => reject(new Error('WASM initialization timeout')), 10000)
                    )
                ]);
                
                await wasmInitWithTimeout;
                
                // Update progress
                if (progressFill) progressFill.style.width = '95%';
                
                // Create transformer instance (always create a fresh one)
                if (transformer) {
                    console.log('⚠️ Disposing old transformer instance');
                    // Dispose of the old transformer if it exists
                    try {
                        if (transformer.free) {
                            transformer.free();
                        }
                    } catch (e) {
                        console.warn('Could not free old transformer:', e);
                    }
                }
                
                transformer = new MusicTransformer();
                
                // Verify transformer is working
                if (!transformer) {
                    throw new Error('Transformer creation failed');
                }
                
                // Verify critical methods exist
                const requiredMethods = ['set_tempo', 'set_melodic', 'set_random_spacing', 'set_instrument', 
                                          'set_sample_rate', 'set_target_duration', 'generate_sequence', 'generate_audio'];
                for (const method of requiredMethods) {
                    if (typeof transformer[method] !== 'function') {
                        throw new Error(`Transformer missing required method: ${method}`);
                    }
                }
                
                console.log('✅ Transformer instance created successfully');
                
                // Initialize transformer with default values to ensure it's in a good state
                try {
                    transformer.set_tempo(120);
                    transformer.set_melodic(false);
                    transformer.set_random_spacing(true);
                    transformer.set_instrument(0);
                    transformer.set_sample_rate(44100);
                    transformer.set_target_duration(12);
                    console.log('✅ Transformer initialized with default settings');
                } catch (defaultsError) {
                    console.error('Failed to set default values:', defaultsError);
                    throw new Error('Transformer initialization failed - could not set defaults');
                }
                
                // Initialize visualization
                const canvas = document.getElementById('neural-canvas');
                if (canvas) {
                    neuralViz = new NeuralNetworkVisualizer(canvas);
                }
                
                // Complete progress bar
                if (progressFill) progressFill.style.width = '100%';
                
                // Mark as successfully initialized
                isInitialized = true;
                isInitializing = false;
                
                // Clear progress interval
                clearInterval(progressInterval);
                
                // Hide loading overlay with smooth transition
                setTimeout(() => {
                    if (loadingOverlay) {
                        loadingOverlay.classList.remove('active');
                    }
                }, 300);
                
                status.textContent = 'Ready to generate music';
                status.classList.remove('generating');
                generateBtn.disabled = false;
                
                updateTransformerStats();
                setupScrubber();
                
                console.log('✅ WASM module initialized successfully');
                return true;
                
            } catch (error) {
                retryCount++;
                console.error(`❌ WASM initialization attempt ${retryCount} failed:`, error);
                
                if (retryCount < maxRetries) {
                    status.textContent = `Retrying... (${retryCount}/${maxRetries})`;
                    // Exponential backoff: 500ms, 1s, 2s, 4s
                    const delay = Math.min(500 * Math.pow(2, retryCount - 1), 4000);
                    await new Promise(resolve => setTimeout(resolve, delay));
                } else {
                    clearInterval(progressInterval);
                    if (loadingOverlay) {
                        loadingOverlay.classList.remove('active');
                    }
                    status.textContent = '❌ Failed to load. Please refresh the page.';
                    status.classList.remove('generating');
                    status.style.color = '#ef4444';
                    generateBtn.disabled = true;
                    isInitialized = false;
                    isInitializing = false;
                    console.error('💥 Failed to initialize WASM after all retries');
                    return false;
                }
            }
        }
        return false;
    })();
    
    return initPromise;
}

function updateTransformerStats() {
    if (!transformer) {
        console.warn('⚠️ updateTransformerStats called but transformer is null');
        return;
    }
    
    try {
        const tempo = parseInt(document.getElementById('tempo').value);
        const melodic = document.getElementById('melodic-mode').checked;
        const instrumentNames = ['Robo', '80s', 'Old Nokia'];
        
        // Calculate derived values
        const beatDuration = 60.0 / tempo;
        const noteDuration = beatDuration / 2.0; // Eighth notes
        const sequenceLength = Math.max(16, Math.ceil(currentTargetDuration / noteDuration));
        const totalSamples = sequenceLength * Math.floor(currentSampleRate * noteDuration);
        
        // Update all stat fields
        document.getElementById('stat-tempo').textContent = tempo;
        document.getElementById('stat-instrument').textContent = instrumentNames[currentInstrument];
        document.getElementById('stat-mode').textContent = melodic ? 'Melodic' : 'Harmonic';
        document.getElementById('stat-sample-rate').textContent = currentSampleRate.toLocaleString() + ' Hz';
        document.getElementById('stat-seq-length').textContent = sequenceLength;
        document.getElementById('stat-duration').textContent = (sequenceLength * noteDuration).toFixed(2) + 's';
        document.getElementById('stat-note-duration').textContent = noteDuration.toFixed(3) + 's';
        document.getElementById('stat-total-samples').textContent = totalSamples.toLocaleString();
        document.getElementById('stat-num-notes').textContent = '8 (Pentatonic)';
        document.getElementById('stat-target-duration').textContent = currentTargetDuration.toFixed(1) + 's';
    } catch (error) {
        console.error('❌ Error updating transformer stats:', error);
    }
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

async function togglePlayPause() {
    if (!currentBuffer) return;
    
    const playPauseIcon = document.getElementById('play-pause-icon');
    
    if (isPlaying) {
        // Pause
        if (currentSource) {
            currentSource.stop();
            currentSource = null;
            pausedAt = audioContext.currentTime - startTime;
        }
        
        isPlaying = false;
        playPauseIcon.textContent = '▶';
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
        }
    } else {
        // Play/Resume
        const elapsed = pausedAt || parseFloat(document.getElementById('current-time').textContent.split(':').reduce((acc, val) => acc * 60 + parseFloat(val), 0));
        await ensureAudioContext();
        await playFromPosition(elapsed);
        playPauseIcon.textContent = '⏸';
    }
}

function updateScrubber() {
    if (!isPlaying || !currentBuffer) return;
    
    const elapsed = audioContext.currentTime - startTime;
    const duration = currentBuffer.duration;
    const progress = Math.min((elapsed / duration) * 100, 100);
    
    document.getElementById('scrubber-progress').style.width = `${progress}%`;
    document.getElementById('scrubber-handle').style.left = `${progress}%`;
    document.getElementById('current-time').textContent = formatTime(elapsed);
    
    // Update waveform playhead
    drawWaveformPlayhead(progress);
    
    if (elapsed < duration && isPlaying) {
        animationFrame = requestAnimationFrame(updateScrubber);
    } else if (elapsed >= duration) {
        isPlaying = false;
        document.getElementById('play-pause-icon').textContent = '▶';
        document.getElementById('status').textContent = 'Playback complete - Ready to generate again';
        document.getElementById('status').classList.remove('generating');
        // Re-enable generate button immediately after playback finishes
        document.getElementById('generate-btn').disabled = false;
        isGenerating = false;
    }
}

function setupScrubber() {
    const track = document.getElementById('scrubber-track');
    const handle = document.getElementById('scrubber-handle');
    
    function seek(e) {
        if (!currentBuffer) return;
        
        const rect = track.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percentage = Math.max(0, Math.min(1, x / rect.width));
        const seekTime = percentage * currentBuffer.duration;
        
        // Stop current playback
        if (currentSource) {
            try {
                currentSource.stop();
            } catch(e) {}
            currentSource = null;
        }
        
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
        }
        
        playFromPosition(seekTime);
    }
    
    // Touch events for mobile
    let touchStartX = 0;
    
    handle.addEventListener('touchstart', (e) => {
        isDragging = true;
        touchStartX = e.touches[0].clientX;
        e.preventDefault();
    }, { passive: false });
    
    handle.addEventListener('mousedown', (e) => {
        isDragging = true;
        e.preventDefault();
    });
    
    document.addEventListener('touchmove', (e) => {
        if (isDragging && currentBuffer) {
            const rect = track.getBoundingClientRect();
            const x = e.touches[0].clientX - rect.left;
            const percentage = Math.max(0, Math.min(1, x / rect.width));
            const seekTime = percentage * currentBuffer.duration;
            
            // Update UI immediately
            const progress = percentage * 100;
            document.getElementById('scrubber-progress').style.width = `${progress}%`;
            document.getElementById('scrubber-handle').style.left = `${progress}%`;
            document.getElementById('current-time').textContent = formatTime(seekTime);
        }
    }, { passive: false });
    
    document.addEventListener('mousemove', (e) => {
        if (isDragging && currentBuffer) {
            const rect = track.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const percentage = Math.max(0, Math.min(1, x / rect.width));
            const seekTime = percentage * currentBuffer.duration;
            
            // Update UI immediately
            const progress = percentage * 100;
            document.getElementById('scrubber-progress').style.width = `${progress}%`;
            document.getElementById('scrubber-handle').style.left = `${progress}%`;
            document.getElementById('current-time').textContent = formatTime(seekTime);
        }
    });
    
    document.addEventListener('touchend', (e) => {
        if (isDragging && currentBuffer) {
            const rect = track.getBoundingClientRect();
            const x = touchStartX - rect.left;
            const percentage = Math.max(0, Math.min(1, x / rect.width));
            const seekTime = percentage * currentBuffer.duration;
            
            isDragging = false;
            playFromPosition(seekTime);
        }
    });
    
    document.addEventListener('mouseup', () => {
        if (isDragging && currentBuffer) {
            isDragging = false;
        }
    });
    
    track.addEventListener('click', seek);
    track.addEventListener('touchend', (e) => {
        if (!isDragging && e.target === track) {
            seek(e.changedTouches[0]);
        }
    });
}

async function playFromPosition(startOffset) {
    if (!currentBuffer) return;
    
    // Stop any existing playback
    if (currentSource) {
        try {
            currentSource.stop();
        } catch(e) {}
        currentSource = null;
    }
    
    // Use Web Audio API for all browsers
    await ensureAudioContext();
    
    currentSource = audioContext.createBufferSource();
    currentSource.buffer = currentBuffer;
    currentSource.connect(audioContext.destination);
    currentSource.start(0, startOffset);
    
    startTime = audioContext.currentTime - startOffset;
    isPlaying = true;
    
    // Update play/pause button
    document.getElementById('play-pause-icon').textContent = '⏸';
    document.getElementById('play-pause-btn').disabled = false;
    
    currentSource.onended = () => {
        if (audioContext.currentTime - startTime >= currentBuffer.duration - 0.1) {
            isPlaying = false;
            document.getElementById('play-pause-icon').textContent = '▶';
            document.getElementById('status').textContent = 'Playback complete - Ready to generate again';
            document.getElementById('status').classList.remove('generating');
            document.getElementById('generate-btn').disabled = false;
            isGenerating = false;
        }
    };
    
    updateScrubber();
}

function exportToWav() {
    if (!currentBuffer) return;
    
    const numberOfChannels = 1;
    const sampleRate = currentBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    
    const channelData = currentBuffer.getChannelData(0);
    const samples = new Int16Array(channelData.length);
    
    // Convert float32 to int16
    for (let i = 0; i < channelData.length; i++) {
        const s = Math.max(-1, Math.min(1, channelData[i]));
        samples[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    
    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, format, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numberOfChannels * bitDepth / 8, true);
    view.setUint16(32, numberOfChannels * bitDepth / 8, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    
    // Write audio data
    for (let i = 0; i < samples.length; i++) {
        view.setInt16(44 + i * 2, samples[i], true);
    }
    
    // Create download
    const blob = new Blob([buffer], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    a.download = `neural-music-${timestamp}.wav`;
    
    document.body.appendChild(a);
    a.click();
    
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 100);
}

async function generateAndPlay() {
    // Ensure WASM is initialized before proceeding
    if (!isInitialized) {
        console.log('⏳ WASM not ready, waiting for initialization...');
        const status = document.getElementById('status');
        status.textContent = 'Initializing... please wait';
        await initApp();
        
        // Check if initialization succeeded
        if (!isInitialized || !transformer) {
            status.textContent = 'Failed to initialize. Please refresh the page.';
            return;
        }
    }
    
    // Don't allow generation while changing instruments
    if (isChangingInstrument) {
        console.log('⏳ Waiting for instrument change to complete...');
        return;
    }
    
    if (isGenerating) return;
    
    isGenerating = true;
    const button = document.getElementById('generate-btn');
    const status = document.getElementById('status');
    const scrubber = document.getElementById('time-scrubber');

    // Clean up any existing playback
    if (currentSource) {
        try {
            currentSource.stop();
        } catch(e) {}
        currentSource = null;
    }
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
    }
    
    pausedAt = 0;
    isPlaying = false;
    scrubber.classList.remove('visible');
    button.disabled = true;
    status.textContent = 'Generating sequence...';
    status.classList.add('generating');

    try {
        // Ensure settings are applied before generation
        const tempo = parseInt(document.getElementById('tempo').value);
        const melodic = document.getElementById('melodic-mode').checked;
        const randomSpacing = document.getElementById('random-spacing-mode').checked;
        
        // Apply all settings to the transformer
        try {
            transformer.set_tempo(tempo);
            transformer.set_melodic(melodic);
            transformer.set_random_spacing(randomSpacing);
            transformer.set_instrument(currentInstrument);
            transformer.set_sample_rate(currentSampleRate);
            transformer.set_target_duration(currentTargetDuration);
        } catch (settingsError) {
            console.error('Error applying transformer settings:', settingsError);
            throw new Error('Failed to configure transformer. Please refresh the page.');
        }
        
        // Generate unique sequence each time (truly random)
        console.log('Generating sequence with settings:', { tempo, melodic, randomSpacing, currentInstrument, currentSampleRate, currentTargetDuration });
        
        let sequence;
        try {
            sequence = transformer.generate_sequence();
        } catch (seqError) {
            console.error('WASM error during sequence generation:', seqError);
            throw new Error('Failed to generate sequence. Try changing instrument or refreshing the page.');
        }
        
        if (!sequence || sequence.length === 0) {
            throw new Error('Failed to generate sequence - sequence is empty');
        }
        
        console.log('Sequence generated:', sequence.length, 'notes');
        
        status.textContent = 'Neural network processing...';
        await neuralViz.generateSequence(sequence);
        
        status.textContent = 'Synthesizing audio...';
        
        // Generate audio - duration is calculated from sequence length and tempo
        let audioSamples;
        try {
            audioSamples = transformer.generate_audio();
        } catch (audioError) {
            console.error('WASM error during audio generation:', audioError);
            throw new Error('Failed to generate audio. Try changing instrument or refreshing the page.');
        }
        
        if (!audioSamples || audioSamples.length === 0) {
            throw new Error('Failed to generate audio - audio samples are empty');
        }
        
        const duration = transformer.get_duration();
        console.log('Audio generated:', audioSamples.length, 'samples, duration:', duration.toFixed(2), 's');
        
        // Create AudioBuffer for playback
        // Initialize AudioContext if needed
        if (!audioContext) {
            const Ctx = window.AudioContext || window.webkitAudioContext;
            audioContext = new Ctx();
        }
        
        const targetRate = audioContext.sampleRate;
        const samplesForPlayback = (currentSampleRate === targetRate)
            ? audioSamples
            : resampleFloat32(audioSamples, currentSampleRate, targetRate);
        currentBuffer = audioContext.createBuffer(1, samplesForPlayback.length, targetRate);
        const channelData = currentBuffer.getChannelData(0);
        channelData.set(samplesForPlayback);
        
        // Draw waveform visualization
        drawWaveform(currentBuffer);
        
        // Update transformer stats with actual values
        document.getElementById('stat-seq-length').textContent = sequence.length;
        document.getElementById('stat-duration').textContent = duration.toFixed(2) + 's';
        document.getElementById('stat-total-samples').textContent = samplesForPlayback.length.toLocaleString();
        
        document.getElementById('total-time').textContent = formatTime(duration);
        document.getElementById('current-time').textContent = '0:00';
        document.getElementById('scrubber-progress').style.width = '0%';
        document.getElementById('scrubber-handle').style.left = '0%';
        
        status.textContent = 'Playing...';
        scrubber.classList.add('visible');
        
        // Re-enable the generate button immediately so user can generate another while this is playing
        button.disabled = false;
        isGenerating = false;
        
        await playFromPosition(0);
        
    } catch (error) {
        console.error('Error generating music:', error);
        status.textContent = 'Error occurred. Please try again.';
        status.classList.remove('generating');
        button.disabled = false;
        isGenerating = false;
    }
}

// Event listeners
document.getElementById('generate-btn').addEventListener('click', generateAndPlay);
document.getElementById('export-btn').addEventListener('click', exportToWav);
document.getElementById('play-pause-btn').addEventListener('click', togglePlayPause);

const tempoSlider = document.getElementById('tempo');
const tempoDisplay = document.getElementById('tempo-display');
const melodicCheckbox = document.getElementById('melodic-mode');
const randomSpacingCheckbox = document.getElementById('random-spacing-mode');
const sampleRateSlider = document.getElementById('sample-rate');
const sampleRateDisplay = document.getElementById('sample-rate-display');
const targetDurationSlider = document.getElementById('target-duration');
const targetDurationDisplay = document.getElementById('target-duration-display');

tempoSlider.addEventListener('input', async (e) => {
    const tempo = parseInt(e.target.value);
    tempoDisplay.textContent = tempo;
    if (!isInitialized) {
        await initApp();
    }
    if (transformer) {
        transformer.set_tempo(tempo);
        updateTransformerStats();
    }
});

melodicCheckbox.addEventListener('change', async (e) => {
    if (!isInitialized) {
        await initApp();
    }
    if (transformer) {
        transformer.set_melodic(e.target.checked);
        updateTransformerStats();
    }
});

randomSpacingCheckbox.addEventListener('change', async (e) => {
    if (!isInitialized) {
        await initApp();
    }
    if (transformer) {
        transformer.set_random_spacing(e.target.checked);
    }
});

sampleRateSlider.addEventListener('input', async (e) => {
    const sampleRate = parseInt(e.target.value);
    currentSampleRate = sampleRate;
    sampleRateDisplay.textContent = sampleRate.toLocaleString();
    if (!isInitialized) {
        await initApp();
    }
    if (transformer) {
        transformer.set_sample_rate(sampleRate);
        updateTransformerStats();
    }
});

targetDurationSlider.addEventListener('input', async (e) => {
    const duration = parseInt(e.target.value);
    currentTargetDuration = duration;
    targetDurationDisplay.textContent = duration;
    if (!isInitialized) {
        await initApp();
    }
    if (transformer) {
        transformer.set_target_duration(duration);
        updateTransformerStats();
    }
});

// One-time global unlock on first user interaction for AudioContext
const unlockHandler = async () => {
    await ensureAudioContext();
};
window.addEventListener('touchend', unlockHandler, { once: true, passive: true });
window.addEventListener('click', unlockHandler, { once: true });

// Instrument selector
document.querySelectorAll('.instrument-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
        console.log('🎸 Instrument button clicked:', e.target.dataset.instrument);
        
        // Prevent multiple simultaneous instrument changes
        if (isChangingInstrument) {
            console.log('⏳ Instrument change already in progress...');
            return;
        }
        
        const status = document.getElementById('status');
        const loadingOverlay = document.getElementById('wasm-loading-overlay');
        const progressFill = document.getElementById('wasm-progress-fill');
        
        // Ensure WASM is initialized before changing instrument
        if (!isInitialized) {
            console.log('⏳ Waiting for WASM initialization before changing instrument...');
            const previousText = status.textContent;
            status.textContent = 'Loading... please wait';
            
            await initApp();
            
            // Check if initialization succeeded
            if (!isInitialized || !transformer) {
                status.textContent = 'Failed to initialize. Please refresh the page.';
                return;
            }
            status.textContent = previousText;
        }
        
        // Double-check transformer exists
        if (!transformer) {
            console.error('❌ Transformer is null even though isInitialized is true');
            status.textContent = 'System error. Please refresh the page.';
            return;
        }
        
        // Set the flag to prevent music generation during instrument change
        isChangingInstrument = true;
        
        // Show loading overlay while changing instrument
        if (loadingOverlay) {
            loadingOverlay.classList.add('active');
            // Quick progress animation
            if (progressFill) progressFill.style.width = '0%';
        }
        
        // Stop any playing audio and clear the buffer
        if (currentSource) {
            try {
                currentSource.stop();
            } catch(e) {}
            currentSource = null;
        }
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
        }
        currentBuffer = null;
        isPlaying = false;
        pausedAt = 0;
        
        // Hide the scrubber since we cleared the audio
        const scrubber = document.getElementById('time-scrubber');
        if (scrubber) {
            scrubber.classList.remove('visible');
        }
        
        // Update status
        const previousStatusText = status.textContent;
        status.textContent = 'Changing instrument...';
        status.classList.add('generating');
        
        // Remove active class from all buttons
        document.querySelectorAll('.instrument-btn').forEach(b => b.classList.remove('active'));
        // Add active class to clicked button
        e.target.classList.add('active');
        
        // Set instrument in transformer
        const instrumentValue = parseInt(e.target.dataset.instrument);
        const instrumentNames = ['Robo', '80s', 'Old Nokia'];
        currentInstrument = instrumentValue;
        
        console.log('🎵 Setting instrument to:', instrumentValue, 'Transformer exists:', !!transformer);
        
        try {
            // Verify the transformer object has the method
            if (typeof transformer.set_instrument !== 'function') {
                throw new Error('set_instrument method not found on transformer');
            }
            
            // Animate progress
            if (progressFill) progressFill.style.width = '50%';
            
            // The WASM set_instrument expects a number (0, 1, 2)
            transformer.set_instrument(instrumentValue);
            
            // Give the transformer a moment to fully process the change
            // This prevents "index out of bounds" errors from transient state
            await new Promise(resolve => setTimeout(resolve, 150));
            
            // Complete progress
            if (progressFill) progressFill.style.width = '100%';
            
            updateTransformerStats();
            console.log(`✅ Instrument successfully changed to: ${instrumentValue} (${instrumentNames[instrumentValue]})`);
            
            // Update status with success message
            status.textContent = `Instrument changed to ${instrumentNames[instrumentValue]} - Ready to generate music`;
            status.classList.remove('generating');
            
            // Hide loading overlay after a brief moment
            setTimeout(() => {
                if (loadingOverlay) {
                    loadingOverlay.classList.remove('active');
                }
                // Reset status after showing success briefly
                setTimeout(() => {
                    status.textContent = 'Ready to generate music';
                    // Clear the flag - instrument change is complete
                    isChangingInstrument = false;
                }, 1500);
            }, 300);
            
        } catch (error) {
            console.error('❌ Error setting instrument:', error);
            console.error('Error name:', error.name);
            console.error('Error message:', error.message);
            console.error('Error stack:', error.stack);
            console.error('Transformer type:', typeof transformer);
            console.error('Transformer keys:', transformer ? Object.keys(transformer) : 'null');
            
            // Hide loading overlay on error
            if (loadingOverlay) {
                loadingOverlay.classList.remove('active');
            }
            
            status.textContent = `Error: ${error.message || 'Failed to change instrument'}`;
            status.classList.remove('generating');
            
            setTimeout(() => {
                status.textContent = 'Ready to generate music';
                // Clear the flag even on error
                isChangingInstrument = false;
            }, 3000);
        }
    });
});

// Info button handlers
const infoButton = document.getElementById('info-button');
const infoPopup = document.getElementById('info-popup');
const infoPopupClose = document.getElementById('info-popup-close');

infoButton.addEventListener('click', () => {
    infoPopup.classList.add('active');
});

infoPopupClose.addEventListener('click', () => {
    infoPopup.classList.remove('active');
});

// Close popup when clicking outside
infoPopup.addEventListener('click', (e) => {
    if (e.target === infoPopup) {
        infoPopup.classList.remove('active');
    }
});

// Mobile warning popup handlers
const mobileWarningPopup = document.getElementById('mobile-warning-popup');
const mobileWarningClose = document.getElementById('mobile-warning-close');

// Function to detect mobile devices
function isMobileDevice() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) 
        || (navigator.maxTouchPoints && navigator.maxTouchPoints > 2)
        || window.innerWidth <= 768;
}

// Show mobile warning on page load if on mobile
if (isMobileDevice()) {
    mobileWarningPopup.classList.add('active');
}

mobileWarningClose.addEventListener('click', () => {
    mobileWarningPopup.classList.remove('active');
});

// Close popup when clicking outside
mobileWarningPopup.addEventListener('click', (e) => {
    if (e.target === mobileWarningPopup) {
        mobileWarningPopup.classList.remove('active');
    }
});

// Redraw waveform on window resize
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        if (currentBuffer) {
            drawWaveform(currentBuffer);
            if (isPlaying) {
                const elapsed = audioContext.currentTime - startTime;
                const duration = currentBuffer.duration;
                const progress = Math.min((elapsed / duration) * 100, 100);
                drawWaveformPlayhead(progress);
            }
        }
    }, 100);
});

// Initialize WASM immediately when DOM is ready - start loading ASAP
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Start initialization immediately, don't wait
        initApp().catch(err => {
            console.error('Failed to initialize on DOMContentLoaded:', err);
        });
    });
} else {
    // DOM is already loaded (module scripts are deferred) - start immediately
    initApp().catch(err => {
        console.error('Failed to initialize immediately:', err);
    });
}

// Preload WASM module as early as possible (runs in parallel with DOM loading)
// This is just a performance optimization - initApp will handle actual initialization
(async () => {
    try {
        await ensureWasmInit();
        console.log('✅ WASM module preloaded successfully');
    } catch (error) {
        console.warn('⚠️ WASM preload failed (initApp will retry):', error);
    }
})();
