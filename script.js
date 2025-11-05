// @license magnet:?xt=urn:btih:1f739d935676111cfff4b4693e3816e664797050&dn=gpl-3.0.txt GPL-3.0
/**
 * ML Visualizer - Interactive demonstrations of AI and machine learning architectures
 * Copyright (C) 2024 Kenneth (Alex) Jenkins
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

// ====== Utility Functions ======
// Debounce function for resize events
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Smoothly scroll a demo canvas/section into view on desktop only
function scrollIntoViewIfDesktop(elOrId, opts = { behavior: 'smooth', block: 'center' }) {
    try {
        if (!window.matchMedia || !window.matchMedia('(min-width: 1025px)').matches) return;
        const el = typeof elOrId === 'string' ? document.getElementById(elOrId) : elOrId;
        if (el && typeof el.scrollIntoView === 'function') {
            el.scrollIntoView(opts);
        }
    } catch (_) { /* noop */ }
}

// ====== Global State ======
const state = {
    currentView: 'homepage', // homepage or demo
    currentDemo: null,
    perceptron: {
        running: false,
        speed: 3,
        epoch: 0
    },
    rbm: {
        running: false,
        speed: 3,
        phase: 'idle',
        animationProgress: 0
    },
    autoencoder: {
        running: false,
        speed: 3,
        phase: 'idle',
        animationProgress: 0
    },
    ising: {
        running: false,
        speed: 5,
        temperature: 2.27
    },
    hopfield: {
        patterns: [],
        currentState: null
    },
    transformer: {
        running: false,
        speed: 3,
        phase: 'idle'
    },
    deepPerceptron: {
        running: false,
        speed: 1.5,
        epoch: 0
    },
    normalizingFlow: {
        running: false,
        speed: 3,
        phase: 'idle'
    },
    vae: {
        running: false,
        speed: 3,
        phase: 'idle'
    },
    cnnEncoderDecoder: {
        running: false,
        speed: 3,
        phase: 'idle'
    },
    mamba2: {
        running: false,
        speed: 3,
        phase: 'idle'
    },
    cuda: {
        running: false,
        speed: 3,
        phase: 'idle'
    }
};

// ====== Homepage Navigation ======
document.addEventListener('DOMContentLoaded', () => {
    // Demo card click handlers
    document.querySelectorAll('.demo-card').forEach(card => {
        card.addEventListener('click', () => {
            const demo = card.dataset.demo;
            showDemo(demo);
        });
    });

    // Back button handler
    document.getElementById('back-to-home').addEventListener('click', () => {
        showHomepage();
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
});

function showHomepage() {
    document.getElementById('homepage').classList.add('active');
    document.getElementById('demo-view').classList.remove('active');
    
    // Stop all running animations
    state.perceptron.running = false;
    state.rbm.running = false;
    state.autoencoder.running = false;
    state.ising.running = false;
    state.transformer.running = false;
    state.deepPerceptron.running = false;
    state.normalizingFlow.running = false;
    state.vae.running = false;
    state.cnnEncoderDecoder.running = false;
    state.mamba2.running = false;
    state.cuda.running = false;
    
    state.currentView = 'homepage';
    state.currentDemo = null;
}

function showDemo(demoName) {
    document.getElementById('homepage').classList.remove('active');
    document.getElementById('demo-view').classList.add('active');
    
    // Hide all demos
    document.querySelectorAll('.visualization-section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Show selected demo
    document.getElementById(demoName).classList.add('active');
    
    state.currentView = 'demo';
    state.currentDemo = demoName;
    
    // Wait for DOM update, then initialize the specific demo's visualization
    setTimeout(() => {
        switch(demoName) {
            case 'perceptron':
                if (perceptronViz) {
                    perceptronViz.resize();
                    resetPerceptron();
                } else {
                    initPerceptron();
                }
                break;
            case 'rbm':
                if (rbmViz) {
                    rbmViz.resize();
                    resetRBM();
                } else {
                    initRBM();
                }
                break;
            case 'autoencoder':
                if (aencoderViz) {
                    aencoderViz.resize();
                    resetAutoencoder();
                } else {
                    initAutoencoder();
                }
                break;
            case 'ising':
                if (isingViz) {
                    isingViz.resize();
                    resetIsing();
                } else {
                    initIsing();
                }
                break;
            case 'hopfield':
                if (hopfieldViz) {
                    hopfieldViz.resize();
                    resetHopfield();
                } else {
                    initHopfield();
                }
                break;
            case 'transformer':
                if (transformerViz) {
                    transformerViz.resize();
                    resetTransformer();
                } else {
                    initTransformer();
                }
                break;
            case 'deep-perceptron':
                if (deepPerceptronViz) {
                    deepPerceptronViz.resize();
                    resetDeepPerceptron();
                } else {
                    initDeepPerceptron();
                }
                break;
            case 'normalizing-flow':
                if (normalizingFlowViz) {
                    normalizingFlowViz.resize();
                    resetNormalizingFlow();
                } else {
                    initNormalizingFlow();
                }
                break;
            case 'vae':
                if (vaeViz) {
                    vaeViz.resize();
                    resetVAE();
                } else {
                    initVAE();
                }
                break;
            case 'cnn-encoder-decoder':
                if (cnnViz) {
                    cnnViz.resize();
                    resetCNN();
                } else {
                    initCNNEncoderDecoder();
                }
                break;
            case 'mamba2':
                if (mamba2Viz) {
                    mamba2Viz.resize();
                    resetMamba2();
                } else {
                    initMamba2();
                }
                break;
            case 'cuda':
                if (cudaViz) {
                    cudaViz.resize();
                    resetCUDA();
                } else {
                    initCUDA();
                }
                break;
        }
    }, 10);
}

// ====== Reset Functions for Each Demo ======
function resetPerceptron() {
    state.perceptron.running = false;
    state.perceptron.epoch = 0;
    if (perceptronViz) {
        perceptronViz.currentSample = 0;
        perceptronViz.setupNodes();
        perceptronViz.generateTrainingData();
        perceptronViz.draw();
    }
    const startBtn = document.getElementById('perceptron-start');
    if (startBtn) startBtn.textContent = 'Start Training';
}

function resetRBM() {
    state.rbm.running = false;
    state.rbm.phase = 'idle';
    state.rbm.animationProgress = 0;
    if (rbmViz) {
        rbmViz.setupNodes();
        rbmViz.particles = [];
        rbmViz.draw();
    }
    const startBtn = document.getElementById('rbm-start');
    if (startBtn) startBtn.textContent = 'Start Training';
}

function resetAutoencoder() {
    state.autoencoder.running = false;
    state.autoencoder.phase = 'idle';
    state.autoencoder.animationProgress = 0;
    if (aencoderViz) {
        aencoderViz.setupLayers();
        aencoderViz.particles = [];
        aencoderViz.draw();
    }
    const startBtn = document.getElementById('ae-start');
    if (startBtn) startBtn.textContent = 'Start Encoding';
}

function resetIsing() {
    state.ising.running = false;
    if (isingViz) {
        isingViz.initializeSpins();
        isingViz.draw();
    }
    const startBtn = document.getElementById('ising-start');
    if (startBtn) startBtn.textContent = 'Start Simulation';
}

function resetHopfield() {
    state.hopfield.patterns = [];
    state.hopfield.currentState = null;
    if (hopfieldViz) {
        hopfieldViz.initializeState();
        hopfieldViz.patterns = [];
        hopfieldViz.weights = [];
        hopfieldViz.isRecalling = false;
        hopfieldViz.draw();
    }
    const patternDisplay = document.getElementById('hopfield-patterns');
    if (patternDisplay) patternDisplay.textContent = '0';
}

function resetTransformer() {
    state.transformer.running = false;
    state.transformer.phase = 'idle';
    const input = document.getElementById('transformer-input');
    if (input) input.value = '';
    const list = document.getElementById('transformer-topk-list');
    if (list) {
        list.innerHTML = '<li style="text-align: center; color: #999; padding: 20px; font-style: italic;">Type a prompt and click "Predict next token" to see the magic! ✨</li>';
    }
    const appendBtn = document.getElementById('transformer-append');
    if (appendBtn) appendBtn.disabled = true;
    const charCounter = document.getElementById('char-counter');
    if (charCounter) charCounter.textContent = '0/200';
    if (transformerViz) {
        transformerViz.draw();
    }
}

function resetDeepPerceptron() {
    state.deepPerceptron.running = false;
    state.deepPerceptron.epoch = 0;
    if (deepPerceptronViz) {
        deepPerceptronViz.setupLayers();
        deepPerceptronViz.particles = [];
        deepPerceptronViz.draw();
    }
    const startBtn = document.getElementById('deep-start');
    if (startBtn) startBtn.textContent = 'Start Training';
}

function resetNormalizingFlow() {
    state.normalizingFlow.running = false;
    state.normalizingFlow.phase = 'idle';
    if (normalizingFlowViz) {
        normalizingFlowViz.setupLayers();
        normalizingFlowViz.particles = [];
        normalizingFlowViz.draw();
    }
    const startBtn = document.getElementById('nf-start');
    if (startBtn) startBtn.textContent = 'Start Flow';
}

function resetVAE() {
    state.vae.running = false;
    state.vae.phase = 'idle';
    if (vaeViz) {
        vaeViz.setupLayers();
        vaeViz.particles = [];
        vaeViz.draw();
    }
    const startBtn = document.getElementById('vae-start');
    if (startBtn) startBtn.textContent = 'Start Encoding';
}

function resetCNN() {
    state.cnnEncoderDecoder.running = false;
    state.cnnEncoderDecoder.phase = 'idle';
    if (cnnViz) {
        cnnViz.setupMaps();
        cnnViz.particles = [];
        cnnViz.draw();
    }
    const startBtn = document.getElementById('cnn-start');
    if (startBtn) startBtn.textContent = 'Process Image';
}

function resetMamba2() {
    state.mamba2.running = false;
    state.mamba2.phase = 'idle';
    if (mamba2Viz) {
        mamba2Viz.setupLayers();
        mamba2Viz.particles = [];
        mamba2Viz.draw();
    }
    const startBtn = document.getElementById('mamba2-start');
    if (startBtn) startBtn.textContent = 'Process Sequence';
}

function resetCUDA() {
    state.cuda.running = false;
    state.cuda.phase = 'idle';
    if (cudaViz) {
        cudaViz.animationTime = 0;
        cudaViz.setupArchitecture();
        cudaViz.draw();
    }
    const startBtn = document.getElementById('cuda-start');
    if (startBtn) startBtn.textContent = 'Launch Kernel';
}

// ====== Perceptron Visualization ======
let perceptronViz = null;
let perceptronInitialized = false;

class PerceptronVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.inputs = [];
        this.output = null;
        this.weights = [];
        this.bias = Math.random() * 2 - 1;
        this.trainingData = [];
        this.currentSample = 0;
        
        this.resize();
        window.addEventListener('resize', debounce(() => this.resize(), 250));
    }
    
    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.setupNodes();
        this.generateTrainingData();
    }
    
    setupNodes() {
        const inputCount = 4;
        // Responsive node radius and padding based on canvas size
        const nodeRadius = Math.max(15, Math.min(30, this.canvas.width / 20));
        const padding = Math.max(50, Math.min(100, this.canvas.width / 8));
        
        this.inputs = [];
        this.weights = [];
        
        // Input layer (left)
        const inputX = padding;
        const inputSpacing = (this.canvas.height - 2 * padding) / (inputCount - 1);
        
        for (let i = 0; i < inputCount; i++) {
            this.inputs.push({
                x: inputX,
                y: padding + i * inputSpacing,
                radius: nodeRadius,
                value: Math.random()
            });
            this.weights.push(Math.random() * 2 - 1);
        }
        
        // Output node (right)
        this.output = {
            x: this.canvas.width - padding,
            y: this.canvas.height / 2,
            radius: nodeRadius + Math.min(10, nodeRadius * 0.3),
            value: 0,
            target: 0
        };
    }
    
    generateTrainingData() {
        this.trainingData = [];
        for (let i = 0; i < 100; i++) {
            const inputs = Array(this.inputs.length).fill(0).map(() => Math.random());
            // Simple linear separation: sum of inputs > 2 → class 1
            const target = inputs.reduce((a, b) => a + b, 0) > 2 ? 1 : 0;
            this.trainingData.push({ inputs, target });
        }
    }
    
    update() {
        if (!state.perceptron.running) return;
        
        // Get current training sample
        const sample = this.trainingData[this.currentSample];
        
        // Update input values
        this.inputs.forEach((input, i) => {
            input.value = sample.inputs[i];
        });
        
        // Compute output
        let sum = this.bias;
        this.inputs.forEach((input, i) => {
            sum += input.value * this.weights[i];
        });
        const predicted = sum > 0 ? 1 : 0;
        this.output.value = predicted;
        this.output.target = sample.target;
        
        // Update weights (perceptron learning rule)
        const error = sample.target - predicted;
        const learningRate = 0.1;
        
        if (error !== 0) {
            this.inputs.forEach((input, i) => {
                this.weights[i] += learningRate * error * input.value;
            });
            this.bias += learningRate * error;
        }
        
        // Move to next sample
        this.currentSample = (this.currentSample + 1) % this.trainingData.length;
        if (this.currentSample === 0) {
            state.perceptron.epoch++;
        }
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Responsive font sizes
        const baseFontSize = Math.max(10, Math.min(14, this.canvas.width / 40));
        const labelFontSize = Math.max(11, Math.min(16, this.canvas.width / 35));
        const epochFontSize = Math.max(14, Math.min(20, this.canvas.width / 30));
        
        // Draw connections with weights
        this.inputs.forEach((input, i) => {
            const weight = this.weights[i];
            const alpha = Math.min(Math.abs(weight) * 0.5, 1);
            this.ctx.strokeStyle = weight > 0 
                ? `rgba(67, 160, 71, ${alpha})` 
                : `rgba(229, 57, 53, ${alpha})`;
            this.ctx.lineWidth = Math.abs(weight) * 3;
            this.ctx.beginPath();
            this.ctx.moveTo(input.x + input.radius, input.y);
            this.ctx.lineTo(this.output.x - this.output.radius, this.output.y);
            this.ctx.stroke();
            
            // Draw weight value (only if canvas is wide enough)
            if (this.canvas.width > 400) {
                const midX = (input.x + this.output.x) / 2;
                const midY = (input.y + this.output.y) / 2;
                this.ctx.fillStyle = '#333';
                this.ctx.font = `bold ${baseFontSize}px sans-serif`;
                this.ctx.fillText(weight.toFixed(2), midX, midY);
            }
        });
        
        // Draw input nodes
        this.inputs.forEach((input, i) => {
            this.drawNode(input, '#1976d2', input.value);
            
            // Label
            this.ctx.fillStyle = '#333';
            this.ctx.font = `${baseFontSize}px sans-serif`;
            this.ctx.textAlign = 'right';
            this.ctx.fillText(`x${i}`, input.x - input.radius - 10, input.y + 5);
        });
        
        // Draw output node
        const outputColor = this.output.value === 1 ? '#43a047' : '#e53935';
        this.drawNode(this.output, outputColor, this.output.value);
        
        // Show target vs predicted
        this.ctx.fillStyle = '#333';
        this.ctx.font = `bold ${labelFontSize}px sans-serif`;
        this.ctx.textAlign = 'left';
        const labelX = Math.min(this.output.x + this.output.radius + 20, this.canvas.width - 100);
        this.ctx.fillText(`Predicted: ${this.output.value}`, labelX, this.output.y - 10);
        this.ctx.fillText(`Target: ${this.output.target}`, labelX, this.output.y + 15);
        
        // Show epoch
        if (state.perceptron.running) {
            this.ctx.fillStyle = '#667eea';
            this.ctx.font = `bold ${epochFontSize}px sans-serif`;
            this.ctx.textAlign = 'center';
            this.ctx.fillText(`Epoch: ${state.perceptron.epoch}`, this.canvas.width / 2, 40);
        }
    }
    
    drawNode(node, color, value) {
        // Glow
        const glowRadius = node.radius + Math.min(8, node.radius * 0.3);
        const gradient = this.ctx.createRadialGradient(
            node.x, node.y, node.radius,
            node.x, node.y, glowRadius
        );
        gradient.addColorStop(0, `${color}40`);
        gradient.addColorStop(1, `${color}00`);
        
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Node circle
        this.ctx.fillStyle = color;
        this.ctx.globalAlpha = 0.3 + value * 0.7;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Border
        this.ctx.globalAlpha = 1;
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = Math.max(2, node.radius / 10);
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        this.ctx.stroke();
        
        // Value
        this.ctx.fillStyle = 'white';
        const valueFontSize = Math.max(10, Math.min(14, node.radius * 0.8));
        this.ctx.font = `bold ${valueFontSize}px sans-serif`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(value.toFixed(2), node.x, node.y);
    }
}

function initPerceptron() {
    const canvas = document.getElementById('perceptron-canvas');
    if (!canvas) return;
    
    if (!perceptronViz) {
        perceptronViz = new PerceptronVisualizer(canvas);
    } else {
        perceptronViz.resize();
    }
    
    if (!perceptronInitialized) {
        perceptronInitialized = true;
        
        document.getElementById('perceptron-start').addEventListener('click', () => {
            // Ensure canvas is visible while animation runs (desktop only)
            scrollIntoViewIfDesktop('perceptron-canvas');
            state.perceptron.running = !state.perceptron.running;
            document.getElementById('perceptron-start').textContent = 
                state.perceptron.running ? 'Pause Training' : 'Start Training';
            if (state.perceptron.running) animatePerceptron();
        });
        
        document.getElementById('perceptron-reset').addEventListener('click', () => {
            resetPerceptron();
        });
        
        document.getElementById('perceptron-speed').addEventListener('input', (e) => {
            state.perceptron.speed = parseInt(e.target.value);
        });
    }
    
    perceptronViz.draw();
}

function animatePerceptron() {
    if (!state.perceptron.running || state.currentDemo !== 'perceptron') return;
    
    perceptronViz.update();
    perceptronViz.draw();
    
    setTimeout(() => {
        requestAnimationFrame(animatePerceptron);
    }, 1000 / state.perceptron.speed);
}

// ====== RBM Visualization ======
let rbmViz = null;
let rbmInitialized = false;
class RBMVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.visibleNodes = [];
        this.hiddenNodes = [];
        this.connections = [];
        this.particles = [];
        
        this.resize();
        window.addEventListener('resize', debounce(() => this.resize(), 250));
    }
    
    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.setupNodes();
    }
    
    setupNodes() {
        const visibleCount = 8;
        const hiddenCount = 5;
        // Responsive node radius and padding
        const nodeRadius = Math.max(12, Math.min(25, this.canvas.width / 30));
        const padding = Math.max(50, Math.min(100, this.canvas.width / 8));
        
        this.visibleNodes = [];
        this.hiddenNodes = [];
        this.connections = [];
        
        // Visible layer (bottom)
        const visibleY = this.canvas.height - padding;
        const visibleSpacing = (this.canvas.width - 2 * padding) / (visibleCount - 1);
        
        for (let i = 0; i < visibleCount; i++) {
            this.visibleNodes.push({
                x: padding + i * visibleSpacing,
                y: visibleY,
                radius: nodeRadius,
                activation: Math.random(),
                type: 'visible'
            });
        }
        
        // Hidden layer (top)
        const hiddenY = padding;
        const hiddenSpacing = (this.canvas.width - 2 * padding) / (hiddenCount - 1);
        
        for (let i = 0; i < hiddenCount; i++) {
            this.hiddenNodes.push({
                x: padding + i * hiddenSpacing,
                y: hiddenY,
                radius: nodeRadius,
                activation: Math.random(),
                type: 'hidden'
            });
        }
        
        // Create connections
        for (let v of this.visibleNodes) {
            for (let h of this.hiddenNodes) {
                this.connections.push({
                    from: v,
                    to: h,
                    weight: Math.random() * 2 - 1,
                    active: false
                });
            }
        }
    }
    
    update() {
        if (!state.rbm.running) return;
        
        const speed = state.rbm.speed / 500; // Reduced from 100 to 500 for slower animation
        state.rbm.animationProgress += speed;
        
        // Cycle through phases
        if (state.rbm.animationProgress >= 1) {
            state.rbm.animationProgress = 0;
            if (state.rbm.phase === 'forward') {
                state.rbm.phase = 'backward';
            } else {
                state.rbm.phase = 'forward';
                // Update activations
                this.visibleNodes.forEach(node => {
                    node.activation = Math.random();
                });
            }
        }
        
        // Update hidden nodes based on visible nodes (forward pass)
        if (state.rbm.phase === 'forward') {
            this.hiddenNodes.forEach((hNode, hIdx) => {
                let sum = 0;
                this.visibleNodes.forEach((vNode, vIdx) => {
                    const connIdx = vIdx * this.hiddenNodes.length + hIdx;
                    sum += vNode.activation * this.connections[connIdx].weight;
                });
                hNode.activation = 1 / (1 + Math.exp(-sum)); // sigmoid
            });
        }
        
        // Update visible nodes based on hidden nodes (backward pass)
        if (state.rbm.phase === 'backward') {
            this.visibleNodes.forEach((vNode, vIdx) => {
                let sum = 0;
                this.hiddenNodes.forEach((hNode, hIdx) => {
                    const connIdx = vIdx * this.hiddenNodes.length + hIdx;
                    sum += hNode.activation * this.connections[connIdx].weight;
                });
                vNode.activation = 1 / (1 + Math.exp(-sum));
            });
        }
        
        // Create particles
        if (Math.random() < 0.3) {
            const sourceNodes = state.rbm.phase === 'forward' ? this.visibleNodes : this.hiddenNodes;
            const targetNodes = state.rbm.phase === 'forward' ? this.hiddenNodes : this.visibleNodes;
            
            const source = sourceNodes[Math.floor(Math.random() * sourceNodes.length)];
            const target = targetNodes[Math.floor(Math.random() * targetNodes.length)];
            
            this.particles.push({
                x: source.x,
                y: source.y,
                targetX: target.x,
                targetY: target.y,
                progress: 0,
                life: 1
            });
        }
        
        // Update particles - slower movement for better visibility
        this.particles = this.particles.filter(p => {
            p.progress += speed * 0.8; // Reduced from 2 to 0.8
            p.life -= speed * 0.2; // Reduced from 0.5 to 0.2
            return p.life > 0 && p.progress < 1;
        });
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Responsive particle size
        const particleSize = Math.max(5, Math.min(8, this.canvas.width / 80));
        
        // Draw connections
        this.connections.forEach(conn => {
            const alpha = Math.abs(conn.weight) * 0.3;
            this.ctx.strokeStyle = conn.weight > 0 
                ? `rgba(102, 126, 234, ${alpha})` 
                : `rgba(234, 102, 102, ${alpha})`;
            this.ctx.lineWidth = Math.abs(conn.weight) * 2;
            this.ctx.beginPath();
            this.ctx.moveTo(conn.from.x, conn.from.y);
            this.ctx.lineTo(conn.to.x, conn.to.y);
            this.ctx.stroke();
        });
        
        // Draw particles
        this.particles.forEach(p => {
            const x = p.x + (p.targetX - p.x) * p.progress;
            const y = p.y + (p.targetY - p.y) * p.progress;
            
            const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, particleSize);
            gradient.addColorStop(0, `rgba(255, 215, 0, ${p.life})`);
            gradient.addColorStop(1, `rgba(255, 215, 0, 0)`);
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, particleSize, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        // Draw visible nodes
        this.visibleNodes.forEach(node => {
            this.drawNode(node, '#1976d2');
        });
        
        // Draw hidden nodes
        this.hiddenNodes.forEach(node => {
            this.drawNode(node, '#7b1fa2');
        });
        
        // Draw phase indicator
        if (state.rbm.running) {
            const text = state.rbm.phase === 'forward' ? '→ Forward Pass' : '← Backward Pass';
            const fontSize = Math.max(14, Math.min(18, this.canvas.width / 35));
            this.ctx.fillStyle = '#333';
            this.ctx.font = `bold ${fontSize}px sans-serif`;
            this.ctx.textAlign = 'center';
            this.ctx.fillText(text, this.canvas.width / 2, 30);
        }
    }
    
    drawNode(node, color) {
        // Outer glow based on activation
        const glowRadius = node.radius + Math.min(10, node.radius * 0.4);
        const gradient = this.ctx.createRadialGradient(
            node.x, node.y, node.radius,
            node.x, node.y, glowRadius
        );
        gradient.addColorStop(0, `${color}40`);
        gradient.addColorStop(1, `${color}00`);
        
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Node circle
        this.ctx.fillStyle = color;
        this.ctx.globalAlpha = 0.2 + node.activation * 0.8;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Node border
        this.ctx.globalAlpha = 1;
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = Math.max(2, node.radius / 10);
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        this.ctx.stroke();
        
        // Activation value
        this.ctx.fillStyle = 'white';
        const fontSize = Math.max(9, Math.min(12, node.radius * 0.7));
        this.ctx.font = `bold ${fontSize}px sans-serif`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(node.activation.toFixed(2), node.x, node.y);
    }
}

function initRBM() {
    const canvas = document.getElementById('rbm-canvas');
    if (!canvas) return;
    
    if (!rbmViz) {
        rbmViz = new RBMVisualizer(canvas);
    } else {
        rbmViz.resize();
    }
    
    if (!rbmInitialized) {
        rbmInitialized = true;
        
        document.getElementById('rbm-start').addEventListener('click', () => {
            // Ensure canvas is visible while animation runs (desktop only)
            scrollIntoViewIfDesktop('rbm-canvas');
            state.rbm.running = !state.rbm.running;
            const btn = document.getElementById('rbm-start');
            
            if (state.rbm.running) {
                btn.textContent = 'Pause Training';
                state.rbm.phase = 'forward';
                state.rbm.animationProgress = 0;
                animateRBM();
            } else {
                btn.textContent = 'Start Training';
            }
        });
        
        document.getElementById('rbm-reset').addEventListener('click', () => {
            resetRBM();
        });
        
        document.getElementById('rbm-speed').addEventListener('input', (e) => {
            state.rbm.speed = parseInt(e.target.value);
        });
    }
    
    rbmViz.draw();
}

function animateRBM() {
    if (!state.rbm.running || state.currentDemo !== 'rbm') return;
    
    rbmViz.update();
    rbmViz.draw();
    requestAnimationFrame(animateRBM);
}

// ====== Autoencoder Visualization ======
let aencoderViz = null;
let aencoderInitialized = false;
class AutoencoderVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.layers = [];
        this.particles = [];
        
        this.resize();
        window.addEventListener('resize', debounce(() => this.resize(), 250));
    }
    
    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.setupLayers();
    }
    
    setupLayers() {
        const layerSizes = [8, 5, 3, 5, 8];
        const layerColors = ['#388e3c', '#f57f17', '#e65100', '#f57f17', '#388e3c'];
        // Responsive node radius and padding
        const nodeRadius = Math.max(10, Math.min(20, this.canvas.width / 40));
        const padding = Math.max(40, Math.min(80, this.canvas.width / 10));
        
        this.layers = [];
        const layerSpacing = (this.canvas.width - 2 * padding) / (layerSizes.length - 1);
        
        layerSizes.forEach((size, layerIdx) => {
            const nodes = [];
            const x = padding + layerIdx * layerSpacing;
            const totalHeight = size * nodeRadius * 3;
            const startY = (this.canvas.height - totalHeight) / 2;
            
            for (let i = 0; i < size; i++) {
                nodes.push({
                    x: x,
                    y: startY + i * nodeRadius * 3 + nodeRadius * 1.5,
                    radius: nodeRadius,
                    activation: 0,
                    color: layerColors[layerIdx]
                });
            }
            
            this.layers.push(nodes);
        });
    }
    
    update() {
        if (!state.autoencoder.running) return;
        
        const speed = state.autoencoder.speed / 500; // Reduced from 100 to 500 for slower animation
        state.autoencoder.animationProgress += speed;
        
        // Cycle through phases
        if (state.autoencoder.animationProgress >= 1) {
            state.autoencoder.animationProgress = 0;
            if (state.autoencoder.phase === 'encoding') {
                state.autoencoder.phase = 'decoding';
            } else {
                state.autoencoder.phase = 'encoding';
                // Reset input
                this.layers[0].forEach(node => {
                    node.activation = Math.random();
                });
            }
        }
        
        // Forward propagation
        const currentLayer = state.autoencoder.phase === 'encoding' 
            ? Math.floor(state.autoencoder.animationProgress * 2.5)
            : 2 + Math.floor(state.autoencoder.animationProgress * 2.5);
        
        for (let i = 0; i <= Math.min(currentLayer, this.layers.length - 2); i++) {
            const layer = this.layers[i];
            const nextLayer = this.layers[i + 1];
            
            nextLayer.forEach(nextNode => {
                let sum = 0;
                layer.forEach(node => {
                    sum += node.activation * (Math.random() * 0.5 + 0.5);
                });
                nextNode.activation = Math.min(1, sum / layer.length);
            });
        }
        
        // Create particles
        if (Math.random() < 0.2) {
            const layerIdx = Math.min(currentLayer, this.layers.length - 2);
            if (layerIdx >= 0) {
                const fromLayer = this.layers[layerIdx];
                const toLayer = this.layers[layerIdx + 1];
                
                const fromNode = fromLayer[Math.floor(Math.random() * fromLayer.length)];
                const toNode = toLayer[Math.floor(Math.random() * toLayer.length)];
                
                this.particles.push({
                    x: fromNode.x,
                    y: fromNode.y,
                    targetX: toNode.x,
                    targetY: toNode.y,
                    progress: 0,
                    life: 1
                });
            }
        }
        
        // Update particles - slower movement for better visibility
        this.particles = this.particles.filter(p => {
            p.progress += speed * 1.2; // Reduced from 3 to 1.2
            p.life -= speed * 0.2; // Reduced from 0.5 to 0.2
            return p.life > 0 && p.progress < 1;
        });
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw connections
        for (let i = 0; i < this.layers.length - 1; i++) {
            const layer = this.layers[i];
            const nextLayer = this.layers[i + 1];
            
            layer.forEach(node1 => {
                nextLayer.forEach(node2 => {
                    this.ctx.strokeStyle = 'rgba(200, 200, 200, 0.2)';
                    this.ctx.lineWidth = 1;
                    this.ctx.beginPath();
                    this.ctx.moveTo(node1.x, node1.y);
                    this.ctx.lineTo(node2.x, node2.y);
                    this.ctx.stroke();
                });
            });
        }
        
        // Draw particles
        this.particles.forEach(p => {
            const x = p.x + (p.targetX - p.x) * p.progress;
            const y = p.y + (p.targetY - p.y) * p.progress;
            
            const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, 6);
            gradient.addColorStop(0, `rgba(255, 165, 0, ${p.life})`);
            gradient.addColorStop(1, `rgba(255, 165, 0, 0)`);
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 6, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        // Draw nodes
        this.layers.forEach(layer => {
            layer.forEach(node => {
                this.drawNode(node);
            });
        });
        
        // Draw phase indicator
        if (state.autoencoder.running) {
            const text = state.autoencoder.phase === 'encoding' 
                ? 'Encoding → Compression' 
                : 'Decoding → Reconstruction';
            this.ctx.fillStyle = '#333';
            this.ctx.font = 'bold 18px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(text, this.canvas.width / 2, 30);
        }
    }
    
    drawNode(node) {
        // Outer glow
        const glowRadius = node.radius + 8;
        const gradient = this.ctx.createRadialGradient(
            node.x, node.y, node.radius,
            node.x, node.y, glowRadius
        );
        gradient.addColorStop(0, `${node.color}30`);
        gradient.addColorStop(1, `${node.color}00`);
        
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Node circle
        this.ctx.fillStyle = node.color;
        this.ctx.globalAlpha = 0.2 + node.activation * 0.8;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Node border
        this.ctx.globalAlpha = 1;
        this.ctx.strokeStyle = node.color;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        this.ctx.stroke();
    }
}

function initAutoencoder() {
    const canvas = document.getElementById('ae-canvas');
    if (!canvas) return;
    
    if (!aencoderViz) {
        aencoderViz = new AutoencoderVisualizer(canvas);
    } else {
        aencoderViz.resize();
    }
    
    if (!aencoderInitialized) {
        aencoderInitialized = true;
        
        document.getElementById('ae-start').addEventListener('click', () => {
            // Ensure canvas is visible while animation runs (desktop only)
            scrollIntoViewIfDesktop('ae-canvas');
            state.autoencoder.running = !state.autoencoder.running;
            const btn = document.getElementById('ae-start');
            
            if (state.autoencoder.running) {
                btn.textContent = 'Pause Encoding';
                state.autoencoder.phase = 'encoding';
                state.autoencoder.animationProgress = 0;
                animateAutoencoder();
            } else {
                btn.textContent = 'Start Encoding';
            }
        });
        
        document.getElementById('ae-reset').addEventListener('click', () => {
            resetAutoencoder();
        });
        
        document.getElementById('ae-speed').addEventListener('input', (e) => {
            state.autoencoder.speed = parseInt(e.target.value);
        });
    }
    
    aencoderViz.draw();
}

function animateAutoencoder() {
    if (!state.autoencoder.running || state.currentDemo !== 'autoencoder') return;
    
    aencoderViz.update();
    aencoderViz.draw();
    requestAnimationFrame(animateAutoencoder);
}

// ====== Ising Model Visualization ======
let isingViz = null;
let isingInitialized = false;

class IsingVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.gridSize = 50;
        this.spins = [];
        this.temperature = 2.27; // Critical temperature
        this.energy = 0;
        
        this.resize();
        window.addEventListener('resize', debounce(() => this.resize(), 250));
    }
    
    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.cellSize = Math.min(this.canvas.width, this.canvas.height) / this.gridSize;
        this.initializeSpins();
    }
    
    initializeSpins() {
        this.spins = [];
        for (let i = 0; i < this.gridSize; i++) {
            this.spins[i] = [];
            for (let j = 0; j < this.gridSize; j++) {
                this.spins[i][j] = Math.random() < 0.5 ? 1 : -1;
            }
        }
        this.calculateEnergy();
    }
    
    calculateEnergy() {
        this.energy = 0;
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                const spin = this.spins[i][j];
                const right = this.spins[(i + 1) % this.gridSize][j];
                const down = this.spins[i][(j + 1) % this.gridSize];
                this.energy -= spin * (right + down);
            }
        }
    }
    
    update() {
        if (!state.ising.running) return;
        
        this.temperature = state.ising.temperature;
        
        // Monte Carlo update - multiple steps per frame
        const stepsPerFrame = state.ising.speed * 10;
        for (let step = 0; step < stepsPerFrame; step++) {
            const i = Math.floor(Math.random() * this.gridSize);
            const j = Math.floor(Math.random() * this.gridSize);
            
            // Calculate energy change if we flip this spin
            const spin = this.spins[i][j];
            const left = this.spins[(i - 1 + this.gridSize) % this.gridSize][j];
            const right = this.spins[(i + 1) % this.gridSize][j];
            const up = this.spins[i][(j - 1 + this.gridSize) % this.gridSize];
            const down = this.spins[i][(j + 1) % this.gridSize];
            
            const neighbors = left + right + up + down;
            const deltaE = 2 * spin * neighbors;
            
            // Metropolis algorithm
            if (deltaE < 0 || Math.random() < Math.exp(-deltaE / this.temperature)) {
                this.spins[i][j] *= -1;
                this.energy += deltaE;
            }
        }
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        const offsetX = (this.canvas.width - this.gridSize * this.cellSize) / 2;
        const offsetY = (this.canvas.height - this.gridSize * this.cellSize) / 2;
        
        // Draw spins
        for (let i = 0; i < this.gridSize; i++) {
            for (let j = 0; j < this.gridSize; j++) {
                const x = offsetX + i * this.cellSize;
                const y = offsetY + j * this.cellSize;
                
                this.ctx.fillStyle = this.spins[i][j] === 1 ? '#ffffff' : '#000000';
                this.ctx.fillRect(x, y, this.cellSize, this.cellSize);
            }
        }
        
        // Update display
        const tempDisplay = document.getElementById('ising-temp-display');
        const energyDisplay = document.getElementById('ising-energy-display');
        if (tempDisplay) tempDisplay.textContent = this.temperature.toFixed(2);
        if (energyDisplay) energyDisplay.textContent = this.energy.toFixed(0);
    }
}

function initIsing() {
    const canvas = document.getElementById('ising-canvas');
    if (!canvas) return;
    
    if (!isingViz) {
        isingViz = new IsingVisualizer(canvas);
    } else {
        isingViz.resize();
    }
    
    if (!isingInitialized) {
        isingInitialized = true;
        
        document.getElementById('ising-start').addEventListener('click', () => {
            // Ensure canvas is visible while animation runs (desktop only)
            scrollIntoViewIfDesktop('ising-canvas');
            state.ising.running = !state.ising.running;
            document.getElementById('ising-start').textContent = 
                state.ising.running ? 'Pause Simulation' : 'Start Simulation';
            if (state.ising.running) animateIsing();
        });
        
        document.getElementById('ising-reset').addEventListener('click', () => {
            resetIsing();
        });
        
        document.getElementById('ising-temp').addEventListener('input', (e) => {
            state.ising.temperature = parseFloat(e.target.value);
        });
        
        document.getElementById('ising-speed').addEventListener('input', (e) => {
            state.ising.speed = parseInt(e.target.value);
        });
    }
    
    isingViz.draw();
}

function animateIsing() {
    if (!state.ising.running || state.currentDemo !== 'ising') return;
    
    isingViz.update();
    isingViz.draw();
    
    setTimeout(() => {
        requestAnimationFrame(animateIsing);
    }, 50);
}

// ====== Hopfield Network Visualization ======
let hopfieldViz = null;
let hopfieldInitialized = false;

class HopfieldVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.size = 15; // 15x15 grid
        this.state = [];
        this.weights = [];
        this.patterns = [];
        this.isRecalling = false;
        
        this.resize();
        window.addEventListener('resize', debounce(() => this.resize(), 250));
        this.setupMouseInteraction();
    }
    
    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        // Make cell size more responsive, with minimum size for mobile
        const maxSize = Math.min(this.canvas.width, this.canvas.height);
        this.cellSize = Math.max(15, maxSize / this.size);
        this.initializeState();
    }
    
    initializeState() {
        this.state = [];
        for (let i = 0; i < this.size; i++) {
            this.state[i] = [];
            for (let j = 0; j < this.size; j++) {
                this.state[i][j] = Math.random() < 0.5 ? 1 : -1;
            }
        }
    }
    
    setupMouseInteraction() {
        this.canvas.addEventListener('click', (e) => {
            if (this.isRecalling) return;
            
            const rect = this.canvas.getBoundingClientRect();
            const offsetX = (this.canvas.width - this.size * this.cellSize) / 2;
            const offsetY = (this.canvas.height - this.size * this.cellSize) / 2;
            
            const x = e.clientX - rect.left - offsetX;
            const y = e.clientY - rect.top - offsetY;
            
            const i = Math.floor(x / this.cellSize);
            const j = Math.floor(y / this.cellSize);
            
            if (i >= 0 && i < this.size && j >= 0 && j < this.size) {
                this.state[i][j] *= -1;
                this.draw();
            }
        });
    }
    
    storePatterns() {
        // Create some predefined patterns
        this.patterns = [];
        
        // Pattern 1: X shape
        const pattern1 = [];
        for (let i = 0; i < this.size; i++) {
            pattern1[i] = [];
            for (let j = 0; j < this.size; j++) {
                pattern1[i][j] = (i === j || i === this.size - 1 - j) ? 1 : -1;
            }
        }
        this.patterns.push(pattern1);
        
        // Pattern 2: Plus shape
        const pattern2 = [];
        const mid = Math.floor(this.size / 2);
        for (let i = 0; i < this.size; i++) {
            pattern2[i] = [];
            for (let j = 0; j < this.size; j++) {
                pattern2[i][j] = (i === mid || j === mid) ? 1 : -1;
            }
        }
        this.patterns.push(pattern2);
        
        // Pattern 3: Checkboard
        const pattern3 = [];
        for (let i = 0; i < this.size; i++) {
            pattern3[i] = [];
            for (let j = 0; j < this.size; j++) {
                pattern3[i][j] = ((i + j) % 2 === 0) ? 1 : -1;
            }
        }
        this.patterns.push(pattern3);
        
        // Train weights using Hebbian learning
        const n = this.size * this.size;
        this.weights = Array(n).fill(0).map(() => Array(n).fill(0));
        
        for (const pattern of this.patterns) {
            const flat = pattern.flat();
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    if (i !== j) {
                        this.weights[i][j] += flat[i] * flat[j];
                    }
                }
            }
        }
        
        document.getElementById('hopfield-patterns').textContent = this.patterns.length;
    }
    
    recall() {
        if (this.patterns.length === 0) return;
        
        this.isRecalling = true;
        let maxIterations = 20;
        let iteration = 0;
        
        const recallStep = () => {
            if (iteration >= maxIterations) {
                this.isRecalling = false;
                return;
            }
            
            // Asynchronous update
            const i = Math.floor(Math.random() * this.size);
            const j = Math.floor(Math.random() * this.size);
            const idx = i * this.size + j;
            
            let sum = 0;
            for (let k = 0; k < this.size * this.size; k++) {
                const ki = Math.floor(k / this.size);
                const kj = k % this.size;
                sum += this.weights[idx][k] * this.state[ki][kj];
            }
            
            this.state[i][j] = sum >= 0 ? 1 : -1;
            this.draw();
            
            iteration++;
            setTimeout(recallStep, 50);
        };
        
        recallStep();
    }
    
    addNoise(noiseLevel) {
        const numFlips = Math.floor((this.size * this.size * noiseLevel) / 100);
        for (let f = 0; f < numFlips; f++) {
            const i = Math.floor(Math.random() * this.size);
            const j = Math.floor(Math.random() * this.size);
            this.state[i][j] *= -1;
        }
        this.draw();
    }
    
    clearCanvas() {
        // Clear the canvas to all white (all -1 values)
        for (let i = 0; i < this.size; i++) {
            for (let j = 0; j < this.size; j++) {
                this.state[i][j] = -1;
            }
        }
        this.isRecalling = false;
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        const offsetX = (this.canvas.width - this.size * this.cellSize) / 2;
        const offsetY = (this.canvas.height - this.size * this.cellSize) / 2;
        
        for (let i = 0; i < this.size; i++) {
            for (let j = 0; j < this.size; j++) {
                const x = offsetX + i * this.cellSize;
                const y = offsetY + j * this.cellSize;
                
                this.ctx.fillStyle = this.state[i][j] === 1 ? '#667eea' : '#ffffff';
                this.ctx.fillRect(x, y, this.cellSize - 1, this.cellSize - 1);
                
                this.ctx.strokeStyle = '#ddd';
                this.ctx.strokeRect(x, y, this.cellSize - 1, this.cellSize - 1);
            }
        }
        
        // Instructions
        if (!this.isRecalling) {
            this.ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
            this.ctx.fillRect(10, 10, this.canvas.width - 20, 60);
            this.ctx.fillStyle = '#333';
            this.ctx.font = '14px sans-serif';
            this.ctx.textAlign = 'left';
            this.ctx.fillText('Click cells to draw a pattern', 20, 30);
            this.ctx.fillText('Then test recall with added noise', 20, 50);
        }
    }
}

function initHopfield() {
    const canvas = document.getElementById('hopfield-canvas');
    if (!canvas) return;
    
    if (!hopfieldViz) {
        hopfieldViz = new HopfieldVisualizer(canvas);
    } else {
        hopfieldViz.resize();
    }
    
    if (!hopfieldInitialized) {
        hopfieldInitialized = true;
        
        document.getElementById('hopfield-store').addEventListener('click', () => {
            // Keep canvas in view while interacting (desktop only)
            scrollIntoViewIfDesktop('hopfield-canvas');
            hopfieldViz.storePatterns();
            hopfieldViz.draw();
        });
        
        document.getElementById('hopfield-recall').addEventListener('click', () => {
            // Keep canvas in view during recall animation (desktop only)
            scrollIntoViewIfDesktop('hopfield-canvas');
            const noiseLevel = parseInt(document.getElementById('hopfield-noise').value);
            hopfieldViz.addNoise(noiseLevel);
            setTimeout(() => hopfieldViz.recall(), 500);
        });
        
        document.getElementById('hopfield-clear').addEventListener('click', () => {
            hopfieldViz.clearCanvas();
            hopfieldViz.draw();
        });
        
        document.getElementById('hopfield-reset').addEventListener('click', () => {
            resetHopfield();
        });
    }
    
    hopfieldViz.draw();
}

// ====== Transformer Visualization ======
let transformerViz = null;
let transformerInitialized = false;
let toyLMBuilt = false;

// Simple toy language model (unigram + bigram) for next-token prediction
const toyLM = {
    unigram: new Map(),
    bigram: new Map(),
    vocab: new Set(),
    total: 0,
    sos: '<s>',
    eos: '</s>'
};

function tokenize(text) {
    if (!text) return [];
    const matches = text.toLowerCase().match(/[\w']+|[.,!?;:]/g);
    return matches ? matches : [];
}

function buildToyLanguageModel() {
    if (toyLMBuilt) return;
    const corpus = [
        'the cat sat on the mat',
        'the dog sat on the rug',
        'the cat chased the mouse',
        'the dog chased the cat',
        'a cat likes fish',
        'a dog likes bones',
        'the mat was soft and comfortable',
        'the rug was red and warm',
        'the cat slept peacefully',
        'the dog barked loudly',
        'cats and dogs are animals',
        'attention helps pick the next token',
        'transformers predict the next token',
        'machine learning is fascinating',
        'neural networks learn patterns',
        'the model generates text',
        'artificial intelligence is powerful',
        'deep learning requires data',
        'the algorithm processes information',
        'language models understand context',
        'the cat is sleeping now',
        'the dog is running fast',
        'machine learning models are trained',
        'neural networks use backpropagation',
        'the transformer uses attention',
        'attention mechanisms are important',
        'i love machine learning',
        'i think neural networks',
        'i like transformers',
        'what is machine learning',
        'what are neural networks',
        'how does attention work',
        'why use transformers',
        'this is amazing',
        'this model works',
        'the sun is shining',
        'the moon is bright',
        'the sky is blue',
        'the grass is green'
    ];

    for (const line of corpus) {
        const tokens = [toyLM.sos, ...tokenize(line), toyLM.eos];
        for (let i = 0; i < tokens.length; i++) {
            const t = tokens[i];
            toyLM.vocab.add(t);
            toyLM.unigram.set(t, (toyLM.unigram.get(t) || 0) + 1);
            toyLM.total++;
            if (i > 0) {
                const prev = tokens[i - 1];
                if (!toyLM.bigram.has(prev)) toyLM.bigram.set(prev, new Map());
                const row = toyLM.bigram.get(prev);
                row.set(t, (row.get(t) || 0) + 1);
            }
        }
    }
    toyLMBuilt = true;
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - maxLogit));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
}

function predictNextTokens(prompt, temperature = 1.0, topK = 5) {
    buildToyLanguageModel();
    const tokens = tokenize(prompt);
    const last = tokens.length ? tokens[tokens.length - 1] : toyLM.sos;

    let dist = new Map();
    if (toyLM.bigram.has(last)) {
        // Use bigram distribution if we have seen this context
        dist = new Map(toyLM.bigram.get(last));
    } else {
        // Fallback to unigram distribution (excluding sos)
        toyLM.unigram.forEach((c, t) => {
            if (t !== toyLM.sos) dist.set(t, c);
        });
    }

    // Convert counts to logits and apply temperature (log-count smoothing)
    const entries = Array.from(dist.entries());
    const logits = entries.map(([t, c]) => Math.log(1 + c) / Math.max(temperature, 0.05));
    const probs = softmax(logits);
    const scored = entries.map(([t], i) => ({ token: t, prob: probs[i] }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, Math.max(1, Math.min(10, topK)));
    return scored;
}

function updateTransformerResults(preds) {
    const list = document.getElementById('transformer-topk-list');
    if (!list) return;
    
    // Fade out animation
    list.style.opacity = '0';
    list.style.transform = 'translateY(10px)';
    
    setTimeout(() => {
        list.innerHTML = '';
        
        if (preds.length === 0) {
            const emptyMsg = document.createElement('li');
            emptyMsg.style.textAlign = 'center';
            emptyMsg.style.color = '#999';
            emptyMsg.style.padding = '20px';
            emptyMsg.style.fontStyle = 'italic';
            emptyMsg.textContent = 'Type something and click "Predict next token" to see predictions!';
            list.appendChild(emptyMsg);
        } else {
            preds.forEach(({ token, prob }, idx) => {
                const li = document.createElement('li');
                li.className = 'token-item';
                li.style.opacity = '0';
                li.style.transform = 'translateX(-20px)';
                
                const label = document.createElement('div');
                label.className = 'token-label';
                label.textContent = token;
                
                const bar = document.createElement('div');
                bar.className = 'token-bar';
                
                const fill = document.createElement('div');
                fill.className = 'token-bar-fill';
                fill.style.width = `${(prob * 100).toFixed(1)}%`;
                bar.appendChild(fill);
                
                const pct = document.createElement('div');
                pct.className = 'token-prob';
                pct.textContent = `${(prob * 100).toFixed(1)}%`;
                
                li.appendChild(label);
                li.appendChild(bar);
                li.appendChild(pct);
                
                if (idx === 0) {
                    li.style.fontWeight = '700';
                    li.style.background = 'linear-gradient(135deg, #f8f9ff 0%, #eff1ff 100%)';
                }
                
                list.appendChild(li);
                
                // Stagger the animation
                setTimeout(() => {
                    li.style.transition = 'all 0.4s ease';
                    li.style.opacity = '1';
                    li.style.transform = 'translateX(0)';
                }, idx * 50);
            });
        }
        
        // Fade in
        list.style.transition = 'all 0.3s ease';
        list.style.opacity = '1';
        list.style.transform = 'translateY(0)';
    }, 150);
}

class TransformerVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.tokens = ['the', 'cat', 'sat', 'on', 'mat'];
        this.layers = [];
        this.attentionWeights = [];
        this.particles = [];
        this.phase = 0; // 0: embedding, 1: attention, 2: context, 3: feed-forward, 4: output
        this.phaseProgress = 0;
        this.animationTime = 0;
        this.attentionFocus = 0; // Which token is currently being attended to
        this.predictions = [];
        this.predictedTop = null;
        
        this.resize();
        this.setupLayers();
        window.addEventListener('resize', debounce(() => this.resize(), 250));
    }
    
    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.setupLayers();
    }
    
    setupLayers() {
        const layerCount = 5; // Prompt, Context, Scores, Softmax, Next Token
        const tokenCount = this.tokens.length || 1;
        const layerColors = ['#4A90E2', '#f57f17', '#9B59B6', '#E74C3C', '#27AE60'];
        const layerNames = ['Prompt', 'Context', 'Scores', 'Softmax', 'Next'];
        
        // Responsive node radius and padding
        const nodeRadius = Math.max(10, Math.min(15, this.canvas.width / 50));
        const padding = Math.max(60, Math.min(100, this.canvas.width / 10));
        
        this.layers = [];
        const layerSpacing = (this.canvas.width - 2 * padding) / (layerCount - 1);
        
        for (let l = 0; l < layerCount; l++) {
            const nodes = [];
            const x = padding + l * layerSpacing;
            const verticalSpacing = (this.canvas.height - 120) / tokenCount;
            
            for (let t = 0; t < tokenCount; t++) {
                nodes.push({
                    x: x,
                    y: 60 + t * verticalSpacing + verticalSpacing / 2,
                    radius: nodeRadius,
                    activation: 0,
                    targetActivation: 0,
                    color: layerColors[l],
                    token: this.tokens[t],
                    layerName: layerNames[l]
                });
            }
            
            this.layers.push(nodes);
        }
        
        // Build simple context weights: last token attends to previous tokens with recency bias
        this.attentionWeights = [];
        for (let i = 0; i < tokenCount; i++) {
            this.attentionWeights[i] = [];
            for (let j = 0; j < tokenCount; j++) {
                const dist = Math.max(1, Math.abs(i - j));
                const base = i === j ? 0.6 : 1 / (dist + 0.5);
                this.attentionWeights[i][j] = base;
            }
        }
        // Normalize each row to [0,1]
        for (let i = 0; i < tokenCount; i++) {
            const row = this.attentionWeights[i];
            const max = Math.max(...row);
            if (max > 0) {
                for (let j = 0; j < tokenCount; j++) row[j] = row[j] / max;
            }
        }
    }

    setTokens(tokens) {
        this.tokens = (tokens && tokens.length ? tokens : ['<start>']).map(t => String(t));
        this.attentionFocus = Math.max(0, this.tokens.length - 1);
        this.setupLayers();
        this.draw();
    }

    setPredictions(preds) {
        this.predictions = preds || [];
        this.predictedTop = this.predictions.length ? this.predictions[0].token : null;
        
        // Trigger a brief animation when predictions change
        if (this.predictions.length > 0) {
            this.triggerPredictionAnimation();
        }
        this.draw();
    }
    
    triggerPredictionAnimation() {
        // Flash the output layer
        if (this.layers.length > 0) {
            const outputLayer = this.layers[this.layers.length - 1];
            outputLayer.forEach(node => {
                node.targetActivation = 1.0;
                setTimeout(() => {
                    node.targetActivation = 0.7;
                }, 200);
            });
        }
        
        // Create a burst of particles
        for (let i = 0; i < 15; i++) {
            setTimeout(() => {
                if (this.layers.length >= 2) {
                    const fromLayer = this.layers[this.layers.length - 2];
                    const toLayer = this.layers[this.layers.length - 1];
                    const idx = Math.floor(Math.random() * Math.min(fromLayer.length, toLayer.length));
                    if (fromLayer[idx] && toLayer[idx]) {
                        this.particles.push({
                            x: fromLayer[idx].x,
                            y: fromLayer[idx].y,
                            targetX: toLayer[idx].x,
                            targetY: toLayer[idx].y,
                            progress: 0,
                            life: 1,
                            color: '#27AE60',
                            isPrediction: true
                        });
                    }
                }
            }, i * 30);
        }
    }
    
    update() {
        if (!state.transformer.running || !this.layers || this.layers.length === 0) return;
        
        this.animationTime += 0.02 * state.transformer.speed;
        
        // Smooth phase progression
        this.phaseProgress += 0.008 * state.transformer.speed;
        
        if (this.phaseProgress >= 1.0) {
            this.phaseProgress = 0;
            this.phase = (this.phase + 1) % this.layers.length;
            
            // Move attention focus when in attention phase
            if (this.phase === 1) {
                this.attentionFocus = (this.attentionFocus + 1) % this.tokens.length;
            }
        }
        
        // Update layer activations smoothly
        this.layers.forEach((layer, layerIdx) => {
            layer.forEach((node, tokenIdx) => {
                // Calculate target activation based on current phase
                if (layerIdx < this.phase) {
                    node.targetActivation = 0.7;
                } else if (layerIdx === this.phase) {
                    // Smooth activation during current phase
                    if (layerIdx === 1 && this.attentionWeights[this.attentionFocus] && 
                        this.attentionWeights[this.attentionFocus][tokenIdx] !== undefined) { 
                        // Attention layer - show attention pattern
                        const weight = this.attentionWeights[this.attentionFocus][tokenIdx];
                        node.targetActivation = 0.3 + weight * 0.7;
                    } else {
                        node.targetActivation = 0.5 + Math.sin(this.animationTime + tokenIdx) * 0.3;
                    }
                } else {
                    node.targetActivation = 0.1;
                }
                
                // Smooth interpolation
                node.activation += (node.targetActivation - node.activation) * 0.15;
            });
        });
        
        // Create smooth particle flow
        if (Math.random() < 0.25 * state.transformer.speed) {
            const currentLayerIdx = this.phase;
            const nextLayerIdx = (this.phase + 1) % this.layers.length;
            
            if (this.layers[currentLayerIdx] && this.layers[nextLayerIdx]) {
                const tokenIdx = Math.floor(Math.random() * this.tokens.length);
                const fromNode = this.layers[currentLayerIdx][tokenIdx];
                const toNode = this.layers[nextLayerIdx][tokenIdx];
                
                if (fromNode && toNode) {
                    this.particles.push({
                        x: fromNode.x,
                        y: fromNode.y,
                        targetX: toNode.x,
                        targetY: toNode.y,
                        progress: 0,
                        life: 1,
                        color: fromNode.color
                    });
                }
            }
        }
        
        // During attention phase, create cross-attention particles
        if (this.phase === 1 && this.layers[1] && Math.random() < 0.15 * state.transformer.speed) {
            const attentionLayer = this.layers[1];
            const fromIdx = this.attentionFocus;
            const toIdx = Math.floor(Math.random() * this.tokens.length);
            
            if (fromIdx !== toIdx && attentionLayer[fromIdx] && attentionLayer[toIdx] &&
                this.attentionWeights[fromIdx] && this.attentionWeights[fromIdx][toIdx] > 0.3) {
                this.particles.push({
                    x: attentionLayer[fromIdx].x,
                    y: attentionLayer[fromIdx].y,
                    targetX: attentionLayer[toIdx].x,
                    targetY: attentionLayer[toIdx].y,
                    progress: 0,
                    life: 1,
                    color: '#f57f17',
                    isAttention: true
                });
            }
        }
        
        // Update particles with smooth motion
        this.particles = this.particles.filter(p => {
            p.progress += 0.03;
            p.life -= 0.015;
            return p.life > 0 && p.progress < 1;
        });
    }
    
    draw() {
        if (!this.ctx || !this.canvas) return;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw attention connections when in attention phase
        if (this.phase === 1 && this.layers.length > 1 && this.attentionWeights.length > 0) {
            const attentionLayer = this.layers[1];
            if (!attentionLayer || this.attentionFocus >= attentionLayer.length) return;
            
            const focusNode = attentionLayer[this.attentionFocus];
            if (!focusNode) return;
            
            for (let j = 0; j < attentionLayer.length; j++) {
                if (this.attentionFocus !== j && this.attentionWeights[this.attentionFocus]) {
                    const weight = this.attentionWeights[this.attentionFocus][j];
                    const alpha = weight * 0.6 * this.phaseProgress;
                    const targetNode = attentionLayer[j];
                    
                    if (!targetNode) continue;
                    
                    this.ctx.strokeStyle = `rgba(247, 127, 23, ${alpha})`;
                    this.ctx.lineWidth = 1 + weight * 3;
                    this.ctx.beginPath();
                    this.ctx.moveTo(focusNode.x, focusNode.y);
                    this.ctx.lineTo(targetNode.x, targetNode.y);
                    this.ctx.stroke();
                    
                    // Draw arrow head
                    const angle = Math.atan2(targetNode.y - focusNode.y, targetNode.x - focusNode.x);
                    const arrowSize = 8;
                    const arrowX = targetNode.x - Math.cos(angle) * (targetNode.radius + 5);
                    const arrowY = targetNode.y - Math.sin(angle) * (targetNode.radius + 5);
                    
                    this.ctx.fillStyle = `rgba(247, 127, 23, ${alpha})`;
                    this.ctx.beginPath();
                    this.ctx.moveTo(arrowX, arrowY);
                    this.ctx.lineTo(
                        arrowX - arrowSize * Math.cos(angle - Math.PI / 6),
                        arrowY - arrowSize * Math.sin(angle - Math.PI / 6)
                    );
                    this.ctx.lineTo(
                        arrowX - arrowSize * Math.cos(angle + Math.PI / 6),
                        arrowY - arrowSize * Math.sin(angle + Math.PI / 6)
                    );
                    this.ctx.closePath();
                    this.ctx.fill();
                }
            }
        }
        
        // Draw layer connections (faded)
        for (let i = 0; i < this.layers.length - 1; i++) {
            const layer = this.layers[i];
            const nextLayer = this.layers[i + 1];
            
            layer.forEach((node1, idx1) => {
                const node2 = nextLayer[idx1]; // Connect same token positions
                this.ctx.strokeStyle = 'rgba(200, 200, 200, 0.1)';
                this.ctx.lineWidth = 1;
                this.ctx.beginPath();
                this.ctx.moveTo(node1.x, node1.y);
                this.ctx.lineTo(node2.x, node2.y);
                this.ctx.stroke();
            });
        }
        
        // Draw particles
        this.particles.forEach(p => {
            const x = p.x + (p.targetX - p.x) * this.easeInOutCubic(p.progress);
            const y = p.y + (p.targetY - p.y) * this.easeInOutCubic(p.progress);
            
            // Prediction particles are larger and more prominent
            const particleSize = p.isPrediction ? 8 : (p.isAttention ? 4 : 5);
            const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, particleSize + 2);
            
            // Convert hex color to RGB
            let rgb = '102, 126, 234'; // default
            if (p.color && p.color.startsWith('#')) {
                const hex = p.color.slice(1);
                const r = parseInt(hex.slice(0, 2), 16);
                const g = parseInt(hex.slice(2, 4), 16);
                const b = parseInt(hex.slice(4, 6), 16);
                rgb = `${r}, ${g}, ${b}`;
            }
            
            const alpha = p.isPrediction ? Math.min(1, p.life * 1.5) : p.life;
            gradient.addColorStop(0, `rgba(${rgb}, ${alpha})`);
            gradient.addColorStop(1, `rgba(${rgb}, 0)`);
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, particleSize, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Add glow effect for prediction particles
            if (p.isPrediction) {
                this.ctx.strokeStyle = `rgba(${rgb}, ${alpha * 0.3})`;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.arc(x, y, particleSize + 2, 0, Math.PI * 2);
                this.ctx.stroke();
            }
        });
        
        // Draw nodes with token labels
        this.layers.forEach((layer, layerIdx) => {
            layer.forEach((node, tokenIdx) => {
                this.drawNode(node, layerIdx === 1 && tokenIdx === this.attentionFocus && this.phase === 1);
                
                // Show token label on first layer
                if (layerIdx === 0) {
                    this.ctx.fillStyle = '#333';
                    this.ctx.font = 'bold 13px sans-serif';
                    this.ctx.textAlign = 'right';
                    this.ctx.fillText(node.token, node.x - node.radius - 12, node.y + 4);
                }
            });
        });
        
        // Draw phase indicator
        this.ctx.fillStyle = '#333';
        this.ctx.font = 'bold 14px sans-serif';
        this.ctx.textAlign = 'center';
        const phaseNames = ['Prompt', 'Context (self‑attention)', 'Scores', 'Softmax', 'Next token'];
        this.ctx.fillText(phaseNames[this.phase], this.canvas.width / 2, 25);

        // Show top-1 predicted token near the output column with enhanced styling
        if (this.predictedTop) {
            const outputLayer = this.layers[this.layers.length - 1];
            if (outputLayer && outputLayer.length) {
                const anchor = outputLayer[Math.min(outputLayer.length - 1, this.attentionFocus)];
                
                // Draw a rounded rectangle background
                const text = `${this.predictedTop}`;
                this.ctx.font = 'bold 16px monospace';
                const textWidth = this.ctx.measureText(text).width;
                const padding = 10;
                const boxX = anchor.x + anchor.radius + 18;
                const boxY = anchor.y - 12;
                const boxWidth = textWidth + padding * 2;
                const boxHeight = 28;
                
                // Background with shadow
                this.ctx.shadowColor = 'rgba(39, 174, 96, 0.3)';
                this.ctx.shadowBlur = 10;
                this.ctx.fillStyle = '#27AE60';
                this.ctx.beginPath();
                this.ctx.roundRect(boxX, boxY, boxWidth, boxHeight, 6);
                this.ctx.fill();
                this.ctx.shadowBlur = 0;
                
                // Text
                this.ctx.fillStyle = '#fff';
                this.ctx.textAlign = 'left';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText(text, boxX + padding, anchor.y + 1);
                
                // Arrow pointing to the box
                this.ctx.fillStyle = '#27AE60';
                this.ctx.beginPath();
                this.ctx.moveTo(anchor.x + anchor.radius + 12, anchor.y);
                this.ctx.lineTo(boxX - 6, anchor.y - 5);
                this.ctx.lineTo(boxX - 6, anchor.y + 5);
                this.ctx.closePath();
                this.ctx.fill();
            }
        }
    }
    
    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }
    
    drawNode(node, isHighlighted) {
        // Glow
        const glowRadius = node.radius + (isHighlighted ? 10 : 6);
        const gradient = this.ctx.createRadialGradient(
            node.x, node.y, node.radius,
            node.x, node.y, glowRadius
        );
        const glowAlpha = isHighlighted ? '60' : '30';
        gradient.addColorStop(0, `${node.color}${glowAlpha}`);
        gradient.addColorStop(1, `${node.color}00`);
        
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Node circle
        this.ctx.fillStyle = node.color;
        this.ctx.globalAlpha = 0.2 + node.activation * 0.8;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Border
        this.ctx.globalAlpha = 1;
        this.ctx.strokeStyle = node.color;
        this.ctx.lineWidth = isHighlighted ? 3 : 2;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        this.ctx.stroke();
    }
}

function initTransformer() {
    const canvas = document.getElementById('transformer-canvas');
    if (!canvas) return;
    
    // Initialize (static) visual for context, but focus on interactive next-token UI
    if (!transformerViz) {
        transformerViz = new TransformerVisualizer(canvas);
        transformerViz.setTokens(['<start>']);
        transformerViz.draw();
    } else {
        transformerViz.resize();
        transformerViz.draw();
    }

    if (!transformerInitialized) {
        transformerInitialized = true;

        // Wire up interactive next-token controls
        const inputEl = document.getElementById('transformer-input');
        const tempEl = document.getElementById('transformer-temp');
        const tempDisp = document.getElementById('transformer-temp-display');
        const topkEl = document.getElementById('transformer-topk');
        const topkDisp = document.getElementById('transformer-topk-display');
        const predictBtn = document.getElementById('transformer-predict');
        const appendBtn = document.getElementById('transformer-append');
        const clearBtn = document.getElementById('transformer-clear');
        const charCounter = document.getElementById('char-counter');

        const doPredict = () => {
            // Keep the visualization in view on desktop while interacting
            scrollIntoViewIfDesktop('transformer-canvas', { behavior: 'smooth', block: 'center' });
            // Add loading state
            predictBtn.disabled = true;
            predictBtn.textContent = '🔮 Predicting...';
            
            // Small delay to show the loading state
            setTimeout(() => {
                const preds = predictNextTokens(inputEl.value, parseFloat(tempEl.value), parseInt(topkEl.value));
                updateTransformerResults(preds);
                // Update viz to reflect prompt and predicted top token
                const displayTokens = (inputEl.value || '').trim().length ? inputEl.value.split(/\s+/) : [];
                transformerViz.setTokens(displayTokens);
                transformerViz.setPredictions(preds);
                
                // Update append button to show what will be appended
                if (preds.length > 0 && preds[0].token !== toyLM.eos) {
                    appendBtn.textContent = `➕ Append "${preds[0].token}"`;
                    appendBtn.disabled = false;
                } else {
                    appendBtn.textContent = '➕ Append top prediction';
                    appendBtn.disabled = true;
                }
                
                // Reset predict button
                predictBtn.disabled = false;
                predictBtn.textContent = '🔮 Predict next token';
            }, 100);
        };

        tempEl.addEventListener('input', () => {
            tempDisp.textContent = parseFloat(tempEl.value).toFixed(1);
        });
        topkEl.addEventListener('input', () => {
            topkDisp.textContent = parseInt(topkEl.value, 10);
        });
        predictBtn.addEventListener('click', doPredict);
        inputEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                doPredict();
            }
        });
        inputEl.addEventListener('input', () => {
            // Live update of the visualization to mirror prompt edits
            const displayTokens = (inputEl.value || '').trim().length ? inputEl.value.split(/\s+/) : [];
            transformerViz.setTokens(displayTokens);
            
            // Update character counter
            if (charCounter) {
                const len = inputEl.value.length;
                charCounter.textContent = `${len}/200`;
                charCounter.style.color = len > 150 ? '#e74c3c' : (len > 100 ? '#f39c12' : '#999');
            }
        });
        appendBtn.addEventListener('click', () => {
            const preds = predictNextTokens(inputEl.value, parseFloat(tempEl.value), 1);
            if (preds.length > 0) {
                const tok = preds[0].token;
                if (tok === toyLM.eos) return;
                inputEl.value = (inputEl.value ? inputEl.value + ' ' : '') + tok;
                
                // Trigger input event to update counter
                inputEl.dispatchEvent(new Event('input'));
                
                // Auto-predict after appending
                doPredict();
            }
        });
        clearBtn.addEventListener('click', () => {
            inputEl.value = '';
            updateTransformerResults([]);
            transformerViz.setTokens([]);
            transformerViz.setPredictions([]);
            
            // Reset character counter
            if (charCounter) {
                charCounter.textContent = '0/200';
                charCounter.style.color = '#999';
            }
            
            // Disable append button
            appendBtn.textContent = '➕ Append top prediction';
            appendBtn.disabled = true;
        });
        
        // Initialize append button state
        appendBtn.disabled = true;
    }
    
}

function animateTransformer() {
    if (!state.transformer.running || state.currentDemo !== 'transformer') return;
    
    transformerViz.update();
    transformerViz.draw();
    requestAnimationFrame(animateTransformer);
}

// ====== Deep Perceptron Visualization ======
let deepPerceptronViz = null;
let deepPerceptronInitialized = false;

class DeepPerceptronVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.layers = [];
        this.particles = [];
        this.currentLayer = 0;
        this.layerProgress = 0;
        this.cycleProgress = 0;
        this.resize();
        window.addEventListener('resize', debounce(() => this.resize(), 250));
    }
    
    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.setupLayers();
    }
    
    setupLayers() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const layerSizes = [3, 5, 4, 3, 2];
        const layerColors = ['#4A90E2', '#9B59B6', '#E74C3C', '#F39C12', '#50E3C2'];
        const layerNames = ['INPUT', 'HIDDEN 1', 'HIDDEN 2', 'HIDDEN 3', 'OUTPUT'];
        this.layers = [];
        
        const nodeRadius = Math.max(8, Math.min(15, w / 50));
        const spacing = w / (layerSizes.length + 1);
        
        layerSizes.forEach((size, layerIdx) => {
            const layer = [];
            const x = spacing * (layerIdx + 1);
            const nodeSpacing = h / (size + 1);
            
            for (let i = 0; i < size; i++) {
                layer.push({
                    x: x,
                    y: nodeSpacing * (i + 1),
                    radius: nodeRadius,
                    activation: 0,
                    targetActivation: 0,
                    color: layerColors[layerIdx],
                    layerName: layerNames[layerIdx]
                });
            }
            this.layers.push(layer);
        });
    }
    
    update() {
        if (!this.layers || this.layers.length === 0) return;
        
        const speed = state.deepPerceptron.speed || 3;
        
        // Smooth activation interpolation
        this.layers.forEach(layer => {
            if (!layer) return;
            layer.forEach(node => {
                if (!node) return;
                const diff = (node.targetActivation || 0) - (node.activation || 0);
                node.activation = (node.activation || 0) + diff * 0.15;
            });
        });
        
        // Linear cycle progress (slowed down)
        this.cycleProgress += 0.002 * speed;
        
        // Reset cycle after completing all layers
        if (this.cycleProgress >= 1.0) {
            this.cycleProgress = 0;
            this.currentLayer = 0;
            this.layerProgress = 0;
            
            // Clear particles
            this.particles = [];
            
            // Reset all nodes
            this.layers.forEach(layer => {
                if (!layer) return;
                layer.forEach(node => {
                    if (!node) return;
                    node.activation = 0;
                    node.targetActivation = 0;
                });
            });
            
            // Activate input layer
            if (this.layers[0]) {
                this.layers[0].forEach(node => {
                    if (node) node.targetActivation = 0.8 + Math.random() * 0.2;
                });
            }
            return;
        }
        
        // Determine which layer should be processing
        const totalLayers = this.layers.length;
        const layerDuration = 1.0 / totalLayers;
        const targetLayer = Math.min(totalLayers - 1, Math.floor(this.cycleProgress / layerDuration));
        
        // Smoothly transition between layers
        if (targetLayer !== this.currentLayer && targetLayer < totalLayers) {
            // Spawn particles to next layer
            const currentLayerNodes = this.layers[this.currentLayer];
            const nextLayerNodes = this.layers[targetLayer];
            
            if (currentLayerNodes && nextLayerNodes) {
                nextLayerNodes.forEach((targetNode, idx) => {
                    if (!targetNode) return;
                    
                    // Calculate activation from previous layer
                    let weightedSum = 0;
                    let totalWeight = 0;
                    
                    currentLayerNodes.forEach(sourceNode => {
                        if (!sourceNode) return;
                        const weight = 0.5 + Math.random() * 0.5;
                        weightedSum += (sourceNode.activation || 0) * weight;
                        totalWeight += weight;
                    });
                    
                    // Set target activation (ReLU-like)
                    targetNode.targetActivation = Math.min(1.0, Math.max(0.1, weightedSum / totalWeight));
                    
                    // Create particles from a few source nodes
                    const numParticles = Math.min(2, currentLayerNodes.length);
                    for (let i = 0; i < numParticles; i++) {
                        const sourceNode = currentLayerNodes[Math.floor(Math.random() * currentLayerNodes.length)];
                        if (!sourceNode || (sourceNode.activation || 0) < 0.2) continue;
                        
                        this.particles.push({
                            x: sourceNode.x,
                            y: sourceNode.y,
                            targetX: targetNode.x,
                            targetY: targetNode.y,
                            progress: 0,
                            life: 1,
                            speed: 0.015
                        });
                    }
                });
            }
            
            this.currentLayer = targetLayer;
        }
        
        // Update particles smoothly
        this.particles = this.particles.filter(p => {
            if (!p) return false;
            p.progress = Math.min(1, (p.progress || 0) + (p.speed || 0.015));
            p.life = Math.max(0, (p.life || 1) - 0.008);
            return p.life > 0 && p.progress < 1;
        });
    }
    
    draw() {
        if (!this.ctx || !this.canvas || !this.layers) return;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw connections
        for (let i = 0; i < this.layers.length - 1; i++) {
            const layer = this.layers[i];
            const nextLayer = this.layers[i + 1];
            if (!layer || !nextLayer) continue;
            
            layer.forEach(node1 => {
                if (!node1) return;
                nextLayer.forEach(node2 => {
                    if (!node2) return;
                    
                    const isActive = (i === this.currentLayer || i === this.currentLayer - 1);
                    this.ctx.strokeStyle = isActive ? 'rgba(200, 200, 200, 0.25)' : 'rgba(200, 200, 200, 0.08)';
                    this.ctx.lineWidth = isActive ? 1.5 : 1;
                    this.ctx.beginPath();
                    this.ctx.moveTo(node1.x, node1.y);
                    this.ctx.lineTo(node2.x, node2.y);
                    this.ctx.stroke();
                });
            });
        }
        
        // Draw particles
        this.particles.forEach(p => {
            if (!p) return;
            
            const t = this.easeInOutCubic(p.progress || 0);
            const x = (p.x || 0) + ((p.targetX || 0) - (p.x || 0)) * t;
            const y = (p.y || 0) + ((p.targetY || 0) - (p.y || 0)) * t;
            
            const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, 6);
            gradient.addColorStop(0, `rgba(102, 126, 234, ${p.life || 0})`);
            gradient.addColorStop(1, `rgba(102, 126, 234, 0)`);
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 5, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        // Draw nodes
        this.layers.forEach((layer, layerIdx) => {
            if (!layer) return;
            
            layer.forEach(node => {
                if (!node) return;
                
                const activation = Math.max(0, Math.min(1, node.activation || 0));
                
                // Glow for active nodes
                if (activation > 0.3) {
                    const glowRadius = (node.radius || 10) + 8;
                    const gradient = this.ctx.createRadialGradient(
                        node.x, node.y, node.radius || 10,
                        node.x, node.y, glowRadius
                    );
                    gradient.addColorStop(0, `${node.color}40`);
                    gradient.addColorStop(1, `${node.color}00`);
                    
                    this.ctx.fillStyle = gradient;
                    this.ctx.beginPath();
                    this.ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
                    this.ctx.fill();
                }
                
                // Node fill
                this.ctx.fillStyle = node.color;
                this.ctx.globalAlpha = 0.2 + activation * 0.8;
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, node.radius || 10, 0, Math.PI * 2);
                this.ctx.fill();
                
                // Node border
                this.ctx.globalAlpha = 1;
                this.ctx.strokeStyle = node.color;
                this.ctx.lineWidth = layerIdx === this.currentLayer ? 3 : 2;
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, node.radius || 10, 0, Math.PI * 2);
                this.ctx.stroke();
            });
        });
    }
    
    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }
}

function initDeepPerceptron() {
    const canvas = document.getElementById('deep-canvas');
    if (!canvas) return;
    
    if (!deepPerceptronViz) {
        deepPerceptronViz = new DeepPerceptronVisualizer(canvas);
    } else {
        deepPerceptronViz.resize();
    }
    
    if (!deepPerceptronInitialized) {
        deepPerceptronInitialized = true;
        
        document.getElementById('deep-start').addEventListener('click', () => {
            // Ensure canvas is visible while animation runs (desktop only)
            scrollIntoViewIfDesktop('deep-canvas');
            state.deepPerceptron.running = !state.deepPerceptron.running;
            document.getElementById('deep-start').textContent = 
                state.deepPerceptron.running ? 'Pause Training' : 'Start Training';
            if (state.deepPerceptron.running) animateDeepPerceptron();
        });
        
        document.getElementById('deep-reset').addEventListener('click', () => {
            resetDeepPerceptron();
        });
        
        document.getElementById('deep-speed').addEventListener('input', (e) => {
            state.deepPerceptron.speed = parseFloat(e.target.value);
        });
    }
    
    deepPerceptronViz.draw();
}

function animateDeepPerceptron() {
    if (!state.deepPerceptron.running || state.currentDemo !== 'deep-perceptron') return;
    
    if (deepPerceptronViz) {
        deepPerceptronViz.update();
        deepPerceptronViz.draw();
    }
    
    requestAnimationFrame(animateDeepPerceptron);
}

// ====== Normalizing Flow Visualization ======
let normalizingFlowViz = null;
let normalizingFlowInitialized = false;

class NormalizingFlowVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.layers = [];
        this.particles = [];
        this.spawnTimer = 0;
        this.resize();
        this.setupLayers();
    }
    
    resize() {
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
        this.setupLayers();
    }
    
    setupLayers() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const layerSizes = [8, 6, 6, 6, 8];
        this.layers = [];
        
        const nodeRadius = Math.max(8, Math.min(12, w / 60));
        const spacing = w / (layerSizes.length + 1);
        
        layerSizes.forEach((size, layerIdx) => {
            const layer = [];
            const x = spacing * (layerIdx + 1);
            const nodeSpacing = h / (size + 1);
            
            for (let i = 0; i < size; i++) {
                layer.push({
                    x: x,
                    y: nodeSpacing * (i + 1),
                    radius: nodeRadius,
                    activation: 0,
                    layerIndex: layerIdx,
                    color: layerIdx === 0 ? '#667EEA' : 
                           layerIdx === 1 ? '#9B59B6' : 
                           layerIdx === 2 ? '#E056FD' : 
                           layerIdx === 3 ? '#3498DB' : '#F77F17'
                });
            }
            this.layers.push(layer);
        });
    }
    
    update() {
        // Decay activations
        this.layers.forEach(layer => {
            layer.forEach(node => node.activation *= 0.92);
        });

        const userSpeed = state.normalizingFlow?.speed || 1;
        const progressStep = 0.008 * userSpeed;
        
        // Update particles and spawn recursively
        this.particles.forEach(p => {
            p.progress += progressStep;
            
            // When particle arrives, activate target and spawn next
            if (p.progress >= 0.95 && !p.spawned) {
                p.targetNode.activation = 1;
                this.spawnNext(p.targetNode);
                p.spawned = true;
            }
        });
        
        // Remove finished particles
        this.particles = this.particles.filter(p => p.progress < 1.2);
        
        // Periodically spawn from first layer
        this.spawnTimer++;
        if (this.spawnTimer % Math.max(40, 100 / userSpeed) === 0) {
            const node = this.layers[0][Math.floor(Math.random() * this.layers[0].length)];
            node.activation = 1;
            this.spawnNext(node);
        }
    }
    
    spawnNext(node) {
        if (node.layerIndex >= this.layers.length - 1) return;
        
        const nextLayer = this.layers[node.layerIndex + 1];
        const target = nextLayer[Math.floor(Math.random() * nextLayer.length)];
        
        this.particles.push({
            x: node.x,
            y: node.y,
            targetX: target.x,
            targetY: target.y,
            targetNode: target,
            progress: 0,
            spawned: false
        });
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        const ease = t => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;
        
        // Draw connections
        for (let i = 0; i < this.layers.length - 1; i++) {
            this.layers[i].forEach(n1 => {
                this.layers[i + 1].forEach(n2 => {
                    this.ctx.strokeStyle = 'rgba(200, 200, 200, 0.1)';
                    this.ctx.lineWidth = 1;
                    this.ctx.beginPath();
                    this.ctx.moveTo(n1.x, n1.y);
                    this.ctx.lineTo(n2.x, n2.y);
                    this.ctx.stroke();
                });
            });
        }
        
        // Draw particles
        this.particles.forEach(p => {
            const t = Math.min(p.progress, 1);
            const eased = ease(t);
            const x = p.x + (p.targetX - p.x) * eased;
            const y = p.y + (p.targetY - p.y) * eased;
            const alpha = 1 - p.progress / 1.2;
            
            const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, 7);
            gradient.addColorStop(0, `rgba(155, 89, 182, ${alpha})`);
            gradient.addColorStop(1, `rgba(155, 89, 182, 0)`);
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 7, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        // Draw nodes
        this.layers.forEach(layer => {
            layer.forEach(node => {
                if (node.activation > 0.3) {
                    const glow = this.ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.radius * 2);
                    glow.addColorStop(0, `${node.color}AA`);
                    glow.addColorStop(1, 'rgba(0, 0, 0, 0)');
                    this.ctx.fillStyle = glow;
                    this.ctx.beginPath();
                    this.ctx.arc(node.x, node.y, node.radius * 2, 0, Math.PI * 2);
                    this.ctx.fill();
                }
                
                this.ctx.fillStyle = node.color;
                this.ctx.globalAlpha = 0.2 + node.activation * 0.8;
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                this.ctx.fill();
                
                this.ctx.globalAlpha = 1;
                this.ctx.strokeStyle = node.color;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                this.ctx.stroke();
            });
        });
    }
}

function initNormalizingFlow() {
    const canvas = document.getElementById('nf-canvas');
    if (!canvas) return;
    
    if (!normalizingFlowViz) {
        normalizingFlowViz = new NormalizingFlowVisualizer(canvas);
    } else {
        normalizingFlowViz.resize();
    }
    
    if (!normalizingFlowInitialized) {
        normalizingFlowInitialized = true;
        
        document.getElementById('nf-start').addEventListener('click', () => {
            // Ensure canvas is visible while animation runs (desktop only)
            scrollIntoViewIfDesktop('nf-canvas');
            state.normalizingFlow.running = !state.normalizingFlow.running;
            document.getElementById('nf-start').textContent = 
                state.normalizingFlow.running ? 'Pause Flow' : 'Start Flow';
            if (state.normalizingFlow.running) animateNormalizingFlow();
        });
        
        document.getElementById('nf-reset').addEventListener('click', () => {
            resetNormalizingFlow();
        });
        
        document.getElementById('nf-speed').addEventListener('input', (e) => {
            state.normalizingFlow.speed = parseInt(e.target.value);
        });
    }
    
    normalizingFlowViz.draw();
}

function animateNormalizingFlow() {
    if (!state.normalizingFlow.running || state.currentDemo !== 'normalizing-flow') return;
    
    normalizingFlowViz.update();
    normalizingFlowViz.draw();
    requestAnimationFrame(animateNormalizingFlow);
}

// ====== VAE Visualization ======
let vaeViz = null;
let vaeInitialized = false;

class VAEVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.layers = [];
        this.particles = [];
        this.currentLayer = 0;
        this.dataFlowTimer = 0;
        this.resize();
        this.setupLayers();
    }
    
    resize() {
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
        this.setupLayers();
    }
    
    setupLayers() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const layerSizes = [8, 5, 3, 5, 8];
        this.layers = [];
        
        // Responsive node radius
        const nodeRadius = Math.max(9, Math.min(14, w / 55));
        const spacing = w / (layerSizes.length + 1);
        
        layerSizes.forEach((size, layerIdx) => {
            const layer = [];
            const x = spacing * (layerIdx + 1);
            const nodeSpacing = h / (size + 1);
            
            for (let i = 0; i < size; i++) {
                layer.push({
                    x: x,
                    y: nodeSpacing * (i + 1),
                    radius: nodeRadius,
                    activation: 0,
                    color: layerIdx === 0 ? '#4ECDC4' : 
                           layerIdx === 2 ? '#FFD93D' :
                           layerIdx === 4 ? '#4ECDC4' : '#C44569'
                });
            }
            this.layers.push(layer);
        });
    }
    
    update() {
        // Smooth decay for all nodes
        this.layers.forEach(layer => {
            layer.forEach(node => {
                node.activation *= 0.92;
            });
        });
        
        // Linear layer-by-layer flow
        this.dataFlowTimer++;
        if (this.dataFlowTimer === 1 || this.dataFlowTimer % 15 === 0) {
            if (this.currentLayer < this.layers.length - 1) {
                const sourceLayer = this.layers[this.currentLayer];
                const targetLayer = this.layers[this.currentLayer + 1];
                
                sourceLayer.forEach(sourceNode => {
                    if (this.currentLayer === 0 || sourceNode.activation > 0.3) {
                        sourceNode.activation = 0.9;
                        const targetNode = targetLayer[Math.floor(Math.random() * targetLayer.length)];
                        
                        this.particles.push({
                            x: sourceNode.x,
                            y: sourceNode.y,
                            targetX: targetNode.x,
                            targetY: targetNode.y,
                            progress: 0,
                            life: 1,
                            targetNode: targetNode
                        });
                    }
                });
                
                this.currentLayer++;
                if (this.currentLayer >= this.layers.length - 1) {
                    this.currentLayer = 0;
                }
            }
        }
        
        // Update particles with smooth easing
        this.particles = this.particles.filter(p => {
            p.progress += 0.08;
            p.life -= 0.02;
            
            // Activate target when particle arrives
            if (p.progress >= 0.9 && p.targetNode) {
                p.targetNode.activation = 0.9;
                p.targetNode = null;
            }
            
            return p.life > 0 && p.progress < 1;
        });
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Smooth easing function
        const easeInOutCubic = (t) => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;
        
        // Draw connections with subtle lines
        for (let i = 0; i < this.layers.length - 1; i++) {
            this.layers[i].forEach(node1 => {
                this.layers[i + 1].forEach(node2 => {
                    this.ctx.strokeStyle = 'rgba(200, 200, 200, 0.12)';
                    this.ctx.lineWidth = 1;
                    this.ctx.beginPath();
                    this.ctx.moveTo(node1.x, node1.y);
                    this.ctx.lineTo(node2.x, node2.y);
                    this.ctx.stroke();
                });
            });
        }
        
        // Draw particles with smooth easing
        this.particles.forEach(p => {
            const eased = easeInOutCubic(p.progress);
            const x = p.x + (p.targetX - p.x) * eased;
            const y = p.y + (p.targetY - p.y) * eased;
            
            const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, 5);
            gradient.addColorStop(0, `rgba(255, 217, 61, ${p.life})`);
            gradient.addColorStop(1, `rgba(255, 217, 61, 0)`);
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 5, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        // Draw nodes with glow effect
        this.layers.forEach(layer => {
            layer.forEach(node => {
                // Inner glow for active nodes
                if (node.activation > 0.3) {
                    const glowGradient = this.ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.radius * 1.5);
                    glowGradient.addColorStop(0, `${node.color}80`);
                    glowGradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
                    this.ctx.fillStyle = glowGradient;
                    this.ctx.beginPath();
                    this.ctx.arc(node.x, node.y, node.radius * 1.5, 0, Math.PI * 2);
                    this.ctx.fill();
                }
                
                this.ctx.fillStyle = node.color;
                this.ctx.globalAlpha = 0.2 + node.activation * 0.8;
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                this.ctx.fill();
                
                this.ctx.globalAlpha = 1;
                this.ctx.strokeStyle = node.color;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                this.ctx.stroke();
            });
        });
    }
}

function initVAE() {
    const canvas = document.getElementById('vae-canvas');
    if (!canvas) return;
    
    if (!vaeViz) {
        vaeViz = new VAEVisualizer(canvas);
    } else {
        vaeViz.resize();
    }
    
    if (!vaeInitialized) {
        vaeInitialized = true;
        
        document.getElementById('vae-start').addEventListener('click', () => {
            // Ensure canvas is visible while animation runs (desktop only)
            scrollIntoViewIfDesktop('vae-canvas');
            state.vae.running = !state.vae.running;
            document.getElementById('vae-start').textContent = 
                state.vae.running ? 'Pause Encoding' : 'Start Encoding';
            if (state.vae.running) animateVAE();
        });
        
        document.getElementById('vae-reset').addEventListener('click', () => {
            resetVAE();
        });
        
        document.getElementById('vae-speed').addEventListener('input', (e) => {
            state.vae.speed = parseInt(e.target.value);
        });
    }
    
    vaeViz.draw();
}

function animateVAE() {
    if (!state.vae.running || state.currentDemo !== 'vae') return;
    
    vaeViz.update();
    vaeViz.draw();
    
    setTimeout(() => {
        requestAnimationFrame(animateVAE);
    }, 1000 / state.vae.speed);
}

// ====== CNN Encoder-Decoder Visualization ======
let cnnViz = null;
let cnnInitialized = false;

class CNNEncoderDecoderVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.featureMaps = [];
        this.particles = [];
        this.resize();
        this.setupMaps();
    }
    
    resize() {
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
        this.setupMaps();
    }
    
    setupMaps() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        // Responsive map sizes - scale down on smaller screens
        const baseMapSizes = [80, 56, 32, 56, 80];
        const scaleFactor = Math.max(0.4, Math.min(1, w / 600));
        const mapSizes = baseMapSizes.map(size => Math.floor(size * scaleFactor));
        const mapCounts = [3, 6, 12, 6, 3];
        this.featureMaps = [];
        this.currentLayer = -1;
        this.dataFlowTimer = 0;
        
        const spacing = w / (mapSizes.length + 1);
        
        mapSizes.forEach((size, idx) => {
            const maps = [];
            const x = spacing * (idx + 1);
            const count = mapCounts[idx];
            
            // Calculate total height of all stacked maps
            const totalHeight = count * size + (count - 1) * Math.max(5, 10 * scaleFactor);
            const startY = h / 2 - totalHeight / 2;
            
            for (let i = 0; i < count; i++) {
                maps.push({
                    x: x - size / 2,  // Center horizontally
                    y: startY + i * (size + Math.max(5, 10 * scaleFactor)),  // Stack vertically with spacing
                    width: size,
                    height: size,
                    activation: 0,
                    targetActivation: 0,
                    color: idx === 0 ? '#3498db' : 
                           idx === 2 ? '#e74c3c' :
                           idx === 4 ? '#2ecc71' : '#9b59b6'
                });
            }
            this.featureMaps.push(maps);
        });
    }
    
    update() {
        // Smooth activation decay (only for nodes that were hit)
        this.featureMaps.forEach(maps => {
            maps.forEach(map => {
                if (map.activation > 0) {
                    map.activation *= 0.90;
                    if (map.activation < 0.05) map.activation = 0;
                }
            });
        });
        
        // Linear data flow: process one layer at a time
        this.dataFlowTimer++;
        
        // Activate first layer initially and create first particles
        if (this.dataFlowTimer === 1) {
            this.currentLayer = 0;
            // Activate all nodes in first layer
            this.featureMaps[0].forEach(map => {
                map.activation = 0.9;
            });
        }
        
        // Every 25 frames, move to next layer (smoother flow timing)
        if ((this.dataFlowTimer === 1 || this.dataFlowTimer % 25 === 0) && this.currentLayer >= 0 && this.currentLayer < this.featureMaps.length - 1) {
            const sourceLayer = this.currentLayer;
            const targetLayer = this.currentLayer + 1;
            
            // Create particles from current layer to next
            const sourceMaps = this.featureMaps[sourceLayer];
            const targetMaps = this.featureMaps[targetLayer];
            
            // Only create particles from ACTIVE source maps
            sourceMaps.forEach((sourceMap, sourceIdx) => {
                if (sourceMap.activation > 0.3) { // Only if node is active
                    const numParticles = Math.ceil(targetMaps.length / sourceMaps.length);
                    for (let i = 0; i < numParticles; i++) {
                        const targetIdx = Math.floor(Math.random() * targetMaps.length);
                        const targetMap = targetMaps[targetIdx];
                        
                        this.particles.push({
                            x: sourceMap.x + sourceMap.width / 2,
                            y: sourceMap.y + sourceMap.height / 2,
                            targetX: targetMap.x + targetMap.width / 2,
                            targetY: targetMap.y + targetMap.height / 2,
                            progress: 0,
                            life: 1,
                            sourceLayer: sourceLayer,
                            targetLayer: targetLayer,
                            targetMapIndex: targetIdx
                        });
                    }
                }
            });
            
            this.currentLayer++;
        }
        
        // Reset after completing the flow
        if (this.currentLayer >= this.featureMaps.length - 1 && this.particles.length === 0) {
            this.dataFlowTimer = 0;
            this.currentLayer = -1;
        }
        
        // Update particles and activate ONLY target maps that are hit
        this.particles = this.particles.filter(p => {
            p.progress += 0.035; // Smoother, more controlled particle movement
            p.life -= 0.02; // Slower life decay for better visibility
            
            // Activate target map when particle is close to arriving
            if (p.progress >= 0.75 && p.progress < 0.85) {
                const targetMaps = this.featureMaps[p.targetLayer];
                if (targetMaps && targetMaps[p.targetMapIndex]) {
                    // Set activation, don't add to it (prevents stacking)
                    targetMaps[p.targetMapIndex].activation = 0.9;
                }
            }
            
            return p.life > 0 && p.progress < 1;
        });
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw connection lines based on active particles - shows actual data flow
        this.ctx.globalAlpha = 0.15;
        this.ctx.strokeStyle = '#7f8c8d';
        this.ctx.lineWidth = 1.5;
        
        // Draw lines for each active particle to show the actual connections being used
        this.particles.forEach(p => {
            if (p.progress < 0.95) { // Only show line while particle is traveling
                this.ctx.beginPath();
                this.ctx.moveTo(p.x, p.y);
                this.ctx.lineTo(p.targetX, p.targetY);
                this.ctx.stroke();
            }
        });
        
        this.ctx.globalAlpha = 1;
        
        // Draw feature maps with smooth shading
        this.featureMaps.forEach((maps, layerIdx) => {
            maps.forEach(map => {
                // Fill with activation-based opacity
                this.ctx.fillStyle = map.color;
                this.ctx.globalAlpha = 0.15 + map.activation * 0.7;
                this.ctx.fillRect(map.x, map.y, map.width, map.height);
                
                // Draw border
                this.ctx.globalAlpha = 0.4 + map.activation * 0.6;
                this.ctx.strokeStyle = map.color;
                this.ctx.lineWidth = 2;
                this.ctx.strokeRect(map.x, map.y, map.width, map.height);
                
                // Add inner glow when active
                if (map.activation > 0.3) {
                    this.ctx.globalAlpha = map.activation * 0.3;
                    this.ctx.strokeStyle = '#ffffff';
                    this.ctx.lineWidth = 1;
                    this.ctx.strokeRect(map.x + 2, map.y + 2, map.width - 4, map.height - 4);
                }
            });
        });
        
        this.ctx.globalAlpha = 1;
        
        // Draw particles with trail effect
        this.particles.forEach(p => {
            const x = p.x + (p.targetX - p.x) * this.easeInOutCubic(p.progress);
            const y = p.y + (p.targetY - p.y) * this.easeInOutCubic(p.progress);
            
            // Particle glow
            const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, 8);
            gradient.addColorStop(0, `rgba(255, 255, 255, ${p.life * 0.9})`);
            gradient.addColorStop(0.3, `rgba(52, 152, 219, ${p.life * 0.7})`);
            gradient.addColorStop(1, `rgba(52, 152, 219, 0)`);
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 8, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Core particle
            this.ctx.fillStyle = `rgba(255, 255, 255, ${p.life})`;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 3, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }
    
    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }
}

function initCNNEncoderDecoder() {
    const canvas = document.getElementById('cnn-canvas');
    if (!canvas) return;
    
    if (!cnnViz) {
        cnnViz = new CNNEncoderDecoderVisualizer(canvas);
    } else {
        cnnViz.resize();
    }
    
    if (!cnnInitialized) {
        cnnInitialized = true;
        
        document.getElementById('cnn-start').addEventListener('click', () => {
            // Ensure canvas is visible while animation runs (desktop only)
            scrollIntoViewIfDesktop('cnn-canvas');
            state.cnnEncoderDecoder.running = !state.cnnEncoderDecoder.running;
            document.getElementById('cnn-start').textContent = 
                state.cnnEncoderDecoder.running ? 'Pause Processing' : 'Process Image';
            if (state.cnnEncoderDecoder.running) animateCNN();
        });
        
        document.getElementById('cnn-reset').addEventListener('click', () => {
            resetCNN();
        });
        
        document.getElementById('cnn-speed').addEventListener('input', (e) => {
            state.cnnEncoderDecoder.speed = parseInt(e.target.value);
        });
    }
    
    cnnViz.draw();
}

let cnnLastFrameTime = 0;
const CNN_BASE_FRAME_DELAY = 1000 / 60; // Target 60fps for smooth animation

function animateCNN(currentTime = 0) {
    if (!state.cnnEncoderDecoder.running || state.currentDemo !== 'cnn-encoder-decoder') return;
    
    // Calculate time-based throttling for speed control
    const targetDelay = CNN_BASE_FRAME_DELAY * (10 / state.cnnEncoderDecoder.speed);
    
    if (currentTime - cnnLastFrameTime >= targetDelay) {
        cnnViz.update();
        cnnViz.draw();
        cnnLastFrameTime = currentTime;
    }
    
    requestAnimationFrame(animateCNN);
}

// ====== Mamba2 Visualization ======
let mamba2Viz = null;
let mamba2Initialized = false;

class Mamba2Visualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.layers = [];
        this.stateHistory = [];
        this.particles = [];
        this.currentLayer = 0;
        this.dataFlowTimer = 0;
        this.resize();
        this.setupLayers();
    }
    
    resize() {
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
        this.setupLayers();
    }
    
    setupLayers() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        const layerSizes = [6, 8, 4, 8, 6];
        this.layers = [];
        
        // Responsive node radius
        const nodeRadius = Math.max(8, Math.min(13, w / 55));
        const spacing = w / (layerSizes.length + 1);
        
        layerSizes.forEach((size, layerIdx) => {
            const layer = [];
            const x = spacing * (layerIdx + 1);
            const nodeSpacing = h / (size + 1);
            
            for (let i = 0; i < size; i++) {
                layer.push({
                    x: x,
                    y: nodeSpacing * (i + 1),
                    radius: nodeRadius,
                    activation: 0,
                    color: layerIdx === 0 ? '#6C5CE7' : 
                           layerIdx === 2 ? '#FD79A8' :
                           layerIdx === 4 ? '#00B894' : '#FDCB6E'
                });
            }
            this.layers.push(layer);
        });
    }
    
    update() {
        // Smooth decay for all nodes
        this.layers.forEach(layer => {
            layer.forEach(node => {
                node.activation *= 0.92;
            });
        });
        
        // Linear layer-by-layer flow - slowed down significantly
        this.dataFlowTimer++;
        if (this.dataFlowTimer === 1 || this.dataFlowTimer % 30 === 0) {
            if (this.currentLayer < this.layers.length - 1) {
                const sourceLayer = this.layers[this.currentLayer];
                const targetLayer = this.layers[this.currentLayer + 1];
                
                sourceLayer.forEach(sourceNode => {
                    if (this.currentLayer === 0 || sourceNode.activation > 0.3) {
                        sourceNode.activation = 0.9;
                        const targetNode = targetLayer[Math.floor(Math.random() * targetLayer.length)];
                        
                        this.particles.push({
                            x: sourceNode.x,
                            y: sourceNode.y,
                            targetX: targetNode.x,
                            targetY: targetNode.y,
                            progress: 0,
                            life: 1,
                            targetNode: targetNode
                        });
                    }
                });
                
                this.currentLayer++;
                if (this.currentLayer >= this.layers.length - 1) {
                    this.currentLayer = 0;
                }
            }
        }
        
        // Update particles with smooth easing - slowed down
        this.particles = this.particles.filter(p => {
            p.progress += 0.04;  // Reduced from 0.08
            p.life -= 0.01;      // Reduced from 0.02
            
            // Activate target when particle arrives
            if (p.progress >= 0.9 && p.targetNode) {
                p.targetNode.activation = 0.9;
                p.targetNode = null;
            }
            
            return p.life > 0 && p.progress < 1;
        });
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Smooth easing function
        const easeInOutCubic = (t) => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;
        
        // Draw connections with subtle lines
        for (let i = 0; i < this.layers.length - 1; i++) {
            this.layers[i].forEach(node1 => {
                this.layers[i + 1].forEach(node2 => {
                    this.ctx.strokeStyle = 'rgba(200, 200, 200, 0.12)';
                    this.ctx.lineWidth = 1;
                    this.ctx.beginPath();
                    this.ctx.moveTo(node1.x, node1.y);
                    this.ctx.lineTo(node2.x, node2.y);
                    this.ctx.stroke();
                });
            });
        }
        
        // Draw particles with smooth easing
        this.particles.forEach(p => {
            const eased = easeInOutCubic(p.progress);
            const x = p.x + (p.targetX - p.x) * eased;
            const y = p.y + (p.targetY - p.y) * eased;
            
            const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, 6);
            gradient.addColorStop(0, `rgba(253, 203, 110, ${p.life})`);
            gradient.addColorStop(1, `rgba(253, 203, 110, 0)`);
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 6, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        // Draw nodes with glow effect
        this.layers.forEach(layer => {
            layer.forEach(node => {
                // Inner glow for active nodes
                if (node.activation > 0.3) {
                    const glowGradient = this.ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.radius * 1.5);
                    glowGradient.addColorStop(0, `${node.color}80`);
                    glowGradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
                    this.ctx.fillStyle = glowGradient;
                    this.ctx.beginPath();
                    this.ctx.arc(node.x, node.y, node.radius * 1.5, 0, Math.PI * 2);
                    this.ctx.fill();
                }
                
                this.ctx.fillStyle = node.color;
                this.ctx.globalAlpha = 0.2 + node.activation * 0.8;
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                this.ctx.fill();
                
                this.ctx.globalAlpha = 1;
                this.ctx.strokeStyle = node.color;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
                this.ctx.stroke();
            });
        });
    }
}

function initMamba2() {
    const canvas = document.getElementById('mamba2-canvas');
    if (!canvas) return;
    
    if (!mamba2Viz) {
        mamba2Viz = new Mamba2Visualizer(canvas);
    } else {
        mamba2Viz.resize();
    }
    
    if (!mamba2Initialized) {
        mamba2Initialized = true;
        
        document.getElementById('mamba2-start').addEventListener('click', () => {
            // Ensure canvas is visible while animation runs (desktop only)
            scrollIntoViewIfDesktop('mamba2-canvas');
            state.mamba2.running = !state.mamba2.running;
            document.getElementById('mamba2-start').textContent = 
                state.mamba2.running ? 'Pause Processing' : 'Process Sequence';
            if (state.mamba2.running) animateMamba2();
        });
        
        document.getElementById('mamba2-reset').addEventListener('click', () => {
            resetMamba2();
        });
        
        document.getElementById('mamba2-speed').addEventListener('input', (e) => {
            state.mamba2.speed = parseInt(e.target.value);
        });
    }
    
    mamba2Viz.draw();
}

function animateMamba2() {
    if (!state.mamba2.running || state.currentDemo !== 'mamba2') return;
    
    mamba2Viz.update();
    mamba2Viz.draw();
    
    // Add delay based on speed setting for better control
    setTimeout(() => {
        requestAnimationFrame(animateMamba2);
    }, 1000 / (state.mamba2.speed * 6)); // Significantly slower than before
}

// ====== CUDA Visualization ======
let cudaViz = null;
let cudaInitialized = false;

class CUDAVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.animationTime = 0;
        this.cycleDuration = 200; // frames per complete cycle (shorter for smoother flow)
        this.blocks = [];
        this.globalMemory = [];
        this.currentPhase = 'idle'; // idle, load, compute, store
        this.resize();
        this.setupArchitecture();
    }
    
    resize() {
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
        this.setupArchitecture();
    }
    
    setupArchitecture() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        // Memory regions on the right side
        this.globalMemory = {
            x: w * 0.75,
            y: h * 0.15,
            width: w * 0.2,
            height: h * 0.7,
            cells: [],
            label: 'Global Memory'
        };
        
        // Create memory cells
        const cellCount = 16;
        const cellHeight = this.globalMemory.height / cellCount;
        for (let i = 0; i < cellCount; i++) {
            this.globalMemory.cells.push({
                x: this.globalMemory.x,
                y: this.globalMemory.y + i * cellHeight,
                width: this.globalMemory.width,
                height: cellHeight - 2,
                data: Math.random() * 100,
                accessed: 0,
                index: i
            });
        }
        
        // SM (Streaming Multiprocessor) blocks on the left
        this.blocks = [];
        const blockCols = 3;
        const blockRows = 2;
        const smWidth = (w * 0.65) / blockCols;
        const smHeight = (h * 0.7) / blockRows;
        
        for (let row = 0; row < blockRows; row++) {
            for (let col = 0; col < blockCols; col++) {
                const block = {
                    x: w * 0.05 + col * smWidth,
                    y: h * 0.15 + row * smHeight,
                    width: smWidth - 15,
                    height: smHeight - 15,
                    id: row * blockCols + col,
                    phase: 'idle',
                    phaseProgress: 0,
                    warps: [],
                    sharedMemory: [],
                    registers: [],
                    activeWarp: 0,
                    memoryAccesses: []
                };
                
                // Create warps (groups of 32 threads, visualized as 8x4)
                const warpCount = 2;
                const threadsPerWarp = 32;
                const threadCols = 8;
                const threadRows = 4;
                const warpAreaHeight = (block.height - 30) / warpCount;
                
                for (let w = 0; w < warpCount; w++) {
                    const warp = {
                        id: w,
                        threads: [],
                        active: false,
                        syncPoint: 0
                    };
                    
                    const threadSize = Math.min(
                        (block.width - 20) / threadCols,
                        warpAreaHeight / threadRows
                    ) - 2;
                    
                    const startX = block.x + 10;
                    const startY = block.y + 25 + w * warpAreaHeight;
                    
                    for (let tRow = 0; tRow < threadRows; tRow++) {
                        for (let tCol = 0; tCol < threadCols; tCol++) {
                            const threadId = tRow * threadCols + tCol;
                            warp.threads.push({
                                id: threadId,
                                x: startX + tCol * (threadSize + 2),
                                y: startY + tRow * (threadSize + 2),
                                size: threadSize,
                                state: 'idle', // idle, loading, computing, storing
                                value: 0,
                                targetMemory: threadId % 16,
                                computeProgress: 0
                            });
                        }
                    }
                    
                    block.warps.push(warp);
                }
                
                // Shared memory for this SM
                block.sharedMemory = Array(8).fill(0).map((_, i) => ({
                    value: 0,
                    accessed: 0,
                    index: i
                }));
                
                this.blocks.push(block);
            }
        }
    }
    
    update() {
        this.animationTime++;
        const cycle = this.animationTime % this.cycleDuration;
        
        // Determine global phase based on cycle - shorter, more balanced phases
        if (cycle < 30) {
            this.currentPhase = 'load';
        } else if (cycle < 140) {
            this.currentPhase = 'compute';
        } else if (cycle < 170) {
            this.currentPhase = 'store';
        } else {
            this.currentPhase = 'idle';
        }
        
        // Update each SM block
        this.blocks.forEach((block, blockIdx) => {
            const blockDelay = blockIdx * 3; // Stagger block execution (reduced for faster flow)
            const localCycle = (cycle + this.cycleDuration - blockDelay) % this.cycleDuration;
            
            // Update block phase
            if (localCycle < 30) {
                block.phase = 'load';
                block.phaseProgress = localCycle / 30;
            } else if (localCycle < 140) {
                block.phase = 'compute';
                block.phaseProgress = (localCycle - 30) / 110;
            } else if (localCycle < 170) {
                block.phase = 'store';
                block.phaseProgress = (localCycle - 140) / 30;
            } else {
                block.phase = 'idle';
                block.phaseProgress = 0;
            }
            
            // Update warps
            block.warps.forEach((warp, warpIdx) => {
                const warpDelay = warpIdx * 8; // Reduced delay for smoother transitions
                const warpCycle = (localCycle + this.cycleDuration - warpDelay) % this.cycleDuration;
                
                warp.active = warpCycle < 170;
                warp.syncPoint = Math.floor(warpCycle / 30) % 3;
                
                // Update threads in warp (all threads in a warp execute in lockstep)
                warp.threads.forEach(thread => {
                    if (warpCycle < 30) {
                        // Loading phase
                        thread.state = 'loading';
                        const loadProgress = warpCycle / 30;
                        thread.value = loadProgress * 100;
                        
                        // Mark memory access with pulsing
                        if (warpCycle % 5 === 0) {
                            this.globalMemory.cells[thread.targetMemory].accessed = 1;
                        }
                    } else if (warpCycle < 140) {
                        // Computing phase - more dynamic visualization
                        thread.state = 'computing';
                        thread.computeProgress = (warpCycle - 30) / 110;
                        // More dynamic value changes during compute
                        const computePhase = thread.computeProgress * Math.PI * 6;
                        thread.value = 50 + Math.sin(computePhase + thread.id * 0.2) * 50;
                        
                        // More frequent shared memory access during compute
                        if (warpCycle % 12 === 0) {
                            const smIdx = thread.id % block.sharedMemory.length;
                            block.sharedMemory[smIdx].accessed = 1;
                            block.sharedMemory[smIdx].value = thread.value;
                        }
                    } else if (warpCycle < 170) {
                        // Storing phase
                        thread.state = 'storing';
                        const storeProgress = (warpCycle - 140) / 30;
                        
                        // Mark memory write with pulsing
                        if (warpCycle % 5 === 0) {
                            this.globalMemory.cells[thread.targetMemory].accessed = 1;
                            this.globalMemory.cells[thread.targetMemory].data = thread.value;
                        }
                    } else {
                        thread.state = 'idle';
                        thread.computeProgress = 0;
                        thread.value *= 0.9; // Gradually decay
                    }
                });
            });
            
            // Decay shared memory access indicators
            block.sharedMemory.forEach(sm => {
                sm.accessed *= 0.85;
            });
            
            // More dynamic active warp switching
            block.activeWarp = Math.floor(localCycle / 85) % block.warps.length;
        });
        
        // Decay global memory access indicators
        this.globalMemory.cells.forEach(cell => {
            cell.accessed *= 0.88;
        });
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        // Draw title
        this.ctx.fillStyle = '#333';
        this.ctx.font = 'bold 18px sans-serif';
        this.ctx.fillText('GPU Architecture: CUDA Kernel Execution', w * 0.05, h * 0.08);
        
        // Draw phase indicator
        this.ctx.font = '14px sans-serif';
        const phaseColors = {
            idle: '#999',
            load: '#3498db',
            compute: '#e74c3c',
            store: '#2ecc71'
        };
        this.ctx.fillStyle = phaseColors[this.currentPhase];
        this.ctx.fillText(`Phase: ${this.currentPhase.toUpperCase()}`, w * 0.05, h * 0.11);
        
        // Draw SM blocks
        this.blocks.forEach((block) => {
            // Block outline with phase color
            const phaseColor = phaseColors[block.phase];
            this.ctx.strokeStyle = phaseColor;
            this.ctx.lineWidth = 3;
            this.ctx.globalAlpha = block.phase === 'idle' ? 0.3 : 0.8;
            this.ctx.strokeRect(block.x, block.y, block.width, block.height);
            
            // Block background
            this.ctx.fillStyle = `${phaseColor}15`;
            this.ctx.globalAlpha = 1;
            this.ctx.fillRect(block.x, block.y, block.width, block.height);
            
            // Block label - more descriptive names
            const smNames = ['Tensor Core 0', 'CUDA Core A', 'CUDA Core B', 
                           'RT Core 0', 'Tensor Core 1', 'CUDA Core C'];
            this.ctx.fillStyle = '#333';
            this.ctx.font = 'bold 11px sans-serif';
            this.ctx.fillText(smNames[block.id] || `SM ${block.id}`, block.x + 5, block.y + 12);
            
            // Draw warps and threads
            block.warps.forEach((warp, wIdx) => {
                const warpActive = warp.active;
                
                // Warp indicator
                if (warpActive) {
                    this.ctx.strokeStyle = block.activeWarp === wIdx ? '#f39c12' : '#95a5a6';
                    this.ctx.lineWidth = block.activeWarp === wIdx ? 2 : 1;
                    this.ctx.globalAlpha = 0.5;
                    const warpBounds = {
                        x: warp.threads[0].x - 2,
                        y: warp.threads[0].y - 2,
                        width: (warp.threads[7].x - warp.threads[0].x) + warp.threads[0].size + 4,
                        height: (warp.threads[24].y - warp.threads[0].y) + warp.threads[0].size + 4
                    };
                    this.ctx.strokeRect(warpBounds.x, warpBounds.y, warpBounds.width, warpBounds.height);
                }
                
                // Draw threads
                warp.threads.forEach(thread => {
                    this.ctx.globalAlpha = 1;
                    
                    // Thread color based on state
                    let color;
                    let intensity = 0.5;
                    
                    switch (thread.state) {
                        case 'loading':
                            color = '#3498db';
                            intensity = 0.7 + Math.sin(this.animationTime * 0.5) * 0.3;
                            break;
                        case 'computing':
                            color = '#e74c3c';
                            // Pulsing effect during compute to show activity
                            const computePulse = Math.sin(this.animationTime * 0.4 + thread.id * 0.3);
                            intensity = 0.6 + computePulse * 0.4;
                            break;
                        case 'storing':
                            color = '#2ecc71';
                            intensity = 0.7 + Math.sin(this.animationTime * 0.5) * 0.3;
                            break;
                        default:
                            color = '#95a5a6';
                            intensity = 0.3;
                    }
                    
                    // Thread square
                    this.ctx.fillStyle = color;
                    this.ctx.globalAlpha = intensity;
                    this.ctx.fillRect(thread.x, thread.y, thread.size, thread.size);
                    
                    // Thread outline
                    this.ctx.strokeStyle = color;
                    this.ctx.globalAlpha = 1;
                    this.ctx.lineWidth = 1;
                    this.ctx.strokeRect(thread.x, thread.y, thread.size, thread.size);
                    
                    // Draw data flow lines during load/store
                    if (thread.state === 'loading' || thread.state === 'storing') {
                        const progress = Math.sin(this.animationTime * 0.2) * 0.5 + 0.5;
                        const memCell = this.globalMemory.cells[thread.targetMemory];
                        
                        const fromX = thread.state === 'loading' ? memCell.x : thread.x + thread.size;
                        const fromY = thread.state === 'loading' ? memCell.y + memCell.height / 2 : thread.y + thread.size / 2;
                        const toX = thread.state === 'loading' ? thread.x + thread.size : memCell.x;
                        const toY = thread.state === 'loading' ? thread.y + thread.size / 2 : memCell.y + memCell.height / 2;
                        
                        const currentX = fromX + (toX - fromX) * progress;
                        const currentY = fromY + (toY - fromY) * progress;
                        
                        this.ctx.strokeStyle = color;
                        this.ctx.globalAlpha = 0.3;
                        this.ctx.lineWidth = 1;
                        this.ctx.setLineDash([2, 2]);
                        this.ctx.beginPath();
                        this.ctx.moveTo(fromX, fromY);
                        this.ctx.lineTo(toX, toY);
                        this.ctx.stroke();
                        this.ctx.setLineDash([]);
                        
                        // Animated dot
                        this.ctx.fillStyle = color;
                        this.ctx.globalAlpha = 0.8;
                        this.ctx.beginPath();
                        this.ctx.arc(currentX, currentY, 2, 0, Math.PI * 2);
                        this.ctx.fill();
                    }
                });
                
                this.ctx.globalAlpha = 1;
            });
        });
        
        // Draw Global Memory
        this.ctx.strokeStyle = '#7f8c8d';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(
            this.globalMemory.x,
            this.globalMemory.y - 20,
            this.globalMemory.width,
            this.globalMemory.height + 20
        );
        
        this.ctx.fillStyle = '#333';
        this.ctx.font = 'bold 12px sans-serif';
        this.ctx.fillText(
            this.globalMemory.label,
            this.globalMemory.x + 5,
            this.globalMemory.y - 5
        );
        
        // Draw memory cells
        this.globalMemory.cells.forEach(cell => {
            // Cell background
            const accessIntensity = cell.accessed;
            this.ctx.fillStyle = accessIntensity > 0.5 
                ? `rgba(52, 152, 219, ${accessIntensity * 0.3})`
                : 'rgba(200, 200, 200, 0.1)';
            this.ctx.fillRect(cell.x, cell.y, cell.width, cell.height);
            
            // Cell border
            this.ctx.strokeStyle = accessIntensity > 0.5 ? '#3498db' : '#bdc3c7';
            this.ctx.lineWidth = accessIntensity > 0.5 ? 2 : 1;
            this.ctx.strokeRect(cell.x, cell.y, cell.width, cell.height);
            
            // Cell data value
            this.ctx.fillStyle = '#333';
            this.ctx.font = '9px monospace';
            this.ctx.fillText(
                Math.floor(cell.data).toString().padStart(2, '0'),
                cell.x + 5,
                cell.y + cell.height / 2 + 3
            );
            
            // Access indicator
            if (accessIntensity > 0.5) {
                this.ctx.fillStyle = '#e74c3c';
                this.ctx.globalAlpha = accessIntensity;
                this.ctx.beginPath();
                this.ctx.arc(cell.x + cell.width - 8, cell.y + 8, 3, 0, Math.PI * 2);
                this.ctx.fill();
                this.ctx.globalAlpha = 1;
            }
        });
        
        // Legend
        const legendX = w * 0.05;
        const legendY = h * 0.88;
        const legendItems = [
            { label: 'Loading', color: '#3498db' },
            { label: 'Computing', color: '#e74c3c' },
            { label: 'Storing', color: '#2ecc71' },
            { label: 'Idle', color: '#95a5a6' }
        ];
        
        this.ctx.font = '11px sans-serif';
        legendItems.forEach((item, i) => {
            const x = legendX + i * 100;
            this.ctx.fillStyle = item.color;
            this.ctx.fillRect(x, legendY, 15, 15);
            this.ctx.strokeStyle = '#333';
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(x, legendY, 15, 15);
            this.ctx.fillStyle = '#333';
            this.ctx.fillText(item.label, x + 20, legendY + 11);
        });
        
        this.ctx.globalAlpha = 1;
    }
}

function initCUDA() {
    const canvas = document.getElementById('cuda-canvas');
    if (!canvas) return;
    
    if (!cudaViz) {
        cudaViz = new CUDAVisualizer(canvas);
    } else {
        cudaViz.resize();
    }
    
    if (!cudaInitialized) {
        cudaInitialized = true;
        
        document.getElementById('cuda-start').addEventListener('click', () => {
            // Ensure canvas is visible while animation runs (desktop only)
            scrollIntoViewIfDesktop('cuda-canvas');
            state.cuda.running = !state.cuda.running;
            document.getElementById('cuda-start').textContent = 
                state.cuda.running ? 'Pause Kernel' : 'Launch Kernel';
            if (state.cuda.running) animateCUDA();
        });
        
        document.getElementById('cuda-reset').addEventListener('click', () => {
            resetCUDA();
        });
        
        document.getElementById('cuda-speed').addEventListener('input', (e) => {
            state.cuda.speed = parseInt(e.target.value);
        });
    }
    
    cudaViz.draw();
}

function animateCUDA() {
    if (!state.cuda.running || state.currentDemo !== 'cuda') return;
    
    cudaViz.update();
    cudaViz.draw();
    
    setTimeout(() => {
        requestAnimationFrame(animateCUDA);
    }, 1000 / state.cuda.speed);
}

// @license-end
