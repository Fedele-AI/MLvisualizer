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
        speed: 3,
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
                if (perceptronViz) perceptronViz.resize();
                else initPerceptron();
                break;
            case 'rbm':
                if (rbmViz) rbmViz.resize();
                else initRBM();
                break;
            case 'autoencoder':
                if (aencoderViz) aencoderViz.resize();
                else initAutoencoder();
                break;
            case 'ising':
                if (isingViz) isingViz.resize();
                else initIsing();
                break;
            case 'hopfield':
                if (hopfieldViz) hopfieldViz.resize();
                else initHopfield();
                break;
            case 'transformer':
                if (transformerViz) transformerViz.resize();
                else initTransformer();
                break;
            case 'deep-perceptron':
                if (deepPerceptronViz) deepPerceptronViz.resize();
                else initDeepPerceptron();
                break;
            case 'normalizing-flow':
                if (normalizingFlowViz) normalizingFlowViz.resize();
                else initNormalizingFlow();
                break;
            case 'vae':
                if (vaeViz) vaeViz.resize();
                else initVAE();
                break;
            case 'cnn-encoder-decoder':
                if (cnnViz) cnnViz.resize();
                else initCNNEncoderDecoder();
                break;
            case 'mamba2':
                if (mamba2Viz) mamba2Viz.resize();
                else initMamba2();
                break;
            case 'cuda':
                if (cudaViz) cudaViz.resize();
                else initCUDA();
                break;
        }
    }, 10);
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
            state.perceptron.running = !state.perceptron.running;
            document.getElementById('perceptron-start').textContent = 
                state.perceptron.running ? 'Pause Training' : 'Start Training';
            if (state.perceptron.running) animatePerceptron();
        });
        
        document.getElementById('perceptron-reset').addEventListener('click', () => {
            state.perceptron.running = false;
            state.perceptron.epoch = 0;
            perceptronViz.currentSample = 0;
            perceptronViz.setupNodes();
            perceptronViz.generateTrainingData();
            perceptronViz.draw();
            document.getElementById('perceptron-start').textContent = 'Start Training';
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
            state.rbm.running = false;
            state.rbm.phase = 'idle';
            state.rbm.animationProgress = 0;
            document.getElementById('rbm-start').textContent = 'Start Training';
            rbmViz.setupNodes();
            rbmViz.particles = [];
            rbmViz.draw();
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
            state.autoencoder.running = false;
            state.autoencoder.phase = 'idle';
            state.autoencoder.animationProgress = 0;
            document.getElementById('ae-start').textContent = 'Start Encoding';
            aencoderViz.setupLayers();
            aencoderViz.particles = [];
            aencoderViz.draw();
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
            state.ising.running = !state.ising.running;
            document.getElementById('ising-start').textContent = 
                state.ising.running ? 'Pause Simulation' : 'Start Simulation';
            if (state.ising.running) animateIsing();
        });
        
        document.getElementById('ising-reset').addEventListener('click', () => {
            state.ising.running = false;
            isingViz.initializeSpins();
            isingViz.draw();
            document.getElementById('ising-start').textContent = 'Start Simulation';
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
            hopfieldViz.storePatterns();
            hopfieldViz.draw();
        });
        
        document.getElementById('hopfield-recall').addEventListener('click', () => {
            const noiseLevel = parseInt(document.getElementById('hopfield-noise').value);
            hopfieldViz.addNoise(noiseLevel);
            setTimeout(() => hopfieldViz.recall(), 500);
        });
        
        document.getElementById('hopfield-reset').addEventListener('click', () => {
            hopfieldViz.initializeState();
            hopfieldViz.patterns = [];
            hopfieldViz.weights = [];
            hopfieldViz.isRecalling = false;
            hopfieldViz.draw();
            document.getElementById('hopfield-patterns').textContent = '0';
        });
    }
    
    hopfieldViz.draw();
}

// ====== Transformer Visualization ======
let transformerViz = null;
let transformerInitialized = false;

class TransformerVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.tokens = ['The', 'cat', 'sat', 'on', 'mat'];
        this.layers = [];
        this.attentionWeights = [];
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
        const layerCount = 5; // Input, Attention, Context, FFN, Output
        const tokenCount = this.tokens.length;
        const layerColors = ['#388e3c', '#f57f17', '#e65100', '#f57f17', '#388e3c'];
        // Responsive node radius and padding
        const nodeRadius = Math.max(10, Math.min(15, this.canvas.width / 50));
        const padding = Math.max(40, Math.min(80, this.canvas.width / 10));
        
        this.layers = [];
        const layerSpacing = (this.canvas.width - 2 * padding) / (layerCount - 1);
        
        for (let l = 0; l < layerCount; l++) {
            const nodes = [];
            const x = padding + l * layerSpacing;
            const verticalSpacing = (this.canvas.height - 100) / tokenCount;
            
            for (let t = 0; t < tokenCount; t++) {
                nodes.push({
                    x: x,
                    y: 50 + t * verticalSpacing + verticalSpacing / 2,
                    radius: nodeRadius,
                    activation: 0,
                    color: layerColors[l],
                    token: this.tokens[t]
                });
            }
            
            this.layers.push(nodes);
        }
        
        // Initialize attention weights
        this.attentionWeights = [];
        for (let i = 0; i < tokenCount; i++) {
            this.attentionWeights[i] = [];
            for (let j = 0; j < tokenCount; j++) {
                this.attentionWeights[i][j] = Math.random();
            }
        }
    }
    
    update() {
        if (!state.transformer.running) return;
        
        // Simulate processing through layers
        const currentLayer = Math.floor(Date.now() / 500) % this.layers.length;
        
        this.layers.forEach((layer, idx) => {
            layer.forEach(node => {
                if (idx === currentLayer) {
                    node.activation = 0.8 + Math.random() * 0.2;
                } else if (idx < currentLayer) {
                    node.activation = 0.5;
                } else {
                    node.activation = 0.1;
                }
            });
        });
        
        // Create particles
        if (Math.random() < 0.15) {
            const fromLayer = this.layers[Math.max(0, currentLayer - 1)];
            const toLayer = this.layers[currentLayer];
            
            if (fromLayer && toLayer) {
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
        
        // Update particles
        this.particles = this.particles.filter(p => {
            p.progress += 0.02;
            p.life -= 0.01;
            return p.life > 0 && p.progress < 1;
        });
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw attention connections (light lines between attention layer tokens)
        if (this.layers.length > 1) {
            const attentionLayer = this.layers[1];
            for (let i = 0; i < attentionLayer.length; i++) {
                for (let j = 0; j < attentionLayer.length; j++) {
                    if (i !== j) {
                        const alpha = this.attentionWeights[i][j] * 0.2;
                        this.ctx.strokeStyle = `rgba(247, 127, 23, ${alpha})`;
                        this.ctx.lineWidth = this.attentionWeights[i][j] * 2;
                        this.ctx.beginPath();
                        this.ctx.moveTo(attentionLayer[i].x, attentionLayer[i].y);
                        this.ctx.lineTo(attentionLayer[j].x, attentionLayer[j].y);
                        this.ctx.stroke();
                    }
                }
            }
        }
        
        // Draw layer connections
        for (let i = 0; i < this.layers.length - 1; i++) {
            const layer = this.layers[i];
            const nextLayer = this.layers[i + 1];
            
            layer.forEach(node1 => {
                nextLayer.forEach(node2 => {
                    this.ctx.strokeStyle = 'rgba(200, 200, 200, 0.15)';
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
            
            const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, 5);
            gradient.addColorStop(0, `rgba(102, 126, 234, ${p.life})`);
            gradient.addColorStop(1, `rgba(102, 126, 234, 0)`);
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 5, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        // Draw nodes with token labels
        this.layers.forEach((layer, layerIdx) => {
            layer.forEach((node, tokenIdx) => {
                this.drawNode(node);
                
                // Show token label on first layer
                if (layerIdx === 0) {
                    this.ctx.fillStyle = '#333';
                    this.ctx.font = 'bold 12px sans-serif';
                    this.ctx.textAlign = 'right';
                    this.ctx.fillText(node.token, node.x - node.radius - 10, node.y + 4);
                }
            });
        });
    }
    
    drawNode(node) {
        // Glow
        const glowRadius = node.radius + 6;
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
        
        // Border
        this.ctx.globalAlpha = 1;
        this.ctx.strokeStyle = node.color;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        this.ctx.stroke();
    }
}

function initTransformer() {
    const canvas = document.getElementById('transformer-canvas');
    if (!canvas) return;
    
    if (!transformerViz) {
        transformerViz = new TransformerVisualizer(canvas);
    } else {
        transformerViz.resize();
    }
    
    if (!transformerInitialized) {
        transformerInitialized = true;
        
        document.getElementById('transformer-start').addEventListener('click', () => {
            state.transformer.running = !state.transformer.running;
            document.getElementById('transformer-start').textContent = 
                state.transformer.running ? 'Pause Processing' : 'Process Sequence';
            if (state.transformer.running) animateTransformer();
        });
        
        document.getElementById('transformer-reset').addEventListener('click', () => {
            state.transformer.running = false;
            transformerViz.setupLayers();
            transformerViz.particles = [];
            transformerViz.draw();
            document.getElementById('transformer-start').textContent = 'Process Sequence';
        });
        
        document.getElementById('transformer-speed').addEventListener('input', (e) => {
            state.transformer.speed = parseInt(e.target.value);
        });
    }
    
    transformerViz.draw();
}

function animateTransformer() {
    if (!state.transformer.running || state.currentDemo !== 'transformer') return;
    
    transformerViz.update();
    transformerViz.draw();
    
    setTimeout(() => {
        requestAnimationFrame(animateTransformer);
    }, 100);
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
        const layerSizes = [3, 5, 4, 3, 2];
        this.layers = [];
        
        // Responsive node radius based on canvas size
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
                    color: layerIdx === 0 ? '#4A90E2' : 
                           layerIdx === layerSizes.length - 1 ? '#50E3C2' : '#9B59B6'
                });
            }
            this.layers.push(layer);
        });
    }
    
    update() {
        // Decay activations smoothly
        this.layers.forEach((layer, idx) => {
            layer.forEach(node => {
                node.activation *= 0.95; // Smooth decay
            });
        });
        
        // Spawn particles
        if (Math.random() < 0.3) {
            const sourceLayer = Math.floor(Math.random() * (this.layers.length - 1));
            const sourceNode = this.layers[sourceLayer][Math.floor(Math.random() * this.layers[sourceLayer].length)];
            const targetNode = this.layers[sourceLayer + 1][Math.floor(Math.random() * this.layers[sourceLayer + 1].length)];
            
            // Activate source node
            sourceNode.activation = 1.0;
            
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
        
        // Update particles
        this.particles = this.particles.filter(p => {
            p.progress += 0.05;
            p.life -= 0.02;
            
            // Activate target node when particle arrives
            if (p.progress >= 0.9 && p.targetNode) {
                p.targetNode.activation = 1.0;
                p.targetNode = null; // Only activate once
            }
            
            return p.life > 0 && p.progress < 1;
        });
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw connections
        for (let i = 0; i < this.layers.length - 1; i++) {
            this.layers[i].forEach(node1 => {
                this.layers[i + 1].forEach(node2 => {
                    this.ctx.strokeStyle = 'rgba(200, 200, 200, 0.15)';
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
            
            const gradient = this.ctx.createRadialGradient(x, y, 0, x, y, 5);
            gradient.addColorStop(0, `rgba(102, 126, 234, ${p.life})`);
            gradient.addColorStop(1, `rgba(102, 126, 234, 0)`);
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 5, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        // Draw nodes
        this.layers.forEach(layer => {
            layer.forEach(node => {
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

function initDeepPerceptron() {
    const canvas = document.getElementById('deep-perceptron-canvas');
    if (!canvas) return;
    
    if (!deepPerceptronViz) {
        deepPerceptronViz = new DeepPerceptronVisualizer(canvas);
    } else {
        deepPerceptronViz.resize();
    }
    
    if (!deepPerceptronInitialized) {
        deepPerceptronInitialized = true;
        
        document.getElementById('deep-perceptron-start').addEventListener('click', () => {
            state.deepPerceptron.running = !state.deepPerceptron.running;
            document.getElementById('deep-perceptron-start').textContent = 
                state.deepPerceptron.running ? 'Pause Training' : 'Start Training';
            if (state.deepPerceptron.running) animateDeepPerceptron();
        });
        
        document.getElementById('deep-perceptron-reset').addEventListener('click', () => {
            state.deepPerceptron.running = false;
            deepPerceptronViz.setupLayers();
            deepPerceptronViz.particles = [];
            deepPerceptronViz.draw();
            document.getElementById('deep-perceptron-start').textContent = 'Start Training';
        });
        
        document.getElementById('deep-perceptron-speed').addEventListener('input', (e) => {
            state.deepPerceptron.speed = parseInt(e.target.value);
        });
    }
    
    deepPerceptronViz.draw();
}

function animateDeepPerceptron() {
    if (!state.deepPerceptron.running || state.currentDemo !== 'deep-perceptron') return;
    
    deepPerceptronViz.update();
    deepPerceptronViz.draw();
    
    setTimeout(() => {
        requestAnimationFrame(animateDeepPerceptron);
    }, 1000 / state.deepPerceptron.speed);
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
        const layerSizes = [8, 6, 6, 6, 8];
        this.layers = [];
        
        // Responsive node radius
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
                    color: layerIdx === 0 ? '#667EEA' :      // Base Z - Blue
                           layerIdx === 1 ? '#9B59B6' :      // Flow 1 - Purple
                           layerIdx === 2 ? '#E056FD' :      // Flow 2 - Pink/Magenta
                           layerIdx === 3 ? '#3498DB' :      // Flow 3 - Cyan/Blue
                           '#F77F17'                         // Data X - Orange
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
                    this.ctx.strokeStyle = 'rgba(200, 200, 200, 0.1)';
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
            gradient.addColorStop(0, `rgba(155, 89, 182, ${p.life})`);
            gradient.addColorStop(1, `rgba(155, 89, 182, 0)`);
            
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(x, y, 6, 0, Math.PI * 2);
            this.ctx.fill();
        });
        
        // Draw nodes with glow effect
        this.layers.forEach(layer => {
            layer.forEach(node => {
                // Inner glow
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
            state.normalizingFlow.running = !state.normalizingFlow.running;
            document.getElementById('nf-start').textContent = 
                state.normalizingFlow.running ? 'Pause Flow' : 'Start Flow';
            if (state.normalizingFlow.running) animateNormalizingFlow();
        });
        
        document.getElementById('nf-reset').addEventListener('click', () => {
            state.normalizingFlow.running = false;
            normalizingFlowViz.setupLayers();
            normalizingFlowViz.particles = [];
            normalizingFlowViz.draw();
            document.getElementById('nf-start').textContent = 'Start Flow';
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
    
    setTimeout(() => {
        requestAnimationFrame(animateNormalizingFlow);
    }, 1000 / state.normalizingFlow.speed);
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
            state.vae.running = !state.vae.running;
            document.getElementById('vae-start').textContent = 
                state.vae.running ? 'Pause Encoding' : 'Start Encoding';
            if (state.vae.running) animateVAE();
        });
        
        document.getElementById('vae-reset').addEventListener('click', () => {
            state.vae.running = false;
            vaeViz.setupLayers();
            vaeViz.particles = [];
            vaeViz.draw();
            document.getElementById('vae-start').textContent = 'Start Encoding';
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
            state.cnnEncoderDecoder.running = !state.cnnEncoderDecoder.running;
            document.getElementById('cnn-start').textContent = 
                state.cnnEncoderDecoder.running ? 'Pause Processing' : 'Process Image';
            if (state.cnnEncoderDecoder.running) animateCNN();
        });
        
        document.getElementById('cnn-reset').addEventListener('click', () => {
            state.cnnEncoderDecoder.running = false;
            cnnViz.setupMaps();
            cnnViz.particles = [];
            cnnViz.draw();
            document.getElementById('cnn-start').textContent = 'Process Image';
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
            state.mamba2.running = !state.mamba2.running;
            document.getElementById('mamba2-start').textContent = 
                state.mamba2.running ? 'Pause Processing' : 'Process Sequence';
            if (state.mamba2.running) animateMamba2();
        });
        
        document.getElementById('mamba2-reset').addEventListener('click', () => {
            state.mamba2.running = false;
            mamba2Viz.setupLayers();
            mamba2Viz.particles = [];
            mamba2Viz.draw();
            document.getElementById('mamba2-start').textContent = 'Process Sequence';
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
    
    setTimeout(() => {
        requestAnimationFrame(animateMamba2);
    }, 1000 / state.mamba2.speed);
}

// ====== CUDA Visualization ======
let cudaViz = null;
let cudaInitialized = false;

class CUDAVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.grid = [];
        this.blocks = [];
        this.threads = [];
        this.activeThreads = [];
        this.resize();
        this.setupGrid();
    }
    
    resize() {
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
        this.setupGrid();
    }
    
    setupGrid() {
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        // Grid structure
        this.grid = {
            x: w * 0.1,
            y: h * 0.1,
            width: w * 0.8,
            height: h * 0.8
        };
        
        // Blocks (4x2 grid of blocks)
        this.blocks = [];
        const blockCols = 4;
        const blockRows = 2;
        const blockWidth = this.grid.width / blockCols;
        const blockHeight = this.grid.height / blockRows;
        
        for (let row = 0; row < blockRows; row++) {
            for (let col = 0; col < blockCols; col++) {
                const block = {
                    x: this.grid.x + col * blockWidth,
                    y: this.grid.y + row * blockHeight,
                    width: blockWidth - 10,
                    height: blockHeight - 10,
                    threads: [],
                    active: 0
                };
                
                // Threads per block (4x4 grid)
                const threadCols = 4;
                const threadRows = 4;
                const threadWidth = block.width / threadCols;
                const threadHeight = block.height / threadRows;
                
                for (let tRow = 0; tRow < threadRows; tRow++) {
                    for (let tCol = 0; tCol < threadCols; tCol++) {
                        block.threads.push({
                            x: block.x + tCol * threadWidth + threadWidth / 2,
                            y: block.y + tRow * threadHeight + threadHeight / 2,
                            radius: 5,
                            active: 0
                        });
                    }
                }
                
                this.blocks.push(block);
            }
        }
    }
    
    update() {
        // Randomly activate blocks and threads
        this.blocks.forEach(block => {
            block.active = Math.random() < 0.3 ? 1 : block.active * 0.9;
            block.threads.forEach(thread => {
                if (Math.random() < 0.1) {
                    thread.active = 1;
                } else {
                    thread.active *= 0.85;
                }
            });
        });
    }
    
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid outline
        this.ctx.strokeStyle = '#667EEA';
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(this.grid.x, this.grid.y, this.grid.width, this.grid.height);
        
        // Draw blocks
        this.blocks.forEach(block => {
            // Block outline
            this.ctx.strokeStyle = '#9B59B6';
            this.ctx.lineWidth = 2;
            this.ctx.globalAlpha = 0.3 + block.active * 0.7;
            this.ctx.strokeRect(block.x, block.y, block.width, block.height);
            
            // Block fill
            this.ctx.fillStyle = 'rgba(155, 89, 182, 0.1)';
            this.ctx.fillRect(block.x, block.y, block.width, block.height);
            
            // Draw threads
            block.threads.forEach(thread => {
                const intensity = thread.active;
                
                // Thread glow
                if (intensity > 0.3) {
                    const gradient = this.ctx.createRadialGradient(
                        thread.x, thread.y, 0,
                        thread.x, thread.y, thread.radius * 3
                    );
                    gradient.addColorStop(0, `rgba(80, 227, 194, ${intensity * 0.6})`);
                    gradient.addColorStop(1, 'rgba(80, 227, 194, 0)');
                    
                    this.ctx.fillStyle = gradient;
                    this.ctx.beginPath();
                    this.ctx.arc(thread.x, thread.y, thread.radius * 3, 0, Math.PI * 2);
                    this.ctx.fill();
                }
                
                // Thread core
                this.ctx.fillStyle = `rgba(80, 227, 194, ${0.3 + intensity * 0.7})`;
                this.ctx.globalAlpha = 1;
                this.ctx.beginPath();
                this.ctx.arc(thread.x, thread.y, thread.radius, 0, Math.PI * 2);
                this.ctx.fill();
                
                this.ctx.strokeStyle = '#50E3C2';
                this.ctx.lineWidth = 1;
                this.ctx.stroke();
            });
        });
        
        // Labels
        this.ctx.globalAlpha = 1;
        this.ctx.fillStyle = '#333';
        this.ctx.font = 'bold 14px sans-serif';
        this.ctx.fillText('Grid of Blocks', this.grid.x, this.grid.y - 10);
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
            state.cuda.running = !state.cuda.running;
            document.getElementById('cuda-start').textContent = 
                state.cuda.running ? 'Pause Kernel' : 'Launch Kernel';
            if (state.cuda.running) animateCUDA();
        });
        
        document.getElementById('cuda-reset').addEventListener('click', () => {
            state.cuda.running = false;
            cudaViz.setupGrid();
            cudaViz.draw();
            document.getElementById('cuda-start').textContent = 'Launch Kernel';
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
