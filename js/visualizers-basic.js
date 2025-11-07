// @license magnet:?xt=urn:btih:1f739d935676111cfff4b4693e3816e664797050&dn=gpl-3.0.txt GPL-3.0
/**
 * ML Visualizer - Basic neural network visualizers
 * Copyright (C) 2025 Kenneth (Alex) Jenkins
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

// This file contains: Perceptron, RBM, Autoencoder, Ising Model, and Hopfield Network visualizers

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
        
        this.resize(true);
        window.addEventListener('resize', debounce(() => this.resize(), 250));
    }
    
    resize(force = false) {
        if (!force && !shouldHandleResize()) return;
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
        perceptronViz.resize(true);
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
        
        this.resize(true);
        window.addEventListener('resize', debounce(() => this.resize(), 250));
    }
    
    resize(force = false) {
        if (!force && !shouldHandleResize()) return;
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
    
    update(dtFrames = 1, speed = (typeof state !== 'undefined' && state.rbm ? state.rbm.speed : 3)) {
        if (!state.rbm.running) return;
        
        const speedFactor = Math.max(0.25, speed / 3);
        // Time-scaled phase progress
        state.rbm.animationProgress += 0.01 * dtFrames * speedFactor;
        
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
        
        // Create particles (time-scaled probability)
        if (Math.random() < 0.3 * Math.min(3, dtFrames) * speedFactor) {
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
        
        // Update particles - slower movement for better visibility (time-scaled)
        this.particles = this.particles.filter(p => {
            p.progress += 0.06 * dtFrames * speedFactor;
            p.life -= 0.015 * dtFrames;
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
        rbmViz.resize(true);
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

let rbmLastTime = 0;
function animateRBM(currentTime = 0) {
    if (!state.rbm.running || state.currentDemo !== 'rbm') return;
    const frameMs = 1000 / 60;
    let dtFrames = 1;
    if (rbmLastTime) {
        dtFrames = Math.max(0.5, Math.min(3, (currentTime - rbmLastTime) / frameMs));
    }
    rbmLastTime = currentTime;
    rbmViz.update(dtFrames, state.rbm.speed);
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
        
        this.resize(true);
        window.addEventListener('resize', debounce(() => this.resize(), 250));
    }
    
    resize(force = false) {
        if (!force && !shouldHandleResize()) return;
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
        aencoderViz.resize(true);
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
        
        this.resize(true);
        window.addEventListener('resize', debounce(() => this.resize(), 250));
    }
    
    resize(force = false) {
        if (!force && !shouldHandleResize()) return;
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
        isingViz.resize(true);
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
        this.dataFlowTimer = 0; // measured in 60fps frames
        this.isRecalling = false;
        
        this.resize(true);
        window.addEventListener('resize', debounce(() => this.resize(), 250));
        this.setupMouseInteraction();
    }
    
    resize(force = false) {
        if (!force && !shouldHandleResize()) return;
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
        hopfieldViz.resize(true);
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

// @license-end
