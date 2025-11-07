// @license magnet:?xt=urn:btih:1f739d935676111cfff4b4693e3816e664797050&dn=gpl-3.0.txt GPL-3.0
/**
 * ML Visualizer - Core utilities and navigation
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

// Mobile viewport-aware resize guard to prevent thrashing on scroll (address bar hide/show)
let _lastViewport = { w: 0, h: 0 };
function isMobileViewport() {
    try {
        return (window.matchMedia && window.matchMedia('(max-width: 1024px)').matches) || (navigator.maxTouchPoints && navigator.maxTouchPoints > 1);
    } catch (_) { return true; }
}

function shouldHandleResize() {
    const w = window.innerWidth || document.documentElement.clientWidth;
    const h = window.innerHeight || document.documentElement.clientHeight;

    // Always allow the very first call
    if (_lastViewport.w === 0 && _lastViewport.h === 0) {
        _lastViewport = { w, h };
        return true;
    }

    // On desktop, always handle
    if (!isMobileViewport()) {
        _lastViewport = { w, h };
        return true;
    }

    // On mobile, ignore height-only fluctuations from browser UI unless large
    const widthChanged = Math.abs(w - _lastViewport.w) > 4;
    const heightChangedSignificant = Math.abs(h - _lastViewport.h) > 120;
    const orientationChanged = (w > h) !== (_lastViewport.w > _lastViewport.h);

    if (widthChanged || heightChangedSignificant || orientationChanged) {
        _lastViewport = { w, h };
        return true;
    }
    return false;
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
        phase: 'idle',
        eosCount: 0,  // Track consecutive </s> tokens
        autoGenerating: false  // Track if we're in auto-generation mode
    },
    deepPerceptron: {
        running: false,
        speed: 5,
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
    
    // Scroll to top for all modules except neural music
    if (demoName !== 'neural-music') {
        window.scrollTo({ top: 0, left: 0, behavior: 'auto' });
    }
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

// @license-end
