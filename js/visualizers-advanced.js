// @license magnet:?xt=urn:btih:1f739d935676111cfff4b4693e3816e664797050&dn=gpl-3.0.txt GPL-3.0
/**
 * ML Visualizer - Advanced neural network visualizers
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

// This file contains: Transformer, DeepPerceptron, NormalizingFlow, VAE, CNN, Mamba2, CUDA visualizers

let transformerViz = null;
let transformerInitialized = false;
let toyLMBuilt = false;

// Enhanced toy language model (unigram + bigram + trigram) for next-token prediction
const toyLM = {
    unigram: new Map(),
    bigram: new Map(),
    trigram: new Map(),
    vocab: new Set(),
    total: 0,
    sos: '<s>',
    eos: '</s>'
};

function tokenize(text) {
    if (!text) return [];
    // Improved tokenization to preserve contractions and handle more cases
    const matches = text.toLowerCase().match(/[\w']+|[.,!?;:]/g);
    return matches ? matches : [];
}

// Get random starter phrases for auto-generation
function getRandomStarter() {
    const starters = [
        'i like',
        'i love',
        'i think',
        'the cat',
        'the dog',
        'my friend',
        'what is',
        'this is',
        'machine learning',
        'neural networks',
        'the sky',
        'pizza is',
        'music is',
        'today is',
        'i want'
    ];
    return starters[Math.floor(Math.random() * starters.length)];
}

function buildToyLanguageModel() {
    if (toyLMBuilt) return;
    // Significantly expanded corpus with diverse topics and natural patterns
    const corpus = [
        // Food and preferences
        'i like pizza that has cheese',
        'i like pizza that is hot',
        'i like pizza with pepperoni',
        'i love pizza that tastes great',
        'pizza is my favorite food',
        'pizza that has mushrooms is good',
        'i like burgers that are juicy',
        'i like pasta that is fresh',
        'i love food that tastes good',
        'i eat pizza every week',
        'my favorite food is pizza',
        'my favorite drink is coffee',
        'my favorite color is blue',
        'your favorite movie is interesting',
        'their favorite song is popular',
        
        // Personal relationships and feelings
        'my best friend is cool',
        'my best friend is awesome',
        'my best friend lives nearby',
        'my best day was yesterday',
        'my best work is here',
        'your best effort is appreciated',
        'his best friend is loyal',
        'her best friend is kind',
        'our best hope is tomorrow',
        'their best team won today',
        'my good friend is smart',
        'my close friend is helpful',
        'my old friend is wise',
        'i have a friend who helps',
        'i know someone who cares',
        'my family is wonderful',
        'my family loves me dearly',
        'my mom is the best',
        'my dad is very kind',
        'my sister is really funny',
        'my brother is quite tall',
        
        // Animals and nature
        'the cat sat on the mat',
        'the dog sat on the rug',
        'the cat chased the mouse',
        'the dog chased the cat',
        'a cat likes fish that is fresh',
        'a dog likes bones that are big',
        'the cat is sleeping now',
        'the dog is running fast',
        'cats and dogs are animals',
        'the bird is flying high',
        'the sun is shining bright',
        'the moon is bright tonight',
        'the sky is blue today',
        'the grass is green and lush',
        'the trees are tall and strong',
        'my pet is very cute',
        'my dog is really playful',
        'my cat is always sleeping',
        
        // Machine learning and AI
        'attention helps pick the next token',
        'transformers predict the next token',
        'machine learning is fascinating and powerful',
        'neural networks learn patterns from data',
        'the model generates text that makes sense',
        'artificial intelligence is powerful and useful',
        'deep learning requires lots of data',
        'the algorithm processes information quickly',
        'language models understand context well',
        'machine learning models are trained carefully',
        'neural networks use backpropagation effectively',
        'the transformer uses attention mechanisms',
        'attention mechanisms are very important',
        'i love machine learning because it works',
        'i think neural networks are amazing',
        'i like transformers that generate text',
        'what is machine learning exactly',
        'what are neural networks made of',
        'how does attention work in transformers',
        'why use transformers for language',
        'my project uses deep learning',
        'my research focuses on ai',
        'my work involves neural networks',
        
        // Common phrases and continuations
        'this is amazing and wonderful',
        'this model works very well',
        'that is interesting to know',
        'that makes sense to me',
        'that has been done before',
        'that was great to see',
        'the way it works is simple',
        'the way that looks is nice',
        'i am learning about transformers',
        'i am studying machine learning',
        'i was thinking about that',
        'i was wondering if you know',
        'you are doing great work',
        'you can do that easily',
        'we are working on this',
        'we can make it better',
        'they are building new models',
        'they have done amazing work',
        'my idea is working well',
        'my plan is to succeed',
        'my goal is to learn',
        'my hope is that you understand',
        'my dream is to help',
        'my wish is for peace',
        
        // Personal experiences
        'i went to the store',
        'i went to school today',
        'i saw a movie yesterday',
        'i read a book recently',
        'i built something new today',
        'i learned something interesting today',
        'i found a great solution',
        'i discovered something amazing',
        'i created a new project',
        'i finished my work early',
        'my life is going well',
        'my day was really good',
        'my week has been busy',
        'my year was very productive',
        'my time here is valuable',
        'my experience has been positive',
        
        // Questions and statements
        'what do you think about that',
        'what can we do next',
        'where is the data stored',
        'when will it be ready',
        'how does that work exactly',
        'why is this important now',
        'who made that decision there',
        'which model is better overall',
        'what is your name exactly',
        'what is your favorite thing',
        'where do you live now',
        'when did that happen exactly',
        'how are you doing today',
        'why did you choose that',
        'who is your best friend',
        
        // Descriptive phrases
        'is very good for this',
        'is really important to know',
        'is quite interesting to see',
        'has been tested thoroughly',
        'has many uses today',
        'was created recently by experts',
        'was made to solve problems',
        'can be used for many things',
        'could be better with changes',
        'should work well for this',
        'would be nice to have',
        'will help us understand more',
        'is absolutely wonderful and great',
        'is incredibly useful for us',
        'is surprisingly effective today',
        'was unexpectedly good yesterday',
        
        // Activities and hobbies
        'i love playing video games',
        'i enjoy reading good books',
        'i like watching great movies',
        'i love listening to music',
        'i enjoy writing creative stories',
        'i like cooking delicious food',
        'i love traveling to places',
        'i enjoy learning new things',
        'my hobby is playing guitar',
        'my hobby is painting pictures',
        'my passion is helping others',
        'my interest is in science',
        
        // Emotions and opinions
        'i feel happy about that',
        'i feel excited for tomorrow',
        'i think that is correct',
        'i think you are right',
        'i believe we can succeed',
        'i believe in your ability',
        'i know this is true',
        'i understand your point clearly',
        'my opinion is that it works',
        'my feeling is that we should',
        'my sense is that things improve',
        'my belief is that people care',
        
        // Time and place
        'today is a great day',
        'today is very special indeed',
        'tomorrow will be better hopefully',
        'yesterday was quite interesting actually',
        'now is the right time',
        'here is the best place',
        'there is a good option',
        'everywhere i look i see',
        'sometimes i wonder about things',
        'always remember to be kind',
        'never give up on dreams',
        'my morning was very productive',
        'my afternoon is looking good',
        'my evening will be relaxing',
        'my night was peaceful and quiet',
        
        // School and learning
        'my teacher is very helpful',
        'my class is really interesting',
        'my school is quite large',
        'my homework is almost done',
        'my grades are improving steadily',
        'my studies are going well',
        'i study computer science daily',
        'i learn new concepts regularly',
        'the lesson was very clear',
        'the course is well designed',
        
        // Technology and computers
        'my computer is running fast',
        'my phone is very useful',
        'my laptop is brand new',
        'my code is working correctly',
        'my program runs smoothly now',
        'my app is almost finished',
        'the software is really powerful',
        'the system works efficiently today',
        'the internet is very fast',
        'the technology is quite advanced',
        
        // Transitions and connectors
        'and then we can see',
        'but that is not all',
        'or we could try something',
        'so we should continue forward',
        'because it makes sense logically',
        'when you think about it',
        'if you look at that',
        'that means we can proceed',
        'which means it works well'
    ];

    for (const line of corpus) {
        const tokens = [toyLM.sos, ...tokenize(line), toyLM.eos];
        for (let i = 0; i < tokens.length; i++) {
            const t = tokens[i];
            toyLM.vocab.add(t);
            toyLM.unigram.set(t, (toyLM.unigram.get(t) || 0) + 1);
            toyLM.total++;
            
            // Bigram
            if (i > 0) {
                const prev = tokens[i - 1];
                if (!toyLM.bigram.has(prev)) toyLM.bigram.set(prev, new Map());
                const row = toyLM.bigram.get(prev);
                row.set(t, (row.get(t) || 0) + 1);
            }
            
            // Trigram (for better context)
            if (i > 1) {
                const prev2 = tokens[i - 2];
                const prev1 = tokens[i - 1];
                const key = `${prev2}|${prev1}`;
                if (!toyLM.trigram.has(key)) toyLM.trigram.set(key, new Map());
                const row = toyLM.trigram.get(key);
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
    
    if (tokens.length === 0) {
        // Start of sequence
        const dist = new Map();
        toyLM.unigram.forEach((c, t) => {
            if (t !== toyLM.sos && t !== toyLM.eos) dist.set(t, c);
        });
        return formatPredictions(dist, temperature, topK);
    }
    
    // Try trigram first (best context)
    if (tokens.length >= 2) {
        const prev2 = tokens[tokens.length - 2];
        const prev1 = tokens[tokens.length - 1];
        const key = `${prev2}|${prev1}`;
        
        if (toyLM.trigram.has(key)) {
            const dist = new Map(toyLM.trigram.get(key));
            // If we have enough trigram data, use it
            const totalCount = Array.from(dist.values()).reduce((a, b) => a + b, 0);
            if (totalCount >= 2) {
                return formatPredictions(dist, temperature, topK);
            }
        }
    }
    
    // Fall back to bigram
    const last = tokens[tokens.length - 1];
    if (toyLM.bigram.has(last)) {
        const dist = new Map(toyLM.bigram.get(last));
        const totalCount = Array.from(dist.values()).reduce((a, b) => a + b, 0);
        if (totalCount >= 1) {
            return formatPredictions(dist, temperature, topK);
        }
    }
    
    // Final fallback: intelligent unigram with context awareness
    // Filter tokens that commonly follow the part of speech of the last token
    const dist = new Map();
    
    // Determine likely next token type based on last token
    const commonFollowers = getCommonFollowers(last);
    
    toyLM.unigram.forEach((c, t) => {
        if (t !== toyLM.sos && t !== toyLM.eos) {
            // Boost tokens that commonly follow this type of word
            let boost = 1.0;
            if (commonFollowers.has(t)) {
                boost = 3.0;
            }
            dist.set(t, c * boost);
        }
    });
    
    return formatPredictions(dist, temperature, topK);
}

function getCommonFollowers(word) {
    // Simple heuristic to determine what typically follows different word types
    const followers = new Set();
    
    const verbs = ['is', 'are', 'was', 'were', 'has', 'have', 'can', 'could', 'should', 'will', 'would', 'like', 'love', 'think', 'know', 'see', 'make', 'get', 'do', 'does'];
    const determiners = ['the', 'a', 'an', 'that', 'this', 'these', 'those'];
    const prepositions = ['of', 'to', 'in', 'on', 'at', 'for', 'with', 'about', 'that'];
    const adjectives = ['good', 'great', 'big', 'small', 'new', 'old', 'important', 'interesting', 'amazing', 'powerful'];
    const nouns = ['cat', 'dog', 'pizza', 'food', 'model', 'network', 'learning', 'data', 'work', 'way', 'time', 'thing'];
    
    if (determiners.includes(word)) {
        // After determiners, expect nouns or adjectives
        adjectives.forEach(w => followers.add(w));
        nouns.forEach(w => followers.add(w));
    } else if (adjectives.includes(word)) {
        // After adjectives, expect nouns
        nouns.forEach(w => followers.add(w));
    } else if (verbs.includes(word)) {
        // After verbs, expect nouns, adjectives, or prepositions
        nouns.forEach(w => followers.add(w));
        adjectives.forEach(w => followers.add(w));
        prepositions.forEach(w => followers.add(w));
    } else if (nouns.includes(word)) {
        // After nouns, expect verbs or prepositions
        verbs.forEach(w => followers.add(w));
        prepositions.forEach(w => followers.add(w));
    } else if (prepositions.includes(word) || word === 'that') {
        // After prepositions, expect determiners or nouns
        determiners.forEach(w => followers.add(w));
        nouns.forEach(w => followers.add(w));
    }
    
    return followers;
}

function formatPredictions(dist, temperature, topK) {
    // Convert counts to logits and apply temperature
    const entries = Array.from(dist.entries()).filter(([t, c]) => c > 0);
    
    if (entries.length === 0) {
        return [];
    }
    
    // Use log smoothing and temperature
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
        
        this.resize(true);
        this.setupLayers();
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
        const layerCount = 5; // Prompt, Context, Scores, Softmax, Next Token
        const tokenCount = this.tokens.length || 1;
        const layerColors = ['#4A90E2', '#f57f17', '#9B59B6', '#E74C3C', '#27AE60'];
        const layerNames = ['Prompt', 'Context', 'Scores', 'Softmax', 'Next'];
        
        // Responsive node radius and padding
        const nodeRadius = Math.max(10, Math.min(15, this.canvas.width / 50));
        // Increased right padding to accommodate predicted token text
        const paddingLeft = Math.max(90, Math.min(130, this.canvas.width / 10 + 30)); // Added 30px for "<start>" text spacing
        const paddingRight = Math.max(120, Math.min(180, this.canvas.width / 6));
        
        this.layers = [];
        const layerSpacing = (this.canvas.width - paddingLeft - paddingRight) / (layerCount - 1);
        
        for (let l = 0; l < layerCount; l++) {
            const nodes = [];
            const x = paddingLeft + l * layerSpacing;
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
        transformerViz.resize(true);
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
            // Stop auto-generation when user manually predicts
            const wasAutoGenerating = state.transformer.autoGenerating;
            state.transformer.autoGenerating = false;
            state.transformer.eosCount = 0;
            
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
                } else if (preds.length > 0 && preds[0].token === toyLM.eos) {
                    appendBtn.textContent = `➕ Append "${toyLM.eos}"`;
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
            // If input is empty and append is enabled, start auto-generation with random prompt
            if (!inputEl.value.trim() && !appendBtn.disabled) {
                state.transformer.autoGenerating = true;
                state.transformer.eosCount = 0;
                inputEl.value = getRandomStarter();
                inputEl.dispatchEvent(new Event('input'));
                doPredict();
                setTimeout(() => {
                    if (state.transformer.autoGenerating && !appendBtn.disabled) {
                        appendBtn.click();
                    }
                }, 800);
                return;
            }
            
            const preds = predictNextTokens(inputEl.value, parseFloat(tempEl.value), 5);
            if (preds.length > 0) {
                // Sample from the predictions based on their probabilities instead of always picking the top one
                const rand = Math.random();
                let cumulative = 0;
                let tok = preds[0].token;
                for (const pred of preds) {
                    cumulative += pred.prob;
                    if (rand <= cumulative) {
                        tok = pred.token;
                        break;
                    }
                }
                
                // Check if it's an end-of-sequence token
                if (tok === toyLM.eos) {
                    state.transformer.eosCount++;
                    
                    // After 2-3 </s> tokens, restart with a random prompt
                    if (state.transformer.eosCount >= 2 + Math.floor(Math.random() * 2)) {
                        state.transformer.eosCount = 0;
                        state.transformer.autoGenerating = true;
                        
                        // Clear and start with random prompt
                        setTimeout(() => {
                            inputEl.value = getRandomStarter();
                            inputEl.dispatchEvent(new Event('input'));
                            doPredict();
                            
                            // Continue auto-generating
                            setTimeout(() => {
                                if (state.transformer.autoGenerating) {
                                    appendBtn.click();
                                }
                            }, 800);
                        }, 500);
                    } else {
                        // Append </s> and continue
                        inputEl.value = (inputEl.value ? inputEl.value + ' ' : '') + tok;
                        inputEl.dispatchEvent(new Event('input'));
                        doPredict();
                        
                        // Auto-continue to trigger next prediction
                        setTimeout(() => {
                            if (state.transformer.autoGenerating) {
                                appendBtn.click();
                            }
                        }, 800);
                    }
                    return;
                } else {
                    // Reset EOS counter when we get a non-EOS token
                    state.transformer.eosCount = 0;
                }
                
                inputEl.value = (inputEl.value ? inputEl.value + ' ' : '') + tok;
                
                // Trigger input event to update counter
                inputEl.dispatchEvent(new Event('input'));
                
                // Auto-predict after appending
                doPredict();
                
                // If in auto-generation mode, continue appending
                if (state.transformer.autoGenerating && inputEl.value.split(/\s+/).length < 20) {
                    setTimeout(() => {
                        if (state.transformer.autoGenerating && !appendBtn.disabled) {
                            appendBtn.click();
                        }
                    }, 800);
                }
            }
        });
        clearBtn.addEventListener('click', () => {
            state.transformer.autoGenerating = false;
            state.transformer.eosCount = 0;
            inputEl.value = '';
            updateTransformerResults([]);
            transformerViz.setTokens(['<start>']);
            transformerViz.setPredictions([]);
            
            // Clear all layer activations
            if (transformerViz.layers) {
                transformerViz.layers.forEach(layer => {
                    layer.forEach(node => {
                        node.activation = 0;
                        node.targetActivation = 0;
                    });
                });
            }
            
            // Clear particles
            transformerViz.particles = [];
            
            // Force redraw to clear visualization
            transformerViz.draw();
            
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
    
    update(dtFrames = 1, speed = (typeof state !== 'undefined' && state.deepPerceptron ? state.deepPerceptron.speed : 3)) {
        if (!this.layers || this.layers.length === 0) return;
        
        const speedFactor = Math.max(0.25, speed / 3);
        
        // Smooth activation interpolation (time-scaled)
        this.layers.forEach(layer => {
            if (!layer) return;
            layer.forEach(node => {
                if (!node) return;
                const diff = (node.targetActivation || 0) - (node.activation || 0);
                node.activation = (node.activation || 0) + diff * (0.15 * Math.min(2, dtFrames));
            });
        });
        
        // Linear cycle progress (time-scaled)
        this.cycleProgress += 0.002 * dtFrames * speedFactor;
        
        // Determine which layer should be processing
    const totalLayers = this.layers.length;
    const layerDuration = 1.0 / totalLayers;
    const rawTargetLayer = Math.min(totalLayers - 1, Math.floor(this.cycleProgress / layerDuration));
    const targetLayer = Math.min(this.currentLayer + 1, rawTargetLayer);
        
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
        
        // Continuous low-rate emission while a layer is active
        if (this.currentLayer < totalLayers - 1) {
            const sourceLayer = this.layers[this.currentLayer];
            const nextLayer = this.layers[this.currentLayer + 1];
            const spawnChance = 0.15 * Math.min(3, dtFrames) * speedFactor;
            if (Math.random() < spawnChance) {
                const activeSources = sourceLayer.filter(n => (n && (n.activation || 0) > 0.2));
                if (activeSources.length > 0) {
                    const sourceNode = activeSources[Math.floor(Math.random() * activeSources.length)];
                    const targetNode = nextLayer[Math.floor(Math.random() * nextLayer.length)];
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
            }
        }
        
        // Update particles smoothly (time-scaled)
        this.particles = this.particles.filter(p => {
            if (!p) return false;
            p.progress = Math.min(1, (p.progress || 0) + (p.speed || 0.015) * dtFrames);
            p.life = Math.max(0, (p.life || 1) - 0.008 * dtFrames);
            return p.life > 0 && p.progress < 1;
        });
        
        // Only reset once we've reached the last layer; otherwise hold near completion
        if (this.cycleProgress >= 1.0) {
            if (this.currentLayer >= totalLayers - 1) {
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
            } else {
                this.cycleProgress = 1.0 - 1e-6;
            }
        }
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
        deepPerceptronViz.resize(true);
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

let deepLastTime = 0;
function animateDeepPerceptron(currentTime = 0) {
    if (!state.deepPerceptron.running || state.currentDemo !== 'deep-perceptron') return;
    
    const frameMs = 1000 / 60;
    let dtFrames = 1;
    if (deepLastTime) {
        dtFrames = Math.max(0.5, Math.min(3, (currentTime - deepLastTime) / frameMs));
    }
    deepLastTime = currentTime;
    
    if (deepPerceptronViz) {
        deepPerceptronViz.update(dtFrames, state.deepPerceptron.speed);
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
        this.resize(true);
        this.setupLayers();
    }
    
    resize(force = false) {
        if (!force && !shouldHandleResize()) return;
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
        normalizingFlowViz.resize(true);
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
        this.resize(true);
        this.setupLayers();
    }
    
    resize(force = false) {
        if (!force && !shouldHandleResize()) return;
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
    
    update(dtFrames = 1, speed = (typeof state !== 'undefined' && state.vae ? state.vae.speed : 3)) {
        // Halve the effective speed
        const speedFactor = 0.5 * Math.max(0.25, speed / 3);
        // Smooth decay for all nodes (time-scaled)
        this.layers.forEach(layer => {
            layer.forEach(node => {
                node.activation *= Math.pow(0.92, dtFrames);
            });
        });
        
        // Linear layer-by-layer flow (time-scaled)
        this.dataFlowTimer += dtFrames * speedFactor;
        // About every 15 frames at 60fps (0.25s baseline), advance flow
        while (this.dataFlowTimer >= 15) {
            this.dataFlowTimer -= 15;
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
            p.progress += 0.08 * dtFrames * speedFactor;
            p.life -= 0.02 * dtFrames;
            
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
        vaeViz.resize(true);
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

let vaeLastTime = 0;
function animateVAE(currentTime = 0) {
    if (!state.vae.running || state.currentDemo !== 'vae') return;
    const frameMs = 1000 / 60;
    let dtFrames = 1;
    if (vaeLastTime) {
        dtFrames = Math.max(0.5, Math.min(3, (currentTime - vaeLastTime) / frameMs));
    }
    vaeLastTime = currentTime;
    vaeViz.update(dtFrames, state.vae.speed);
    vaeViz.draw();
    requestAnimationFrame(animateVAE);
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
        this.dataFlowTimer = 0; // measured in 60fps frames
        this.resize(true);
        this.setupMaps();
    }
    
    resize(force = false) {
        if (!force && !shouldHandleResize()) return;
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
    
    update(dtFrames = 1, speed = (typeof state !== 'undefined' && state.cnnEncoderDecoder ? state.cnnEncoderDecoder.speed : 3)) {
        const speedFactor = Math.max(0.25, speed / 3);
        // Smooth activation decay (only for nodes that were hit) – keep activations alive longer
        this.featureMaps.forEach(maps => {
            maps.forEach(map => {
                if (map.activation > 0) {
                    map.activation *= Math.pow(0.96, dtFrames);
                    if (map.activation < 0.05) map.activation = 0;
                }
            });
        });
        
        // Linear data flow: process one layer at a time (time-scaled)
        this.dataFlowTimer += dtFrames * speedFactor;
        // Activate first layer initially
        if (this.currentLayer === -1 && this.dataFlowTimer > 0) {
            this.currentLayer = 0;
            // Activate all nodes in first layer
            this.featureMaps[0].forEach(map => {
                map.activation = 1.0;
            });
        }
        
        // Every ~15 frames, move to next layer (faster so activations remain above threshold)
        while (this.dataFlowTimer >= 15 && this.currentLayer >= 0 && this.currentLayer < this.featureMaps.length - 1) {
            this.dataFlowTimer -= 15;
            const sourceLayer = this.currentLayer;
            const targetLayer = this.currentLayer + 1;
            
            // Create particles from current layer to next
            const sourceMaps = this.featureMaps[sourceLayer];
            const targetMaps = this.featureMaps[targetLayer];
            
            // Only create particles from ACTIVE source maps
            sourceMaps.forEach((sourceMap, sourceIdx) => {
                if (sourceMap.activation > 0.1) { // More permissive so flow always continues
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
                        // Proactively activate target so it can emit on the next step
                        targetMap.activation = Math.max(targetMap.activation, 0.8);
                    }
                }
            });
            
            this.currentLayer++;
        }
        
        // Between steps, emit a small number of particles continuously from current layer to next
        if (this.currentLayer >= 0 && this.currentLayer < this.featureMaps.length - 1) {
            const sourceMaps = this.featureMaps[this.currentLayer];
            const targetMaps = this.featureMaps[this.currentLayer + 1];
            const spawnChance = 0.18 * Math.min(3, dtFrames) * speedFactor;
            if (Math.random() < spawnChance) {
                // Pick an active source map
                const activeSources = sourceMaps.filter(m => m.activation > 0.1);
                if (activeSources.length > 0) {
                    const s = activeSources[Math.floor(Math.random() * activeSources.length)];
                    const tIdx = Math.floor(Math.random() * targetMaps.length);
                    const t = targetMaps[tIdx];
                    this.particles.push({
                        x: s.x + s.width / 2,
                        y: s.y + s.height / 2,
                        targetX: t.x + t.width / 2,
                        targetY: t.y + t.height / 2,
                        progress: 0,
                        life: 1,
                        sourceLayer: this.currentLayer,
                        targetLayer: this.currentLayer + 1,
                        targetMapIndex: tIdx
                    });
                    t.activation = Math.max(t.activation, 0.7);
                }
            }
        }
        
        // Reset after completing the flow
        if (this.currentLayer >= this.featureMaps.length - 1 && this.particles.length === 0) {
            this.dataFlowTimer = 0;
            this.currentLayer = -1;
        }
        
        // Update particles and activate ONLY target maps that are hit
        this.particles = this.particles.filter(p => {
            p.progress += 0.035 * dtFrames * speedFactor; // time-scaled
            p.life -= 0.02 * dtFrames; // time-scaled
            
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
        cnnViz.resize(true);
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

let cnnLastTime = 0;
function animateCNN(currentTime = 0) {
    if (!state.cnnEncoderDecoder.running || state.currentDemo !== 'cnn-encoder-decoder') return;
    const frameMs = 1000 / 60;
    let dtFrames = 1;
    if (cnnLastTime) {
        dtFrames = Math.max(0.5, Math.min(3, (currentTime - cnnLastTime) / frameMs));
    }
    cnnLastTime = currentTime;
    cnnViz.update(dtFrames, state.cnnEncoderDecoder.speed);
    cnnViz.draw();
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
        this.resize(true);
        this.setupLayers();
    }
    
    resize(force = false) {
        if (!force && !shouldHandleResize()) return;
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
        mamba2Viz.resize(true);
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
        this.resize(true);
        this.setupArchitecture();
    }
    
    resize(force = false) {
        if (!force && !shouldHandleResize()) return;
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
    
    update(dtFrames = 1, speed = (typeof state !== 'undefined' && state.cuda ? state.cuda.speed : 3)) {
        // Halve the effective speed
        const speedFactor = 0.5 * Math.max(0.25, speed / 3);
        this.animationTime += dtFrames * speedFactor;
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
        cudaViz.resize(true);
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

let cudaLastTime = 0;
function animateCUDA(currentTime = 0) {
    if (!state.cuda.running || state.currentDemo !== 'cuda') return;
    
    const frameMs = 1000 / 60;
    let dtFrames = 1;
    if (cudaLastTime) {
        dtFrames = Math.max(0.5, Math.min(3, (currentTime - cudaLastTime) / frameMs));
    }
    cudaLastTime = currentTime;
    
    cudaViz.update(dtFrames, state.cuda.speed);
    cudaViz.draw();
    requestAnimationFrame(animateCUDA);
}

// @license-end
