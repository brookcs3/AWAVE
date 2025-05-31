import Emitter from './Emitter.js';
import Noise from './Noise.js';
import LiquidMetalEffect from './LiquidMetalEffect.js';
// GSAP is loaded globally from CDN

// Check for browser support of required features
const supportsCustomElements = 'customElements' in window;
const supportsSVG = !!document.createElementNS && !!document.createElementNS('http://www.w3.org/2000/svg', 'svg');
const supportsWebAnimations = 'animate' in Element.prototype;

class AWavesVariant2 extends HTMLElement {
  svg;
  bounding;
  mouse;
  lines;
  paths;
  noise;
  isInteractive;
  isPaused;
  isDragging;

  bindEvents() {
    Emitter.on('mousemove', this.onMouseMove, this);
    Emitter.on('resize', this.onResize, this);
    Emitter.on('audioData', this.onAudioData, this);

    this.addEventListener('touchmove', this.onTouchMove.bind(this));
    this.addEventListener('mousedown', this.onMouseDown.bind(this));
    this.addEventListener('mousemove', this.onElementMouseMove.bind(this));
    this.addEventListener('mouseup', this.onMouseUp.bind(this));
    this.addEventListener('intersect', this.onIntersect.bind(this), { passive: true });
    this.addEventListener('introend', this.onIntroEnd.bind(this));
  }

  onAudioData(data) {
    this.audioData.energy = data.energy || 0;
    this.audioData.bass = data.bass || 0;
    this.audioData.mid = data.mid || 0;
    this.audioData.high = data.high || 0;
  }

  onMouseDown(e) {
    this.isDragging = true;
    this.updateMousePosition(e.clientX, e.clientY);
  }

  onElementMouseMove(e) {
    if (this.isDragging) {
      this.updateMousePosition(e.clientX, e.clientY);
    }
  }

  onMouseUp() {
    this.isDragging = false;
  }

  onResize() {
    this.setSize();
    this.setLines();
  }

  onMouseMove(x, y) {
    this.updateMousePosition(x, y);
  }

  onTouchMove(e) {
    e.preventDefault();
    const touch = e.touches[0];
    this.updateMousePosition(touch.clientX, touch.clientY);
  }

  updateMousePosition(x, y) {
    const mouse = this.mouse;
    mouse.x = x - this.bounding.left;
    mouse.y = y - this.bounding.top + window.scrollY;
    if (!mouse.set) {
      mouse.sx = mouse.x;
      mouse.sy = mouse.y;
      mouse.lx = mouse.x;
      mouse.ly = mouse.y;
      mouse.set = true;
    }
  }

  onIntersect(e) {
    this.isPaused = !e.detail.isIntersecting;
    if (this.isPaused) {
      Emitter.off('tick', this.tick, this);
    } else {
      Emitter.on('tick', this.tick, this);
    }
  }
  
  connectedCallback() {
    // Elements
    this.svg = this.querySelector('.js-svg');

    // Properties
    this.mouse = {
      x: -10,
      y: 0,
      lx: 0,
      ly: 0,
      sx: 0,
      sy: 0,
      v: 0,
      vs: 0,
      a: 0,
      set: false,
    };

    this.lines = [];
    this.paths = [];
    // Use a different seed for noise to create a unique animation pattern
    this.noise = new Noise(Math.random() * 1000);
    this.audioData = {
      energy: 0,
      bass: 0,
      mid: 0,
      high: 0
    };

    this.isInteractive = false;
    this.isPaused = false; // Start with animation running
    this.isDragging = false;
    // Initialize the liquid metal effect for this variant
    this.liquidMetalEffect = null; // Will be initialized in connectedCallback
    // Morphing properties to sample Viz 1 style
    this.morphingActive = true; // Can be toggled if needed
    this.morphTimeoutId = null; // For scheduling morphing
    this.viz1Sample = {
      waveAmplitudeX: 32, // Default Viz 1 values to morph towards
      waveAmplitudeY: 16,
      noiseScale: 12
    };
    // Dance routine properties for traveling peaks
    this.danceActive = true; // Can be toggled if needed
    this.danceFocusX = 0.5; // Normalized position (0 to 1) of the peak focus on X axis
    this.danceFocusY = 0.5; // Normalized position (0 to 1) of the peak focus on Y axis
    this.danceJourneyTime = 0; // Tracks progress of the journey for rhythmic movement
    this.danceCycleDuration = 15000; // 15 seconds for a full journey cycle across the grid

    // Remove theme-contrasted class
    document.documentElement.classList.remove('theme-contrasted');

    // Init
    this.setSize();
    this.setLines();
    this.bindEvents();
    // Initialize liquid metal effect with reduced intensity
    this.liquidMetalEffect = new LiquidMetalEffect(this.svg, {
      isActive: true, // Start with the effect active
      minInterval: 30000, // Increase to 30 seconds minimum for less frequent triggering
      maxInterval: 80000, // Increase to 80 seconds maximum for less frequent triggering
      duration: 2500, // Slower animation for a less jarring effect
      color: 'var(--color-secondary)', // Match the visualizer's secondary color
      thickness: 2 // Reduce stroke width for subtler appearance
    });
    // Schedule morphing effect to Viz 1 style
    this.scheduleMorphToViz1();
    // No separate scheduling for dance routine; it's updated in tick
    
    // Start animation immediately
    Emitter.on('tick', this.tick, this);
    
    // Set up animation frame
    const animate = (time) => {
      Emitter.emit('tick', time);
      requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }

  onIntroEnd() {
    this.isInteractive = true;
    
    // Simple animation for lines
    const lines = this.svg.querySelectorAll('.js-line');
    lines.forEach((line, index) => {
      setTimeout(() => {
        line.style.opacity = '1';
      }, index * 50);
    });
  }

  setSize() {
    const bounding = this.getBoundingClientRect();
    this.svg.style.width = '';
    this.svg.style.height = '';
    this.bounding = {
      left: bounding.left,
      top: bounding.top + window.scrollY,
      width: this.clientWidth,
      height: this.clientHeight,
    };
    this.svg.style.width = `${this.bounding.width}px`;
    this.svg.style.height = `${this.bounding.height}px`;
  }

  setLines() {
    const { width, height } = this.bounding;
    this.lines = [];
    if (this.paths) this.paths.forEach((path) => path.remove());
    this.paths = [];
    const xGap = 12; // Adjusted gap for a different visual density
    const yGap = 28; // Adjusted gap for a different visual density
    const oWidth = width + 200;
    const oHeight = height + 30;
    const totalLines = Math.ceil(oWidth / xGap);
    const totalPoints = Math.ceil(oHeight / yGap);
    const xStart = (width - xGap * totalLines) / 2;
    const yStart = (height - yGap * totalPoints) / 2;

    for (let i = 0; i <= totalLines; i++) {
      const points = [];
      for (let j = 0; j <= totalPoints; j++) {
        const point = {
          x: xStart + xGap * i,
          y: yStart + yGap * j,
          wave: { x: 0, y: 0 },
          cursor: { x: 0, y: 0, vx: 0, vy: 0 },
        };
        points.push(point);
      }
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.classList.add('a__line', 'js-line');
      this.svg.appendChild(path);
      this.paths.push(path);
      this.lines.push(points);
    }
    if (this.isPaused) {
      this.drawLines();
    }
  }

  movePoints(time) {
    const { lines, mouse, noise, audioData } = this;
    lines.forEach((points) => {
      points.forEach((p) => {
        const move = noise.perlin2(
          (p.x + time * 0.015) * 0.003, // Adjusted timing for different wave speed
          (p.y + time * 0.007) * 0.002 // Adjusted timing for different wave speed
        ) * 10;
        // Base wave movement influenced by noise with different scaling
        let waveX = Math.cos(move) * 40; // Increased amplitude for a more pronounced effect
        let waveY = Math.sin(move) * 20; // Increased amplitude for a more pronounced effect
        
        // Modify wave movement based on audio data with different influence
        if (audioData.energy > 0) {
          const audioInfluence = audioData.energy * 70; // Increased scale for more dramatic audio response
          waveX += Math.cos(move + audioData.high * 3) * audioInfluence; // Changed to use high frequencies
          waveY += Math.sin(move + audioData.bass * 3) * audioInfluence; // Changed to use bass frequencies
        }
        
        p.wave.x = waveX;
        p.wave.y = waveY;
        
        if (this.isInteractive) {
          const dx = p.x - mouse.sx;
          const dy = p.y - mouse.sy;
          const d = Math.hypot(dx, dy);
          const l = Math.max(200, mouse.vs); // Adjusted interaction radius
          if (d < l) {
            const s = 1 - d / l;
            const f = Math.cos(d * 0.002) * s; // Adjusted force calculation
            p.cursor.vx += Math.cos(mouse.a) * f * l * mouse.vs * 0.0008; // Adjusted force strength
            p.cursor.vy += Math.sin(mouse.a) * f * l * mouse.vs * 0.0008; // Adjusted force strength
          }
          p.cursor.vx += (0 - p.cursor.x) * 0.004; // Adjusted damping
          p.cursor.vy += (0 - p.cursor.y) * 0.004; // Adjusted damping
          p.cursor.vx *= 0.9; // Adjusted friction
          p.cursor.vy *= 0.9; // Adjusted friction
          p.cursor.x += p.cursor.vx * 2.5; // Adjusted movement speed
          p.cursor.y += p.cursor.vy * 2.5; // Adjusted movement speed
          p.cursor.x = Math.min(120, Math.max(-120, p.cursor.x)); // Adjusted bounds
          p.cursor.y = Math.min(120, Math.max(-120, p.cursor.y)); // Adjusted bounds
        }
      });
    });
  }

  moved(point, withCursorForce = true) {
    const coords = {
      x: point.x + point.wave.x + (withCursorForce ? point.cursor.x : 0),
      y: point.y + point.wave.y + (withCursorForce ? point.cursor.y : 0),
    };
    coords.x = Math.round(coords.x * 10) / 10;
    coords.y = Math.round(coords.y * 10) / 10;
    return coords;
  }

  drawLines() {
    const { lines, moved, paths } = this;
    lines.forEach((points, lIndex) => {
      let p1 = moved(points[0], false);
      let d = `M ${p1.x} ${p1.y}`;
      points.forEach((p1, pIndex) => {
        const isLast = pIndex === points.length - 1;
        p1 = moved(p1, !isLast);
        const p2 = moved(points[pIndex + 1] || points[points.length - 1], !isLast);
        d += `L ${p1.x} ${p1.y}`;
      });
      paths[lIndex].setAttribute('d', d);
    });
  }

  tick(time) {
    const { mouse } = this;
    // Adjusted mouse tracking responsiveness for a different feel
    mouse.sx += (mouse.x - mouse.sx) * 0.15;
    mouse.sy += (mouse.y - mouse.sy) * 0.5;
    const dx = mouse.x - mouse.lx;
    const dy = mouse.y - mouse.ly;
    const d = Math.hypot(dx, dy);
    mouse.v = d;
    mouse.vs += (d - mouse.vs) * 0.25;
    mouse.vs = Math.min(120, mouse.vs); // Adjusted max velocity
    mouse.lx = mouse.x;
    mouse.ly = mouse.y;
    mouse.a = Math.atan2(dy, dx);
    this.style.setProperty('--x', `${mouse.sx}px`);
    this.style.setProperty('--y', `${mouse.sy}px`);
    // Update dance routine focus for traveling peaks
    this.updateDanceRoutine(time);
    this.movePoints(time);
    this.drawLines();
    // The liquid metal effect is managed independently via its own scheduling
  }

  // Update the dance routine to move the focus of wave peaks across the grid
  updateDanceRoutine(time) {
    if (!this.danceActive) return;
    
    // Increment journey time based on cycle duration (full cycle in 15 seconds)
    this.danceJourneyTime += 1 / (this.danceCycleDuration / 16.67); // Approx 60fps update
    
    // Reset journey time if it completes a cycle
    if (this.danceJourneyTime >= 1) {
      this.danceJourneyTime = 0;
    }
    
    // Calculate a traveling focus point using a sinusoidal pattern for smooth "dance" movement
    // This creates a figure-8 or circular journey across the grid over the cycle
    const phase = this.danceJourneyTime * Math.PI * 2;
    this.danceFocusX = 0.5 + Math.sin(phase) * 0.4; // Moves left to right and back (0.1 to 0.9)
    this.danceFocusY = 0.5 + Math.cos(phase * 0.5) * 0.3; // Moves top to bottom slower (0.2 to 0.8)
  }

  // Schedule morphing to Viz 1 style randomly every 8-15 seconds for less frequent changes
  scheduleMorphToViz1() {
    if (!this.morphingActive) return;
    
    const interval = Math.random() * 7000 + 8000; // Random between 8-15 seconds for less frequent morphing
    if (this.morphTimeoutId) {
      clearTimeout(this.morphTimeoutId);
    }
    this.morphTimeoutId = setTimeout(() => {
      this.triggerMorphToViz1();
      this.scheduleMorphToViz1(); // Schedule the next morph
    }, interval);
  }

  // Trigger morphing animation to Viz 1 style and back
  triggerMorphToViz1() {
    if (!this.morphingActive) return;
    
    // Determine if morphing should be total or partial (randomly)
    const isTotalMorph = Math.random() > 0.7; // Reduce chance of total morph to 30% for subtler effect
    const morphPercentage = isTotalMorph ? 0.8 : Math.random() * 0.3 + 0.2; // Limit to 20-50% for partial, 80% for total
    
    // Current Viz 2 settings (to morph from)
    const currentWaveX = 40; // From movePoints in Viz 2
    const currentWaveY = 20;
    const currentNoiseScale = 10;
    
    // Target Viz 1 settings (to morph to)
    const targetWaveX = this.viz1Sample.waveAmplitudeX * morphPercentage + currentWaveX * (1 - morphPercentage);
    const targetWaveY = this.viz1Sample.waveAmplitudeY * morphPercentage + currentWaveY * (1 - morphPercentage);
    const targetNoiseScale = this.viz1Sample.noiseScale * morphPercentage + currentNoiseScale * (1 - morphPercentage);
    
    // Use GSAP to animate the morph to Viz 1 style
    gsap.to(this, {
      duration: 2.0, // Slower transition for a smoother, less intense morph
      onUpdate: () => {
        // Update internal values for wave rendering (will be used in movePoints)
        this.morphWaveX = gsap.getProperty(this, 'morphWaveX');
        this.morphWaveY = gsap.getProperty(this, 'morphWaveY');
        this.morphNoiseScale = gsap.getProperty(this, 'morphNoiseScale');
      },
      onComplete: () => {
        // After reaching Viz 1 style, morph back to Viz 2 style
        gsap.to(this, {
          morphWaveX: currentWaveX,
          morphWaveY: currentWaveY,
          morphNoiseScale: currentNoiseScale,
          duration: 2.0, // Slower transition back for a smoother effect
          onUpdate: () => {
            this.morphWaveX = gsap.getProperty(this, 'morphWaveX');
            this.morphWaveY = gsap.getProperty(this, 'morphWaveY');
            this.morphNoiseScale = gsap.getProperty(this, 'morphNoiseScale');
          }
        });
      },
      morphWaveX: targetWaveX,
      morphWaveY: targetWaveY,
      morphNoiseScale: targetNoiseScale
    });
  }

  // Override movePoints to use morphed values if available
  movePoints(time) {
    const { lines, mouse, noise, audioData } = this;
    lines.forEach((points, lineIndex) => {
      // Calculate normalized position of this line for dance focus comparison
      const lineCount = lines.length;
      const lineNormX = lineIndex / lineCount; // 0 to 1 across the grid width
      
      points.forEach((p, pointIndex) => {
        // Calculate normalized position of this point for dance focus comparison
        const pointCount = points.length;
        const pointNormY = pointIndex / pointCount; // 0 to 1 across the grid height
        
        // Calculate distance from the dance focus point
        const dx = lineNormX - this.danceFocusX;
        const dy = pointNormY - this.danceFocusY;
        const distance = Math.hypot(dx, dy);
        
        // Amplify wave movement near the focus point for a traveling peak effect
        const focusRadius = 0.2; // Radius of influence for the peak
        const focusEffect = Math.max(0, 1 - distance / focusRadius); // 1 at focus, 0 outside radius
        const focusMultiplier = 1 + focusEffect * 1.5; // Up to 2.5x amplitude at focus point
        
        const move = noise.perlin2(
          (p.x + time * 0.015) * 0.003, // Adjusted timing for different wave speed
          (p.y + time * 0.007) * 0.002 // Adjusted timing for different wave speed
        ) * (this.morphNoiseScale || 10);
        // Base wave movement influenced by noise with different scaling
        let waveX = Math.cos(move) * (this.morphWaveX || 40) * focusMultiplier; // Use morphed value if available, with focus
        let waveY = Math.sin(move) * (this.morphWaveY || 20) * focusMultiplier; // Use morphed value if available, with focus
        
        // Modify wave movement based on audio data with different influence
        if (audioData.energy > 0) {
          const audioInfluence = audioData.energy * 70; // Increased scale for more dramatic audio response
          waveX += Math.cos(move + audioData.high * 3) * audioInfluence; // Changed to use high frequencies
          waveY += Math.sin(move + audioData.bass * 3) * audioInfluence; // Changed to use bass frequencies
        }
        
        p.wave.x = waveX;
        p.wave.y = waveY;
        
        if (this.isInteractive) {
          const dx = p.x - mouse.sx;
          const dy = p.y - mouse.sy;
          const d = Math.hypot(dx, dy);
          const l = Math.max(200, mouse.vs); // Adjusted interaction radius
          if (d < l) {
            const s = 1 - d / l;
            const f = Math.cos(d * 0.002) * s; // Adjusted force calculation
            p.cursor.vx += Math.cos(mouse.a) * f * l * mouse.vs * 0.0008; // Adjusted force strength
            p.cursor.vy += Math.sin(mouse.a) * f * l * mouse.vs * 0.0008; // Adjusted force strength
          }
          p.cursor.vx += (0 - p.cursor.x) * 0.004; // Adjusted damping
          p.cursor.vy += (0 - p.cursor.y) * 0.004; // Adjusted damping
          p.cursor.vx *= 0.9; // Adjusted friction
          p.cursor.vy *= 0.9; // Adjusted friction
          p.cursor.x += p.cursor.vx * 2.5; // Adjusted movement speed
          p.cursor.y += p.cursor.vy * 2.5; // Adjusted movement speed
          p.cursor.x = Math.min(120, Math.max(-120, p.cursor.x)); // Adjusted bounds
          p.cursor.y = Math.min(120, Math.max(-120, p.cursor.y)); // Adjusted bounds
        }
      });
    });
  }
}

customElements.define('a-waves-variant2', AWavesVariant2);