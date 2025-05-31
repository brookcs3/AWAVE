import Emitter from './Emitter.js';
import Noise from './Noise.js';
// GSAP is loaded globally from CDN

// Check for browser support of required features
const supportsCustomElements = 'customElements' in window;
const supportsSVG = !!document.createElementNS && !!document.createElementNS('http://www.w3.org/2000/svg', 'svg');
const supportsWebAnimations = 'animate' in Element.prototype;

class AWaves extends HTMLElement {
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
    Emitter.on('animationState', this.onAnimationState, this);

    this.addEventListener('touchmove', this.onTouchMove.bind(this));
    this.addEventListener('mousedown', this.onMouseDown.bind(this));
    this.addEventListener('mousemove', this.onElementMouseMove.bind(this));
    this.addEventListener('mouseup', this.onMouseUp.bind(this));
    this.addEventListener('intersect', this.onIntersect.bind(this), { passive: true });
    this.addEventListener('introend', this.onIntroEnd.bind(this));
  }

  onAudioData(data) {
    // Track previous energy for peak detection
    this.prevEnergy = this.audioData.energy || 0;
    this.audioData.energy = data.energy || 0;
    this.audioData.bass = data.bass || 0;
    this.audioData.mid = data.mid || 0;
    this.audioData.high = data.high || 0;
    
    // Detect peak for sunspot effect (significant sudden increase in energy)
    const energyDiff = this.audioData.energy - this.prevEnergy;
    if (energyDiff > 0.2 && this.audioData.energy > 0.3) {
      this.sunspotTrigger = true;
      this.sunspotIntensity = this.audioData.energy;
      this.sunspotTime = Date.now();
    } else if (this.sunspotTrigger && Date.now() - this.sunspotTime > 200) {
      // Fade out after 200ms
      this.sunspotTrigger = false;
      this.sunspotIntensity = 0;
    }
  }

  onAnimationState(state) {
    this.animationState = state || 1;
    console.log('Animation state changed to:', this.animationState);
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
    this.noise = new Noise(Math.random());
    this.audioData = {
      energy: 0,
      bass: 0,
      mid: 0,
      high: 0
    };
    this.animationState = 1; // Default animation state (1 to 4)
    this.prevEnergy = 0;
    this.sunspotTrigger = false;
    this.sunspotIntensity = 0;
    this.sunspotTime = 0;

    this.isInteractive = false;
    this.isPaused = false; // Start with animation running
    this.isDragging = false;

    // Remove theme-contrasted class
    document.documentElement.classList.remove('theme-contrasted');

    // Init
    this.setSize();
    this.setLines();
    this.bindEvents();
    
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
    const xGap = 10;
    const yGap = 32;
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
    const { lines, mouse, noise, audioData, animationState, sunspotTrigger, sunspotIntensity } = this;
    lines.forEach((points) => {
      points.forEach((p) => {
        const move = noise.perlin2(
          (p.x + time * 0.0125) * 0.002,
          (p.y + time * 0.005) * 0.0015
        ) * 12;
        // Base wave movement influenced by noise
        let waveX = Math.cos(move) * 32;
        let waveY = Math.sin(move) * 16;
        
        // Modify wave movement based on audio data with different animation states
        if (audioData.energy > 0) {
          const audioInfluence = audioData.energy * 50; // Scale energy for noticeable effect
          if (animationState === 1) {
            // State 1: Corner stretch with central twist (right then left) and vibrations
            const centerX = this.bounding.width / 2;
            const centerY = this.bounding.height / 2;
            const distFromCenter = Math.hypot(p.x - centerX, p.y - centerY);
            const maxDist = Math.hypot(centerX, centerY);
            const stretchFactor = distFromCenter / maxDist; // Stretch more at edges
            const twistAngle = (time * 0.01 + audioData.bass * 2) * (p.x > centerX ? 1 : -1);
            waveX += Math.cos(twistAngle) * audioInfluence * stretchFactor * 1.5;
            waveY += Math.sin(twistAngle) * audioInfluence * stretchFactor * 1.5;
            // Add vibration effect with high frequencies
            waveX += Math.sin(time * 0.05 + audioData.high * 3) * audioInfluence * 0.2;
            waveY += Math.cos(time * 0.05 + audioData.high * 3) * audioInfluence * 0.2;
          } else if (animationState === 2) {
            // State 2: Horizontal waves driven by bass - wide sweeping motion
            waveX += Math.cos(move + audioData.bass * 3 + p.y * 0.002) * audioInfluence * 1.5;
            waveY += Math.sin(move + audioData.mid * 0.5) * audioInfluence * 0.3;
          } else if (animationState === 3) {
            // State 3: Wide rolling waves top to bottom with varying sizes
            const verticalPos = p.y / this.bounding.height; // Position factor from top to bottom
            const waveSize = Math.sin(verticalPos * Math.PI + time * 0.01) * 0.5 + 0.5; // Varying wave size
            waveX += Math.cos(move + audioData.mid * 2 + verticalPos * 2) * audioInfluence * 0.5 * waveSize;
            waveY += Math.sin(move + audioData.bass * 1.5 + time * 0.02) * audioInfluence * 1.2 * (1 + waveSize * 0.5);
          } else if (animationState === 4) {
            // State 4: Balanced wave pattern - less chaotic, more rhythmic
            const rhythmMove = move + audioData.bass * 1.5 + audioData.mid * 0.8;
            waveX += Math.sin(rhythmMove) * audioInfluence * 0.6;
            waveY += Math.cos(rhythmMove + audioData.high * 0.5) * audioInfluence * 0.6;
          }
        }
        
        // Add sunspot-like variations triggered by frequency peaks
        if (sunspotTrigger) {
          const centerX = this.bounding.width / 2;
          const centerY = this.bounding.height / 2;
          const distFromCenter = Math.hypot(p.x - centerX, p.y - centerY);
          const maxDist = Math.hypot(centerX, centerY);
          if (distFromCenter < maxDist * 0.3) { // Small burst near center
            const burstFactor = (1 - distFromCenter / (maxDist * 0.3)) * sunspotIntensity * 20;
            waveX += Math.cos(time * 0.1 + p.x * 0.01) * burstFactor;
            waveY += Math.sin(time * 0.1 + p.y * 0.01) * burstFactor;
          }
        }
        
        p.wave.x = waveX;
        p.wave.y = waveY;
        
        if (this.isInteractive) {
          const dx = p.x - mouse.sx;
          const dy = p.y - mouse.sy;
          const d = Math.hypot(dx, dy);
          const l = Math.max(175, mouse.vs);
          if (d < l) {
            const s = 1 - d / l;
            const f = Math.cos(d * 0.001) * s;
            p.cursor.vx += Math.cos(mouse.a) * f * l * mouse.vs * 0.00065;
            p.cursor.vy += Math.sin(mouse.a) * f * l * mouse.vs * 0.00065;
          }
          p.cursor.vx += (0 - p.cursor.x) * 0.005;
          p.cursor.vy += (0 - p.cursor.y) * 0.005;
          p.cursor.vx *= 0.925;
          p.cursor.vy *= 0.925;
          p.cursor.x += p.cursor.vx * 2;
          p.cursor.y += p.cursor.vy * 2;
          p.cursor.x = Math.min(100, Math.max(-100, p.cursor.x));
          p.cursor.y = Math.min(100, Math.max(-100, p.cursor.y));
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
    // Increase mouse tracking responsiveness by adjusting the easing factor
    mouse.sx += (mouse.x - mouse.sx) * 0.1; // Increased from 0.1 for faster response
    mouse.sy += (mouse.y - mouse.sy) * 0.4; // Increased from 0.1 for faster response
    const dx = mouse.x - mouse.lx;
    const dy = mouse.y - mouse.ly;
    const d = Math.hypot(dx, dy);
    mouse.v = d;
    mouse.vs += (d - mouse.vs) * 0.2; // Increased from 0.1 for faster velocity tracking
    mouse.vs = Math.min(100, mouse.vs);
    mouse.lx = mouse.x;
    mouse.ly = mouse.y;
    mouse.a = Math.atan2(dy, dx);
    this.style.setProperty('--x', `${mouse.sx}px`);
    this.style.setProperty('--y', `${mouse.sy}px`);
    this.movePoints(time);
    this.drawLines();
  }
}

customElements.define('a-waves', AWaves);