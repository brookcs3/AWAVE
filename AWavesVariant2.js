import Emitter from './Emitter.js';
import Noise from './Noise.js';
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
    this.movePoints(time);
    this.drawLines();
  }
}

customElements.define('a-waves-variant2', AWavesVariant2);