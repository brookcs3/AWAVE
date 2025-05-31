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
    this.colorSyncActive = false; // Track if color sync is active
    this.colorSyncStartTime = 0; // Time when color sync started

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
        // Base wave movement influenced by noise, with subtle baseline and audio enhancement (MilkDrop-inspired)
        // Subtle movement always present; enhanced dynamic movement with audio energy
        let waveX = Math.cos(move + Math.sin(time * 0.01)) * (audioData.energy > 0.05 ? 32 : 10);
        let waveY = Math.sin(move + Math.cos(time * 0.015)) * (audioData.energy > 0.05 ? 16 : 5);
        
        // Modify wave movement based on audio data with different animation states
        if (audioData.energy > 0) {
          const audioInfluence = audioData.energy * 50; // Scale energy for noticeable effect
          if (animationState === 1) {
            // State 1: Radial pulsing from center synchronized with bass, constrained to edges
            const centerX = this.bounding.width / 2;
            const centerY = this.bounding.height / 2;
            const distFromCenter = Math.hypot(p.x - centerX, p.y - centerY);
            const maxDist = Math.hypot(centerX, centerY);
            const radialFactor = distFromCenter / maxDist; // More effect further from center
            // Emphasize bass synchronization with controlled influence to prevent edge separation
            let waveXAdj = Math.cos((p.x - centerX) * 0.005 + audioData.bass * 5) * audioInfluence * radialFactor * 1.5;
            let waveYAdj = Math.sin((p.y - centerY) * 0.005 + audioData.bass * 5) * audioInfluence * radialFactor * 1.5;
            // Constrain movement to stay within screen boundaries
            const maxDisplacementX = this.bounding.width * 0.1; // Limit displacement to 10% of width
            const maxDisplacementY = this.bounding.height * 0.1; // Limit displacement to 10% of height
            waveXAdj = Math.max(-maxDisplacementX, Math.min(maxDisplacementX, waveXAdj));
            waveYAdj = Math.max(-maxDisplacementY, Math.min(maxDisplacementY, waveYAdj));
            waveX += waveXAdj;
            waveY += waveYAdj;
          } else if (animationState === 2) {
            // State 2: Horizontal waves driven by bass - wide sweeping motion
            // Increase bass response for more pronounced effect
            waveX += Math.cos(move + audioData.bass * 5 + p.y * 0.002) * audioInfluence * 2.0;
            waveY += Math.sin(move + audioData.mid * 0.5 + audioData.bass * 2) * audioInfluence * 0.5;
          } else if (animationState === 3) {
            // State 3: Vertical waves synchronized with bass and mid frequencies
            // Significantly increase bass response for stronger vertical motion
            waveX += Math.cos(audioData.mid * 2.0) * audioInfluence * 0.5;
            waveY += Math.sin(audioData.bass * 4.0) * audioInfluence * 2.5;
          } else if (animationState === 4) {
            // State 4: Wave front effect synchronized with music for a thoughtful, appealing look
            // Simulate a traveling wave front across the grid, driven by bass
            const waveFrontPosition = (time * 0.02 + audioData.bass * 3.0) % (this.bounding.width * 2);
            const waveDirection = waveFrontPosition < this.bounding.width ? 1 : -1; // Alternate direction
            const waveCenterX = waveDirection === 1 ? waveFrontPosition : (this.bounding.width * 2 - waveFrontPosition);
            const waveCenterY = this.bounding.height / 2 + Math.sin(time * 0.01 + audioData.mid * 2) * (this.bounding.height / 4);
            const dxWave = p.x - waveCenterX;
            const dyWave = p.y - waveCenterY;
            const distWave = Math.abs(dxWave); // Focus on horizontal distance for wave front
            const waveWidth = 150 + audioData.energy * 80; // Width of wave front influenced by energy
            if (distWave < waveWidth) {
              const intensity = (1 - distWave / waveWidth) * (audioData.energy + 0.7);
              // Create a smooth wave front effect with bass-driven height
              waveX += Math.cos(dxWave * 0.02 + audioData.bass * 2) * intensity * audioInfluence * 1.2 * waveDirection;
              waveY += Math.sin(dyWave * 0.01 + audioData.bass * 3) * intensity * audioInfluence * 1.8;
            } else {
              // Subtle background motion outside the wave front
              waveX += Math.cos(move + audioData.high * 0.5) * audioInfluence * 0.3;
              waveY += Math.sin(move + audioData.mid * 0.5) * audioInfluence * 0.3;
            }
          }
        }
        
        // Add sunspot-like variations triggered by frequency peaks with enhanced burst effect (MilkDrop-inspired custom shape)
        if (sunspotTrigger) {
          const centerX = this.bounding.width / 2;
          const centerY = this.bounding.height / 2;
          const distFromCenter = Math.hypot(p.x - centerX, p.y - centerY);
          const maxDist = Math.hypot(centerX, centerY);
          if (distFromCenter < maxDist * 0.3) { // Small burst near center
            const burstFactor = (1 - distFromCenter / (maxDist * 0.3)) * sunspotIntensity * 30; // Increased intensity
            waveX += Math.cos(time * 0.15 + p.x * 0.02) * burstFactor; // Faster oscillation for burst
            waveY += Math.sin(time * 0.15 + p.y * 0.02) * burstFactor; // Faster oscillation for burst
          }
        }
        
        // Enhanced ColorMorph synchronization with varied frequency detection
        // Detect audio peaks across different frequency bands with varied thresholds
        const bassPeak = audioData.bass > 0.5; // Lower threshold for bass
        const midPeak = audioData.mid > 0.4; // Mid-range threshold
        const highPeak = audioData.high > 0.3; // Higher sensitivity for high frequencies
        if ((bassPeak || midPeak || highPeak) && !this.colorSyncActive) {
          this.colorSyncActive = true;
          this.colorSyncStartTime = Date.now();
          // Dynamic color shift based on frequency dominance
          let dominantHueShift = 0;
          let saturationBoost = audioData.energy * 15; // More dynamic saturation
          if (bassPeak) {
            dominantHueShift = audioData.bass * 40; // Strong bass shifts hue more
          } else if (midPeak) {
            dominantHueShift = audioData.mid * 20 + 30; // Mid shifts to a different spectrum
          } else if (highPeak) {
            dominantHueShift = audioData.high * 10 + 60; // High frequencies shift further
          }
          const newPrimaryHue = (345 + dominantHueShift) % 360; // Varied hue based on frequency
          const newSaturation = Math.min(98, 70 + saturationBoost); // Higher saturation range
          const lightnessAdjust = bassPeak ? 45 : (midPeak ? 50 : 55); // Adjust lightness based on frequency
          // Fluid, expressive color update
          document.documentElement.style.setProperty('--color-primary', this.hslToHex(newPrimaryHue, newSaturation, lightnessAdjust));
          document.documentElement.style.setProperty('--color-secondary', this.hslToHex((newPrimaryHue + 180) % 360, newSaturation - 15, bassPeak ? 10 : 15));
        } else if (this.colorSyncActive && Date.now() - this.colorSyncStartTime > (bassPeak ? 1500 : 800)) {
          // Duration varies based on bass presence for more realistic reaction
          this.colorSyncActive = false;
          // No action needed as ColorMorph will resume its default behavior
        }
        
        p.wave.x = waveX;
        p.wave.y = waveY;
        
        if (this.isInteractive) {
          const dx = p.x - mouse.sx;
          const dy = p.y - mouse.sy;
          const d = Math.hypot(dx, dy);
          const l = Math.max(250, mouse.vs * 1.5); // Increased influence radius and velocity impact
          if (d < l) {
            const s = 1 - d / l;
            const f = Math.cos(d * 0.002) * s; // Increased force factor
            p.cursor.vx += Math.cos(mouse.a) * f * l * mouse.vs * 0.0015; // Stronger force
            p.cursor.vy += Math.sin(mouse.a) * f * l * mouse.vs * 0.0015; // Stronger force
          }
          p.cursor.vx += (0 - p.cursor.x) * 0.01; // Faster return to origin
          p.cursor.vy += (0 - p.cursor.y) * 0.01; // Faster return to origin
          p.cursor.vx *= 0.9; // Slightly less damping for more fluid motion
          p.cursor.vy *= 0.9; // Slightly less damping for more fluid motion
          p.cursor.x += p.cursor.vx * 3; // Increased movement impact
          p.cursor.y += p.cursor.vy * 3; // Increased movement impact
          p.cursor.x = Math.min(150, Math.max(-150, p.cursor.x)); // Wider range of movement
          p.cursor.y = Math.min(150, Math.max(-150, p.cursor.y)); // Wider range of movement
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
    const { lines, moved, paths, audioData } = this;
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
      // MilkDrop-inspired audio-reactive line styling based on frequency bands
      const opacity = 0.5 + audioData.bass * 0.3; // Bass increases opacity
      const strokeWidth = 1 + audioData.high * 0.5; // High frequencies increase thickness
      paths[lIndex].style.opacity = opacity.toFixed(2);
      paths[lIndex].style.strokeWidth = `${strokeWidth.toFixed(2)}px`;
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
  /**
   * Convert HSL values to hex color string (borrowed from ColorMorph.js for color sync)
   */
  hslToHex(h, s, l) {
    l /= 100;
    const a = (s * Math.min(l, 1 - l)) / 100;
    const f = (n) => {
      const k = (n + h / 30) % 12;
      const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
      return Math.round(255 * color)
        .toString(16)
        .padStart(2, '0');
    };
    return `#${f(0)}${f(8)}${f(4)}`;
  }
}

customElements.define('a-waves', AWaves);