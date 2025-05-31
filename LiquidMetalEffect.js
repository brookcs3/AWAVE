/**
 * LiquidMetalEffect.js
 * Implements a pulsing liquid metal ring effect inspired by Terminator 2's T-1000 reforming scene.
 * This effect is designed to be modular and can be easily enabled/disabled in the visualizer.
 */

export default class LiquidMetalEffect {
  constructor(container, options = {}) {
    this.container = container; // The SVG or parent element to apply the effect to
    this.isActive = options.isActive !== undefined ? options.isActive : true; // Whether the effect is enabled
    this.minInterval = options.minInterval || 20000; // Minimum time between effects (20s)
    this.maxInterval = options.maxInterval || 60000; // Maximum time between effects (60s)
    this.duration = options.duration || 2000; // Duration of the pulse effect in milliseconds
    this.color = options.color || 'var(--color-secondary)'; // Color of the pulse ring
    this.thickness = options.thickness || 3; // Stroke width of the ring
    this.ring = null; // The SVG element for the ring
    this.timeoutId = null; // ID for the scheduled effect timeout
    
    // Initialize the effect if active
    if (this.isActive) {
      this.init();
      this.scheduleNextPulse();
    }
  }

  // Initialize the SVG elements needed for the effect
  init() {
    // Create the ring element
    this.ring = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
    this.ring.classList.add('liquid-metal-ring');
    this.ring.setAttribute('cx', '50%');
    this.ring.setAttribute('cy', '100%'); // Start at the bottom
    this.ring.setAttribute('rx', '40%'); // Width of the ring
    this.ring.setAttribute('ry', '10%'); // Height of the ring (flattened for ripple effect)
    this.ring.setAttribute('fill', 'none');
    this.ring.setAttribute('stroke', this.color);
    this.ring.setAttribute('stroke-width', this.thickness);
    this.ring.setAttribute('opacity', '0'); // Initially invisible
    
    // Add glitch-like dash array for a fragmented look
    this.ring.setAttribute('stroke-dasharray', '10,5');
    
    // Append to the container (assumes container is an SVG or has an SVG child)
    if (this.container.tagName === 'svg') {
      this.container.appendChild(this.ring);
    } else {
      const svg = this.container.querySelector('svg');
      if (svg) {
        svg.appendChild(this.ring);
      }
    }
  }

  // Schedule the next pulse effect at a random interval between minInterval and maxInterval
  scheduleNextPulse() {
    if (!this.isActive) return;
    
    const interval = Math.random() * (this.maxInterval - this.minInterval) + this.minInterval;
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
    }
    this.timeoutId = setTimeout(() => {
      this.triggerPulse();
      this.scheduleNextPulse(); // Schedule the next one after this triggers
    }, interval);
  }

  // Trigger the liquid metal pulse effect
  triggerPulse() {
    if (!this.isActive || !this.ring) return;
    
    // Reset the ring to the bottom and make it visible
    this.ring.setAttribute('cy', '100%');
    this.ring.setAttribute('opacity', '0.8');
    
    // Add a glitchy flicker by rapidly changing dash offset
    let dashOffset = 0;
    const flickerInterval = setInterval(() => {
      dashOffset += 2;
      this.ring.setAttribute('stroke-dashoffset', dashOffset);
    }, 50);
    
    // Animate the ring moving upwards and fading out
    const startTime = Date.now();
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / this.duration, 1);
      
      // Move the ring up (cy from 100% to 0%)
      const cy = 100 - (progress * 100);
      this.ring.setAttribute('cy', `${cy}%`);
      
      // Fade out as it moves up
      const opacity = 0.8 * (1 - progress);
      this.ring.setAttribute('opacity', opacity);
      
      // Increase the ry for a stretching effect as it moves up
      const ry = 10 + (progress * 20);
      this.ring.setAttribute('ry', `${ry}%`);
      
      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        // Reset the ring when animation completes
        this.ring.setAttribute('opacity', '0');
        clearInterval(flickerInterval);
      }
    };
    
    requestAnimationFrame(animate);
  }

  // Enable or disable the effect
  setActive(isActive) {
    this.isActive = isActive;
    if (this.isActive) {
      if (!this.ring) {
        this.init();
      }
      this.scheduleNextPulse();
    } else {
      if (this.timeoutId) {
        clearTimeout(this.timeoutId);
        this.timeoutId = null;
      }
      if (this.ring) {
        this.ring.setAttribute('opacity', '0');
      }
    }
  }
}