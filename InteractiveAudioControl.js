/**
 * InteractiveAudioControl
 * 
 * Enables audio manipulation through wireframe grid interaction.
 * Creates a bidirectional relationship where:
 * 1. Audio affects grid visually through visualization
 * 2. Interacting with grid affects audio playback parameters
 * 
 * Integrates Pizzicato.js for advanced audio effects and xa-effects.js for audio processing like harmonic-percussive separation.
 */

export class InteractiveAudioControl {
  /**
   * Create a new interactive audio controller
   * @param {HTMLElement} gridElement The wireframe grid element (AWaves)
   * @param {HTMLAudioElement} audioElement The audio element to control
   * @param {AudioContext} audioContext Optional audio context
   */
  constructor(gridElement, audioElement, audioContext) {
    // Elements
    this.gridElement = gridElement;
    this.audioElement = audioElement;
    
    // Audio nodes
    this.context = audioContext || new (window.AudioContext || window.webkitAudioContext)();
    this.sourceNode = null;
    this.analyserNode = null;
    this.pitchNode = null; // Will be created when needed
    this.filterNode = null; // Will be created when needed
    this.delayNode = null; // Will be created when needed
    this.gainNode = null; // Will be created when needed
    
    // Pizzicato.js integration
    this.pizzicatoSound = null;
    this.pizzicatoEffects = {
      delay: null,
      distortion: null,
      flanger: null,
      reverb: null,
      tremolo: null
    };
    
    // xa-effects.js (LibrosaEffects) integration for audio processing
    this.librosaEffects = null;
    this.processedAudio = {
      harmonic: null,
      percussive: null
    };
    
    // Interaction state
    this.isInteracting = false;
    this.touchId = null;
    this.startX = 0;
    this.startY = 0;
    this.currentX = 0;
    this.currentY = 0;
    
    // Audio parameters
    this.originalPlaybackRate = 1.0;
    this.originalVolume = 1.0;
    
    // Effect parameters
    this.pitchRange = 0.5; // +/- 50% pitch shift
    this.filterRange = 10000; // Hz
    this.delayRange = 0.5; // seconds
    this.volumeRange = 0.5; // +/- 50% volume change
    
    // Zones: divide grid into regions with different effects, expanded for Pizzicato effects
    this.zones = [
      { name: 'pitch', x: [0, 0.25], y: [0, 1], color: 'rgba(255, 50, 50, 0.2)' },
      { name: 'filter', x: [0.25, 0.5], y: [0, 0.5], color: 'rgba(50, 255, 50, 0.2)' },
      { name: 'delay', x: [0.5, 0.75], y: [0, 0.5], color: 'rgba(50, 50, 255, 0.2)' },
      { name: 'volume', x: [0.75, 1], y: [0, 0.5], color: 'rgba(255, 255, 50, 0.2)' },
      { name: 'distortion', x: [0.25, 0.5], y: [0.5, 1], color: 'rgba(255, 50, 255, 0.2)' },
      { name: 'flanger', x: [0.5, 0.75], y: [0.5, 1], color: 'rgba(50, 255, 255, 0.2)' },
      { name: 'reverb', x: [0.75, 1], y: [0.5, 0.75], color: 'rgba(255, 150, 50, 0.2)' },
      { name: 'tremolo', x: [0.75, 1], y: [0.75, 1], color: 'rgba(150, 50, 255, 0.2)' }
    ];
    
    // Grid overlay for visualization
    this.overlay = null;
    
    // Initialize
    this.initialize();
  }
  
  /**
   * Set up audio nodes, Pizzicato and LibrosaEffects integration, and event listeners
   */
  initialize() {
    // Create audio processing chain
    this.setupAudioNodes();
    
    // Initialize Pizzicato.js sound and effects
    this.setupPizzicato();
    
    // Initialize LibrosaEffects for audio processing
    this.setupLibrosaEffects();
    
    // Create overlay for visualizing zones
    this.createOverlay();
    
    // Add event listeners
    this.bindEvents();
    
    // Add data attribute to grid
    this.gridElement.setAttribute('data-audio-interactive', 'true');
    
    // Add CSS class for styling
    this.gridElement.classList.add('audio-interactive');
  }
  
  /**
   * Set up Web Audio API nodes
   */
  setupAudioNodes() {
    try {
      // Create source node from audio element
      this.sourceNode = this.context.createMediaElementSource(this.audioElement);
      
      // Create analyser for visualization
      this.analyserNode = this.context.createAnalyser();
      this.analyserNode.fftSize = 2048;
      
      // Create gain node for volume control
      this.gainNode = this.context.createGain();
      this.gainNode.gain.value = 1.0;
      
      // Connect basic chain
      this.sourceNode.connect(this.analyserNode);
      this.analyserNode.connect(this.gainNode);
      this.gainNode.connect(this.context.destination);
      
      // Store original settings
      this.originalPlaybackRate = this.audioElement.playbackRate;
      this.originalVolume = this.audioElement.volume;
    } catch (error) {
      console.error("Error setting up audio nodes:", error);
    }
  }
  
  /**
   * Set up Pizzicato.js sound and effects
   */
  setupPizzicato() {
    try {
      // Ensure Pizzicato is available
      if (!window.Pizzicato) {
        console.error("Pizzicato.js is not loaded. Advanced audio effects will not be available.");
        return;
      }
      
      // Create a Pizzicato sound from the audio element
      this.pizzicatoSound = new Pizzicato.Sound({
        source: 'file',
        options: {
          path: this.audioElement.src,
          loop: this.audioElement.loop,
          volume: this.audioElement.volume
        }
      });
      
      // Initialize Pizzicato effects
      this.pizzicatoEffects.delay = new Pizzicato.Effects.Delay({
        feedback: 0.5,
        time: 0.3,
        mix: 0.5
      });
      
      this.pizzicatoEffects.distortion = new Pizzicato.Effects.Distortion({
        gain: 0.5
      });
      
      this.pizzicatoEffects.flanger = new Pizzicato.Effects.Flanger({
        time: 0.45,
        speed: 0.2,
        depth: 0.1,
        feedback: 0.1,
        mix: 0.5
      });
      
      this.pizzicatoEffects.reverb = new Pizzicato.Effects.Reverb({
        mix: 0.5,
        time: 0.01,
        decay: 0.01,
        reverse: false
      });
      
      this.pizzicatoEffects.tremolo = new Pizzicato.Effects.Tremolo({
        speed: 4,
        depth: 1,
        mix: 0.8
      });
      
      console.log("Pizzicato.js effects initialized for interactive audio control.");
    } catch (error) {
      console.error("Error setting up Pizzicato.js:", error);
    }
  }
  
  /**
   * Create a visual overlay to show interactive zones
   */
  createOverlay() {
    // Create overlay element
    this.overlay = document.createElement('div');
    this.overlay.className = 'audio-control-overlay';
    this.overlay.style.position = 'absolute';
    this.overlay.style.top = '0';
    this.overlay.style.left = '0';
    this.overlay.style.width = '100%';
    this.overlay.style.height = '100%';
    this.overlay.style.pointerEvents = 'none';
    this.overlay.style.zIndex = '5';
    this.overlay.style.opacity = '0';
    this.overlay.style.transition = 'opacity 0.3s ease';
    
    // Create zone indicators
    this.zones.forEach(zone => {
      const zoneEl = document.createElement('div');
      zoneEl.className = `zone zone-${zone.name}`;
      zoneEl.style.position = 'absolute';
      zoneEl.style.left = `${zone.x[0] * 100}%`;
      zoneEl.style.top = `${zone.y[0] * 100}%`;
      zoneEl.style.width = `${(zone.x[1] - zone.x[0]) * 100}%`;
      zoneEl.style.height = `${(zone.y[1] - zone.y[0]) * 100}%`;
      zoneEl.style.backgroundColor = zone.color;
      zoneEl.style.borderRadius = '8px';
      zoneEl.style.boxShadow = 'inset 0 0 10px rgba(0,0,0,0.2)';
      zoneEl.style.display = 'flex';
      zoneEl.style.alignItems = 'center';
      zoneEl.style.justifyContent = 'center';
      
      // Add label
      const label = document.createElement('span');
      label.textContent = zone.name.toUpperCase();
      label.style.color = 'rgba(0,0,0,0.7)';
      label.style.fontFamily = 'var(--font-family-fraktion, monospace)';
      label.style.fontSize = '14px';
      label.style.fontWeight = 'bold';
      label.style.textTransform = 'uppercase';
      label.style.letterSpacing = '1px';
      label.style.padding = '8px';
      label.style.borderRadius = '4px';
      label.style.backgroundColor = 'rgba(255,255,255,0.4)';
      
      zoneEl.appendChild(label);
      this.overlay.appendChild(zoneEl);
    });
    
    // Create cursor indicator
    const cursor = document.createElement('div');
    cursor.className = 'effect-cursor';
    cursor.style.position = 'absolute';
    cursor.style.width = '20px';
    cursor.style.height = '20px';
    cursor.style.borderRadius = '50%';
    cursor.style.backgroundColor = 'rgba(255,255,255,0.8)';
    cursor.style.boxShadow = '0 0 10px rgba(255,255,255,0.8)';
    cursor.style.transform = 'translate(-50%, -50%)';
    cursor.style.pointerEvents = 'none';
    cursor.style.display = 'none';
    this.cursor = cursor;
    
    this.overlay.appendChild(cursor);
    
    // Position the overlay
    const gridStyles = window.getComputedStyle(this.gridElement);
    if (gridStyles.position === 'static') {
      this.gridElement.style.position = 'relative';
    }
    
    // Add overlay to grid
    this.gridElement.appendChild(this.overlay);
  }
  
  /**
   * Bind event listeners to grid and audio element
   */
  bindEvents() {
    // Mouse events
    this.gridElement.addEventListener('mousedown', this.onInteractionStart.bind(this));
    window.addEventListener('mousemove', this.onInteractionMove.bind(this));
    window.addEventListener('mouseup', this.onInteractionEnd.bind(this));
    
    // Touch events
    this.gridElement.addEventListener('touchstart', this.onInteractionStart.bind(this));
    window.addEventListener('touchmove', this.onInteractionMove.bind(this));
    window.addEventListener('touchend', this.onInteractionEnd.bind(this));
    window.addEventListener('touchcancel', this.onInteractionEnd.bind(this));
    
    // Audio events
    this.audioElement.addEventListener('play', this.onAudioPlay.bind(this));
    this.audioElement.addEventListener('pause', this.onAudioPause.bind(this));
    this.audioElement.addEventListener('ended', this.onAudioEnded.bind(this));
    
    // Info toggle - show/hide overlay on double-click
    this.gridElement.addEventListener('dblclick', this.toggleOverlay.bind(this));
    
    // Keyboard events - press Shift to show overlay temporarily
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Shift') this.showOverlay();
    });
    window.addEventListener('keyup', (e) => {
      if (e.key === 'Shift') this.hideOverlay();
    });
  }
  
  /**
   * Handle start of interaction (mouse down or touch start)
   * @param {Event} e Mouse or touch event
   */
  onInteractionStart(e) {
    e.preventDefault();
    
    // Auto-play audio if not playing
    if (this.audioElement.paused) {
      this.audioElement.play().catch(err => console.error("Failed to play audio:", err));
    }
    
    this.isInteracting = true;
    
    // Get initial position
    if (e.type === 'touchstart') {
      const touch = e.touches[0];
      this.touchId = touch.identifier;
      this.startX = touch.clientX;
      this.startY = touch.clientY;
      this.currentX = touch.clientX;
      this.currentY = touch.clientY;
    } else {
      this.startX = e.clientX;
      this.startY = e.clientY;
      this.currentX = e.clientX;
      this.currentY = e.clientY;
    }
    
    // Show effects while interacting
    this.showOverlay(0.3);
    
    // Show cursor indicator
    this.updateCursorPosition();
    this.cursor.style.display = 'block';
    
    // Add active class
    this.gridElement.classList.add('audio-interacting');
    
    // Start hold detection for repeated delay effect
    this.startHoldDetection();
  }
  
  /**
   * Handle movement during interaction
   * @param {Event} e Mouse or touch event
   */
  onInteractionMove(e) {
    if (!this.isInteracting) return;
    
    // Get current position
    if (e.type === 'touchmove') {
      // Find the touch that started the interaction
      const touch = Array.from(e.touches).find(t => t.identifier === this.touchId);
      if (!touch) return;
      
      this.currentX = touch.clientX;
      this.currentY = touch.clientY;
    } else {
      this.currentX = e.clientX;
      this.currentY = e.clientY;
    }
    
    // Update cursor position
    this.updateCursorPosition();
    
    // Apply audio effects based on position
    this.applyEffects();
    
    // Detect pulling gesture for pitch+echo effect
    this.detectPullingGesture();
  }
  
  /**
   * Handle end of interaction
   */
  onInteractionEnd(e) {
    if (!this.isInteracting) return;
    
    // For touch events, make sure we're ending the right touch
    if (e.type === 'touchend' || e.type === 'touchcancel') {
      const changedTouch = Array.from(e.changedTouches).find(t => t.identifier === this.touchId);
      if (!changedTouch) return;
    }
    
    this.isInteracting = false;
    this.touchId = null;
    
    // Reset audio parameters
    this.resetEffects();
    
    // Clear hold detection
    this.clearHoldDetection();
    
    // Hide effects overlay after a delay
    setTimeout(() => {
      if (!this.isInteracting) {
        this.hideOverlay();
      }
    }, 1000);
    
    // Hide cursor indicator
    this.cursor.style.display = 'none';
    
    // Remove active class
    this.gridElement.classList.remove('audio-interacting');
  }
  
  /**
   * Update the position of the cursor indicator
   */
  updateCursorPosition() {
    // Convert page coordinates to element-relative coordinates
    const rect = this.gridElement.getBoundingClientRect();
    const x = this.currentX - rect.left;
    const y = this.currentY - rect.top;
    
    // Update cursor position
    this.cursor.style.left = `${x}px`;
    this.cursor.style.top = `${y}px`;
    
    // Highlight active zone
    const activeZone = this.getActiveZone(x / rect.width, y / rect.height);
    
    // Reset all zones
    const zoneElements = this.overlay.querySelectorAll('.zone');
    zoneElements.forEach(zone => {
      zone.style.opacity = '0.5';
      zone.style.transform = 'scale(1)';
    });
    
    // Highlight active zone
    if (activeZone) {
      const zoneElement = this.overlay.querySelector(`.zone-${activeZone.name}`);
      if (zoneElement) {
        zoneElement.style.opacity = '0.8';
        zoneElement.style.transform = 'scale(1.05)';
      }
      
      // Update cursor color to match active zone
      this.cursor.style.backgroundColor = activeZone.color.replace('0.2', '0.8');
      this.cursor.style.boxShadow = `0 0 20px ${activeZone.color.replace('0.2', '0.6')}`;
    }
  }
  
  /**
   * Determine which effect zone the cursor is in
   * @param {number} normalizedX X position (0-1)
   * @param {number} normalizedY Y position (0-1)
   * @returns {Object|null} The active zone or null if outside all zones
   */
  getActiveZone(normalizedX, normalizedY) {
    return this.zones.find(zone => {
      return normalizedX >= zone.x[0] && normalizedX <= zone.x[1] &&
             normalizedY >= zone.y[0] && normalizedY <= zone.y[1];
    }) || null;
  }
  
  /**
   * Apply audio effects based on cursor position
   */
  applyEffects() {
    // Calculate normalized position within the grid
    const rect = this.gridElement.getBoundingClientRect();
    const normalizedX = (this.currentX - rect.left) / rect.width;
    const normalizedY = (this.currentY - rect.top) / rect.height;
    
    // Get active zone
    const activeZone = this.getActiveZone(normalizedX, normalizedY);
    if (!activeZone) return;
    
    // Calculate intensity based on distance from center of zone
    const zoneWidth = activeZone.x[1] - activeZone.x[0];
    const zoneHeight = activeZone.y[1] - activeZone.y[0];
    const zoneCenterX = activeZone.x[0] + zoneWidth / 2;
    const zoneCenterY = activeZone.y[0] + zoneHeight / 2;
    
    // Calculate normalized position within zone (0 at center, 1 at edge)
    const zoneX = (normalizedX - zoneCenterX) / (zoneWidth / 2);
    const zoneY = (normalizedY - zoneCenterY) / (zoneHeight / 2);
    
    // Apply different effects based on active zone
    switch(activeZone.name) {
      case 'pitch':
        this.applyPitchEffect(zoneX, zoneY);
        break;
      case 'filter':
        this.applyFilterEffect(zoneX, zoneY);
        break;
      case 'delay':
        this.applyDelayEffect(zoneX, zoneY);
        break;
      case 'volume':
        this.applyVolumeEffect(zoneX, zoneY);
        break;
      case 'distortion':
        this.applyDistortionEffect(zoneX, zoneY);
        break;
      case 'flanger':
        this.applyFlangerEffect(zoneX, zoneY);
        break;
      case 'reverb':
        this.applyReverbEffect(zoneX, zoneY);
        break;
      case 'tremolo':
        this.applyTremoloEffect(zoneX, zoneY);
        break;
    }
  }
  
  /**
   * Apply pitch shift effect
   * @param {number} x Normalized X position in zone (-1 to 1)
   * @param {number} y Normalized Y position in zone (-1 to 1)
   */
  applyPitchEffect(x, y) {
    // Use Y position for pitch control (higher = higher pitch)
    const pitchShift = -y * this.pitchRange;
    
    // Apply pitch shift using playbackRate
    // This is a simple approach that changes both pitch and speed
    const newRate = this.originalPlaybackRate * (1 + pitchShift);
    this.audioElement.playbackRate = Math.max(0.25, Math.min(2.0, newRate));
    
    // Update visualization
    this.cursor.textContent = `${(pitchShift * 100).toFixed(0)}%`;
  }
  
  /**
   * Apply filter effect
   * @param {number} x Normalized X position in zone (-1 to 1)
   * @param {number} y Normalized Y position in zone (-1 to 1)
   */
  applyFilterEffect(x, y) {
    // Create filter node if it doesn't exist
    if (!this.filterNode) {
      this.filterNode = this.context.createBiquadFilter();
      this.filterNode.type = 'lowpass';
      this.filterNode.Q.value = 1;
      
      // Disconnect and reconnect nodes to add filter
      this.analyserNode.disconnect();
      this.analyserNode.connect(this.filterNode);
      this.filterNode.connect(this.gainNode);
    }
    
    // Use X position for filter type (left = lowpass, right = highpass)
    if (x < -0.33) {
      this.filterNode.type = 'lowpass';
    } else if (x > 0.33) {
      this.filterNode.type = 'highpass';
    } else {
      this.filterNode.type = 'bandpass';
    }
    
    // Use Y position for frequency cutoff (higher = higher frequency)
    // Map from 100Hz to 20000Hz exponentially
    const minFreq = 100;
    const maxFreq = 20000;
    const normalizedY = (1 - (-y + 1) / 2); // Convert from -1,1 to 0,1 and invert
    const frequency = minFreq * Math.pow(maxFreq / minFreq, normalizedY);
    
    // Apply filter
    this.filterNode.frequency.setValueAtTime(frequency, this.context.currentTime);
    
    // Update visualization
    this.cursor.textContent = `${Math.round(frequency)}Hz`;
  }
  
  /**
   * Apply delay effect
   * @param {number} x Normalized X position in zone (-1 to 1)
   * @param {number} y Normalized Y position in zone (-1 to 1)
   */
  applyDelayEffect(x, y) {
    if (!this.pizzicatoEffects.delay) return;
    
    // Use Y position for delay time (higher = shorter delay)
    const delayTime = (1 - ((-y + 1) / 2)) * this.delayRange;
    
    // Use X position for feedback amount (right = more feedback)
    const feedbackAmount = ((x + 1) / 2) * 0.7; // Max 0.7 to prevent infinite feedback
    
    // Apply delay settings using Pizzicato
    this.pizzicatoEffects.delay.time = delayTime;
    this.pizzicatoEffects.delay.feedback = feedbackAmount;
    
    // Ensure effect is added to sound if not already
    if (!this.pizzicatoSound.effects.includes(this.pizzicatoEffects.delay)) {
      this.pizzicatoSound.addEffect(this.pizzicatoEffects.delay);
    }
    
    // Update visualization
    this.cursor.textContent = `${(delayTime * 1000).toFixed(0)}ms`;
  }
  
  /**
   * Apply volume effect
   * @param {number} x Normalized X position in zone (-1 to 1)
   * @param {number} y Normalized Y position in zone (-1 to 1)
   */
  applyVolumeEffect(x, y) {
    // Use Y position for volume control (higher = louder)
    const volumeShift = -y * this.volumeRange;
    
    // Apply volume change
    const newVolume = this.originalVolume * (1 + volumeShift);
    this.audioElement.volume = Math.max(0, Math.min(1, newVolume));
    
    // Update visualization
    this.cursor.textContent = `${(volumeShift * 100).toFixed(0)}%`;
  }
  
  /**
   * Apply distortion effect using Pizzicato.js
   * @param {number} x Normalized X position in zone (-1 to 1)
   * @param {number} y Normalized Y position in zone (-1 to 1)
   */
  applyDistortionEffect(x, y) {
    if (!this.pizzicatoEffects.distortion) return;
    
    // Use Y position for distortion gain (higher = more distortion)
    const gain = ((-y + 1) / 2) * 1.0; // Range from 0 to 1
    
    // Apply distortion effect
    this.pizzicatoEffects.distortion.gain = gain;
    
    // Ensure effect is added to sound if not already
    if (!this.pizzicatoSound.effects.includes(this.pizzicatoEffects.distortion)) {
      this.pizzicatoSound.addEffect(this.pizzicatoEffects.distortion);
    }
    
    // Update visualization
    this.cursor.textContent = `Dist: ${(gain * 100).toFixed(0)}%`;
  }
  
  /**
   * Apply flanger effect using Pizzicato.js
   * @param {number} x Normalized X position in zone (-1 to 1)
   * @param {number} y Normalized Y position in zone (-1 to 1)
   */
  applyFlangerEffect(x, y) {
    if (!this.pizzicatoEffects.flanger) return;
    
    // Use Y position for flanger speed (higher = faster)
    const speed = ((-y + 1) / 2) * 1.0; // Range from 0 to 1
    
    // Use X position for flanger feedback (right = more feedback)
    const feedback = ((x + 1) / 2) * 0.8; // Max 0.8
    
    // Apply flanger effect
    this.pizzicatoEffects.flanger.speed = speed;
    this.pizzicatoEffects.flanger.feedback = feedback;
    
    // Ensure effect is added to sound if not already
    if (!this.pizzicatoSound.effects.includes(this.pizzicatoEffects.flanger)) {
      this.pizzicatoSound.addEffect(this.pizzicatoEffects.flanger);
    }
    
    // Update visualization
    this.cursor.textContent = `Flang: ${(speed * 100).toFixed(0)}%`;
  }
  
  /**
   * Apply reverb effect using Pizzicato.js
   * @param {number} x Normalized X position in zone (-1 to 1)
   * @param {number} y Normalized Y position in zone (-1 to 1)
   */
  applyReverbEffect(x, y) {
    if (!this.pizzicatoEffects.reverb) return;
    
    // Use Y position for reverb time (higher = longer)
    const time = ((-y + 1) / 2) * 0.5; // Range from 0 to 0.5
    
    // Use X position for reverb decay (right = more decay)
    const decay = ((x + 1) / 2) * 0.5; // Range from 0 to 0.5
    
    // Apply reverb effect
    this.pizzicatoEffects.reverb.time = time;
    this.pizzicatoEffects.reverb.decay = decay;
    
    // Ensure effect is added to sound if not already
    if (!this.pizzicatoSound.effects.includes(this.pizzicatoEffects.reverb)) {
      this.pizzicatoSound.addEffect(this.pizzicatoEffects.reverb);
    }
    
    // Update visualization
    this.cursor.textContent = `Rev: ${(time * 100).toFixed(0)}%`;
  }
  
  /**
   * Apply tremolo effect using Pizzicato.js
   * @param {number} x Normalized X position in zone (-1 to 1)
   * @param {number} y Normalized Y position in zone (-1 to 1)
   */
  applyTremoloEffect(x, y) {
    if (!this.pizzicatoEffects.tremolo) return;
    
    // Use Y position for tremolo speed (higher = faster)
    const speed = ((-y + 1) / 2) * 10; // Range from 0 to 10
    
    // Use X position for tremolo depth (right = deeper)
    const depth = ((x + 1) / 2) * 1.0; // Range from 0 to 1
    
    // Apply tremolo effect
    this.pizzicatoEffects.tremolo.speed = speed;
    this.pizzicatoEffects.tremolo.depth = depth;
    
    // Ensure effect is added to sound if not already
    if (!this.pizzicatoSound.effects.includes(this.pizzicatoEffects.tremolo)) {
      this.pizzicatoSound.addEffect(this.pizzicatoEffects.tremolo);
    }
    
    // Update visualization
    this.cursor.textContent = `Trem: ${(speed).toFixed(1)}Hz`;
  }
  
  /**
   * Reset all effects to their original values
   */
  resetEffects() {
    // Reset playback rate
    this.audioElement.playbackRate = this.originalPlaybackRate;
    
    // Reset volume
    this.audioElement.volume = this.originalVolume;
    
    // Reset filter if it exists
    if (this.filterNode) {
      this.filterNode.frequency.setValueAtTime(20000, this.context.currentTime);
    }
    
    // Reset Pizzicato effects if they exist
    if (this.pizzicatoSound) {
      // Remove all effects from the sound
      for (const effectName in this.pizzicatoEffects) {
        const effect = this.pizzicatoEffects[effectName];
        if (effect && this.pizzicatoSound.effects.includes(effect)) {
          this.pizzicatoSound.removeEffect(effect);
        }
      }
    }
  }
  
  /**
   * Set up LibrosaEffects for advanced audio processing
   */
  setupLibrosaEffects() {
    try {
      // Ensure LibrosaEffects is available
      if (!window.LibrosaEffects) {
        console.error("xa-effects.js is not loaded. Advanced audio processing will not be available.");
        return;
      }
      
      // Initialize LibrosaEffects
      this.librosaEffects = new LibrosaEffects();
      this.librosaEffects.init();
      
      // Process audio for separation if source is available
      if (this.audioElement.src) {
        this.processAudioForSeparation();
      }
      
      console.log("LibrosaEffects initialized for audio processing.");
    } catch (error) {
      console.error("Error setting up LibrosaEffects:", error);
    }
  }
  
  /**
   * Process audio to separate harmonic and percussive components
   */
  async processAudioForSeparation() {
    if (!this.librosaEffects) return;
    
    try {
      // Load audio data
      const audioData = await this.librosaEffects.loadAudio(this.audioElement.src);
      
      // Perform harmonic-percussive source separation
      const { harmonic, percussive } = this.librosaEffects.hpss(audioData.data);
      
      // Store processed components
      this.processedAudio.harmonic = harmonic;
      this.processedAudio.percussive = percussive;
      
      console.log("Audio processed for harmonic and percussive separation.");
      
      // Emit event to notify visualizer of processed audio data availability
      // This can be used to drive different animations based on audio components
      window.dispatchEvent(new CustomEvent('audioProcessed', {
        detail: {
          harmonic: harmonic,
          percussive: percussive,
          sampleRate: audioData.sampleRate
        }
      }));
    } catch (error) {
      console.error("Error processing audio for separation:", error);
    }
  }
  
  /**
   * Handle audio play event
   */
  onAudioPlay() {
    // Resume audio context if suspended
    if (this.context.state === 'suspended') {
      this.context.resume();
    }
    
    // Process audio for separation if not already done
    if (!this.processedAudio.harmonic && this.librosaEffects) {
      this.processAudioForSeparation();
    }
    
    // Ensure Pizzicato sound is playing
    if (this.pizzicatoSound && !this.pizzicatoSound.playing) {
      this.pizzicatoSound.play();
    }
    
    // Briefly show overlay to indicate interactivity
    this.showOverlay(0.5);
    setTimeout(() => {
      if (!this.isInteracting) {
        this.hideOverlay();
      }
    }, 2000);
  }
  
  /**
   * Handle audio pause event
   */
  onAudioPause() {
    // If we were interacting, stop
    if (this.isInteracting) {
      this.onInteractionEnd({ type: 'pause' });
    }
    
    // Pause Pizzicato sound if playing
    if (this.pizzicatoSound && this.pizzicatoSound.playing) {
      this.pizzicatoSound.pause();
    }
  }
  
  /**
   * Handle audio ended event
   */
  onAudioEnded() {
    // If we were interacting, stop
    if (this.isInteracting) {
      this.onInteractionEnd({ type: 'ended' });
    }
    
    // Stop Pizzicato sound if playing
    if (this.pizzicatoSound && this.pizzicatoSound.playing) {
      this.pizzicatoSound.stop();
    }
  }
  
  /**
   * Show the effects overlay
   * @param {number} opacity Opacity value (0-1)
   */
  showOverlay(opacity = 0.6) {
    this.overlay.style.opacity = opacity.toString();
  }
  
  /**
   * Hide the effects overlay
   */
  hideOverlay() {
    this.overlay.style.opacity = '0';
  }
  
  /**
   * Toggle overlay visibility
   */
  toggleOverlay() {
    if (this.overlay.style.opacity === '0') {
      this.showOverlay();
    } else {
      this.hideOverlay();
    }
  }
  
  /**
   * Start hold detection for repeated delay effect
   */
  startHoldDetection() {
    // Clear any existing hold timer
    if (this.holdTimer) {
      clearTimeout(this.holdTimer);
    }
    
    // Set a timer to detect holding for repeated delay effect
    this.holdTimer = setTimeout(() => {
      if (this.isInteracting) {
        this.applyHoldEffect();
      }
    }, 1000); // 1 second hold to trigger effect
  }
  
  /**
   * Clear hold detection timer
   */
  clearHoldDetection() {
    if (this.holdTimer) {
      clearTimeout(this.holdTimer);
      this.holdTimer = null;
    }
  }
  
  /**
   * Apply hold effect (slow down and repeated delay)
   */
  applyHoldEffect() {
    if (!this.pizzicatoEffects.delay) return;
    
    // Slow down playback rate
    this.audioElement.playbackRate = Math.max(0.25, this.originalPlaybackRate * 0.5);
    
    // Apply repeated delay effect
    this.pizzicatoEffects.delay.time = 0.3;
    this.pizzicatoEffects.delay.feedback = 0.6;
    this.pizzicatoEffects.delay.mix = 0.5;
    
    // Ensure effect is added to sound if not already
    if (!this.pizzicatoSound.effects.includes(this.pizzicatoEffects.delay)) {
      this.pizzicatoSound.addEffect(this.pizzicatoEffects.delay);
    }
    
    // Update visualization
    this.cursor.textContent = `Hold: Slow+Delay`;
  }
  
  /**
   * Detect pulling gesture for pitch+echo effect
   */
  detectPullingGesture() {
    // Calculate distance moved from start point
    const distanceX = this.currentX - this.startX;
    const distanceY = this.currentY - this.startY;
    const distance = Math.sqrt(distanceX * distanceX + distanceY * distanceY);
    
    // If significant movement, consider it a pull
    if (distance > 50) { // Arbitrary threshold for pull detection
      this.applyPullingEffect(distanceY);
    }
  }
  
  /**
   * Apply pulling effect (pitch + echo)
   * @param {number} distanceY Vertical distance moved for pitch adjustment
   */
  applyPullingEffect(distanceY) {
    if (!this.pizzicatoEffects.delay) return;
    
    // Adjust pitch based on vertical movement (up = higher pitch)
    const pitchShift = -(distanceY / 200) * this.pitchRange; // Scale based on movement
    const newRate = this.originalPlaybackRate * (1 + pitchShift);
    this.audioElement.playbackRate = Math.max(0.25, Math.min(2.0, newRate));
    
    // Apply echo effect
    this.pizzicatoEffects.delay.time = 0.4;
    this.pizzicatoEffects.delay.feedback = 0.5;
    this.pizzicatoEffects.delay.mix = 0.4;
    
    // Ensure effect is added to sound if not already
    if (!this.pizzicatoSound.effects.includes(this.pizzicatoEffects.delay)) {
      this.pizzicatoSound.addEffect(this.pizzicatoEffects.delay);
    }
    
    // Update visualization
    this.cursor.textContent = `Pull: Pitch+Echo`;
  }
  
  /**
   * Clean up resources
   */
  dispose() {
    // Remove event listeners
    this.gridElement.removeEventListener('mousedown', this.onInteractionStart);
    window.removeEventListener('mousemove', this.onInteractionMove);
    window.removeEventListener('mouseup', this.onInteractionEnd);
    
    this.gridElement.removeEventListener('touchstart', this.onInteractionStart);
    window.removeEventListener('touchmove', this.onInteractionMove);
    window.removeEventListener('touchend', this.onInteractionEnd);
    window.removeEventListener('touchcancel', this.onInteractionEnd);
    
    this.audioElement.removeEventListener('play', this.onAudioPlay);
    this.audioElement.removeEventListener('pause', this.onAudioPause);
    this.audioElement.removeEventListener('ended', this.onAudioEnded);
    
    // Reset audio parameters
    this.resetEffects();
    
    // Clear any hold timers
    this.clearHoldDetection();
    
    // Remove overlay
    if (this.overlay && this.overlay.parentNode) {
      this.overlay.parentNode.removeChild(this.overlay);
    }
    
    // Remove CSS classes
    this.gridElement.classList.remove('audio-interactive', 'audio-interacting');
    this.gridElement.removeAttribute('data-audio-interactive');
    
    // Disconnect audio nodes
    try {
      if (this.filterNode) {
        this.filterNode.disconnect();
      }
      if (this.delayNode) {
        this.delayNode.disconnect();
      }
      if (this.feedbackNode) {
        this.feedbackNode.disconnect();
      }
      
      this.analyserNode.disconnect();
      this.sourceNode.disconnect();
      
      // Reconnect source directly to destination
      this.sourceNode.connect(this.context.destination);
    } catch (error) {
      console.error("Error disconnecting audio nodes:", error);
    }
  }
}