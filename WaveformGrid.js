/**
 * WaveformGrid
 * Extends AWaves component to respond to audio data for music visualization
 */

import Emitter from './Emitter'
import { AudioVisualizer } from '../beats/frontend/audio/AudioVisualizer'

export class WaveformGrid {
  constructor(wavesElement, audioElement) {
    this.wavesElement = wavesElement
    this.audioElement = audioElement
    this.visualizer = null
    this.isActive = false
    this.intensity = 0
    this.beatIntensity = 0
    this.pulseTimeout = null

    // Audio features
    this.audioFeatures = {
      energy: 0,
      spectrum: [0, 0, 0, 0],
      bass: 0,
      lowMid: 0,
      highMid: 0,
      treble: 0,
      peakFrequency: 0,
      transientEnergy: 0,
    }

    // Configuration
    this.config = {
      beatDecay: 0.05, // How quickly beat effects fade
      energyInfluence: 0.5, // How much overall energy affects the grid
      bassInfluence: 1.0, // How much bass affects the grid
      waveAmplitude: 1.5, // Base wave amplitude multiplier
      cursorForce: 1.2, // Force multiplier for cursor effects
      pulseDuration: 300, // Duration of beat pulse effect in ms
      transientInfluence: 0.8, // How much transients affect the grid
      frequencyScale: 0.2, // Scaling factor for frequency-based effects
      colorShift: true, // Enable frequency-based color shifting
      dynamicDetail: true, // Enable detail level changes with energy
    }

    this.initialize()
  }

  /**
   * Set up audio visualizer and event listeners
   */
  initialize() {
    // Create audio visualizer
    this.visualizer = new AudioVisualizer(this.audioElement)

    // Listen for audio visualization events
    this.visualizer.on('frame', (data) => {
      if (!this.isActive) return

      // Update audio features
      this.audioFeatures = data

      // Decay beat intensity over time
      this.beatIntensity *= 1 - this.config.beatDecay
    })

    this.visualizer.on('beat', (features) => {
      if (!this.isActive) return

      // Pulse effect on beat
      this.beatIntensity = 1.0

      // Clear any existing timeout
      if (this.pulseTimeout) {
        clearTimeout(this.pulseTimeout)
      }

      // Trigger pulse effect on grid
      this.pulse(features.bass * 2)

      // Gradually return to normal
      this.pulseTimeout = setTimeout(() => {
        this.pulse(0)
      }, this.config.pulseDuration)
    })

    // Listen for playback state changes
    this.audioElement.addEventListener('play', () => {
      this.isActive = true
      this.visualizer.start()
    })

    this.audioElement.addEventListener('pause', () => {
      this.isActive = false
      this.visualizer.stop()
      this.resetGrid()
    })

    this.audioElement.addEventListener('ended', () => {
      this.isActive = false
      this.visualizer.stop()
      this.resetGrid()
    })

    // Connect to AWaves component's animation system
    Emitter.on('tick', this.updateGrid, this)
  }

  /**
   * Update grid based on audio data
   * This method will be called on each animation frame
   */
  updateGrid(time) {
    if (!this.wavesElement || !this.isActive) return

    // Calculate current intensity based on audio features
    this.intensity =
      this.audioFeatures.energy * this.config.energyInfluence +
      this.audioFeatures.bass * this.config.bassInfluence +
      this.audioFeatures.transientEnergy * this.config.transientInfluence +
      this.beatIntensity

    // Scale for reasonable values
    const scaledIntensity =
      Math.min(1, this.intensity) * this.config.waveAmplitude

    // Update AWaves custom properties
    this.wavesElement.style.setProperty(
      '--wave-amplitude',
      scaledIntensity.toFixed(2),
    )
    this.wavesElement.style.setProperty(
      '--beat-intensity',
      this.beatIntensity.toFixed(2),
    )

    // Different frequencies affect different aspects of the grid
    this.wavesElement.style.setProperty(
      '--bass-level',
      this.audioFeatures.bass.toFixed(2),
    )
    this.wavesElement.style.setProperty(
      '--mid-level',
      this.audioFeatures.lowMid.toFixed(2),
    )
    this.wavesElement.style.setProperty(
      '--high-level',
      this.audioFeatures.treble.toFixed(2),
    )
    this.wavesElement.style.setProperty(
      '--transient-level',
      this.audioFeatures.transientEnergy.toFixed(2),
    )

    // Advanced: frequency-based effects
    if (this.config.frequencyScale > 0) {
      // Normalize peak frequency to 0-1 range (20Hz - 20000Hz in log scale)
      const logMin = Math.log(20)
      const logMax = Math.log(20000)
      const logFreq = Math.log(
        Math.max(20, Math.min(20000, this.audioFeatures.peakFrequency || 1000)),
      )
      const normalizedFreq = (logFreq - logMin) / (logMax - logMin)

      this.wavesElement.style.setProperty(
        '--freq-normalized',
        normalizedFreq.toFixed(2),
      )

      // Map frequency to hue for color shifts if enabled
      if (this.config.colorShift) {
        // Map to hue (0-360)
        const hue = Math.round(normalizedFreq * 360)
        this.wavesElement.style.setProperty('--freq-hue', hue)
      }
    }

    // Dynamic detail based on energy if enabled
    if (this.config.dynamicDetail) {
      // More energy = more detailed grid
      const detailLevel = 0.5 + this.audioFeatures.energy * 0.5
      this.wavesElement.style.setProperty(
        '--detail-level',
        detailLevel.toFixed(2),
      )
    }
  }

  /**
   * Create a pulse effect on beat detection
   * @param intensity Intensity of the pulse effect
   */
  pulse(intensity) {
    if (!this.wavesElement) return

    // Add a class for CSS animations
    if (intensity > 0) {
      this.wavesElement.classList.add('is-pulsing')
      this.wavesElement.style.setProperty(
        '--pulse-intensity',
        intensity.toFixed(2),
      )
    } else {
      this.wavesElement.classList.remove('is-pulsing')
    }
  }

  /**
   * Reset the grid to its default state
   */
  resetGrid() {
    if (!this.wavesElement) return

    // Reset all audio-reactive properties
    this.wavesElement.style.removeProperty('--wave-amplitude')
    this.wavesElement.style.removeProperty('--beat-intensity')
    this.wavesElement.style.removeProperty('--bass-level')
    this.wavesElement.style.removeProperty('--mid-level')
    this.wavesElement.style.removeProperty('--high-level')
    this.wavesElement.style.removeProperty('--transient-level')
    this.wavesElement.style.removeProperty('--pulse-intensity')
    this.wavesElement.style.removeProperty('--freq-normalized')
    this.wavesElement.style.removeProperty('--freq-hue')
    this.wavesElement.style.removeProperty('--detail-level')
    this.wavesElement.classList.remove('is-pulsing')

    // Reset internal state
    this.intensity = 0
    this.beatIntensity = 0
    this.audioFeatures = {
      energy: 0,
      spectrum: [0, 0, 0, 0],
      bass: 0,
      lowMid: 0,
      highMid: 0,
      treble: 0,
      peakFrequency: 0,
      transientEnergy: 0,
    }
  }

  /**
   * Connect to a different audio element
   * @param audioElement New audio element to visualize
   */
  connectAudio(audioElement) {
    // Clean up previous connection
    this.isActive = false
    if (this.visualizer) {
      this.visualizer.stop()
    }

    // Set new audio element
    this.audioElement = audioElement

    // Reconnect visualizer
    if (this.visualizer) {
      this.visualizer.connectAudio(audioElement)
    } else {
      this.visualizer = new AudioVisualizer(audioElement)
    }

    // Reset grid
    this.resetGrid()
  }

  /**
   * Clean up resources
   */
  dispose() {
    if (this.visualizer) {
      this.visualizer.dispose()
    }

    if (this.pulseTimeout) {
      clearTimeout(this.pulseTimeout)
    }

    Emitter.off('tick', this.updateGrid, this)

    this.resetGrid()
    this.isActive = false
  }
}
