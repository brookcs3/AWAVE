/**
 * ColorMorph - Dynamic color animation utility
 * Animates between analogous colors in the red spectrum
 * with complementary secondary colors
 */

const gsap = window.gsap;

class ColorMorph {
  constructor(options = {}) {
    // Configuration
    this.config = {
      updateInterval: options.updateInterval || 5000, // Time between color changes (ms)
      transitionDuration: options.transitionDuration || 2.5, // Transition duration (s)
      hueRange: options.hueRange || [345, 15], // Hue range for primary (degrees)
      saturationRange: options.saturationRange || [75, 90], // Saturation range (%)
      lightnessRange: options.lightnessRange || [45, 55], // Lightness range (%)
      root: document.documentElement,
    }

    // Current color state
    this.currentColors = {
      primary: { h: 355, s: 85, l: 50 },
      secondary: { h: 175, s: 85, l: 5 }, // Complementary dark
      shadow: { h: 355, s: 85, l: 15 }, // Shadow version of primary
      white: { h: 355, s: 10, l: 95 }, // Tinted white
    }

    // Initialize
    this.init()
  }

  /**
   * Initialize the color animation
   */
  init() {
    // Set initial colors
    this.updateCSSVariables()

    // Start the animation loop
    this.startColorLoop()
  }

  /**
   * Start color animation loop
   */
  startColorLoop() {
    this.animateToNewColors()

    // Set interval for continuous animation
    setInterval(() => {
      this.animateToNewColors()
    }, this.config.updateInterval)
  }

  /**
   * Generate new target colors and animate to them
   */
  animateToNewColors() {
    const targetColors = this.generateTargetColors()

    // Animate from current to target colors
    gsap.to(this.currentColors.primary, {
      h: targetColors.primary.h,
      s: targetColors.primary.s,
      l: targetColors.primary.l,
      duration: this.config.transitionDuration,
      ease: 'sine.inOut',
      onUpdate: () => this.updateCSSVariables(),
    })

    gsap.to(this.currentColors.secondary, {
      h: targetColors.secondary.h,
      s: targetColors.secondary.s,
      l: targetColors.secondary.l,
      duration: this.config.transitionDuration,
      ease: 'sine.inOut',
    })

    gsap.to(this.currentColors.shadow, {
      h: targetColors.shadow.h,
      s: targetColors.shadow.s,
      l: targetColors.shadow.l,
      duration: this.config.transitionDuration,
      ease: 'sine.inOut',
    })

    gsap.to(this.currentColors.white, {
      h: targetColors.white.h,
      s: targetColors.white.s,
      l: targetColors.white.l,
      duration: this.config.transitionDuration,
      ease: 'sine.inOut',
    })
  }

  /**
   * Generate new target colors
   */
  generateTargetColors() {
    // Generate a new primary hue within the red analogous range
    const hueRange = this.config.hueRange
    const primaryHue = this.randomBetween(hueRange[0], hueRange[1])

    // Create complementary hue (opposite on color wheel)
    const complementaryHue = (primaryHue + 180) % 360

    // Generate saturation and lightness values
    const primarySat = this.randomBetween(
      this.config.saturationRange[0],
      this.config.saturationRange[1],
    )

    const primaryLight = this.randomBetween(
      this.config.lightnessRange[0],
      this.config.lightnessRange[1],
    )

    return {
      primary: {
        h: primaryHue,
        s: primarySat,
        l: primaryLight,
      },
      secondary: {
        h: complementaryHue,
        s: primarySat - 10,
        l: 8,
      },
      shadow: {
        h: primaryHue,
        s: primarySat,
        l: 25,
      },
      white: {
        h: primaryHue,
        s: 10,
        l: 95,
      },
    }
  }

  /**
   * Update CSS custom properties with current colors
   */
  updateCSSVariables() {
    const { primary, secondary, shadow, white } = this.currentColors

    // Convert HSL objects to CSS color strings
    const primaryColor = this.hslToHex(primary.h, primary.s, primary.l)
    const secondaryColor = this.hslToHex(secondary.h, secondary.s, secondary.l)
    const shadowColor = this.hslToHex(shadow.h, shadow.s, shadow.l)
    const whiteColor = this.hslToHex(white.h, white.s, white.l)

    // Update CSS variables
    this.config.root.style.setProperty('--color-primary', primaryColor)
    this.config.root.style.setProperty('--color-secondary', secondaryColor)
    this.config.root.style.setProperty('--color-shadow', shadowColor)
    this.config.root.style.setProperty('--color-white', whiteColor)
  }

  /**
   * Generate a random number between min and max
   */
  randomBetween(min, max) {
    return Math.floor(Math.random() * (max - min + 1) + min)
  }

  /**
   * Convert HSL values to hex color string
   */
  hslToHex(h, s, l) {
    l /= 100
    const a = (s * Math.min(l, 1 - l)) / 100
    const f = (n) => {
      const k = (n + h / 30) % 12
      const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1)
      return Math.round(255 * color)
        .toString(16)
        .padStart(2, '0')
    }
    return `#${f(0)}${f(8)}${f(4)}`
  }
}

export default ColorMorph
