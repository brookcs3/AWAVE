# AWAVE

An experimental playground for web-based audio visualization and interaction. It explores combining interactive SVG "wave" grids with real-time audio analysis and effects.

## Overview

`AWaves.js` defines a custom element that renders an animated grid influenced by Perlin noise and user input. A second variant (`AWavesVariant2.js`) layers on more complex behaviours such as traveling peaks and an optional **Liquid Metal** pulse effect. Audio data is captured via the Web Audio API and fed into the animation.

The project also includes `InteractiveAudioControl.js` which maps grid interaction to audio effects via [Pizzicato](https://alemangui.github.io/pizzicato/) and a proof-of-concept `xa-effects.js` port of Librosa's audio utilities.

## Running

No build step is required. Serve the repository with any static server and open `index.html`.

```bash
python3 -m http.server
```

Then visit <http://localhost:8000>.

## File Structure

- `index.html` – Example page wiring everything together
- `AWaves.js` – Base animated grid component
- `AWavesVariant2.js` – Experimental variant with morphing and dance routines
- `LiquidMetalEffect.js` – SVG ring pulse inspired by *Terminator 2*
- `InteractiveAudioControl.js` – Maps pointer gestures to audio effects
- `styles/` – Sass sources compiled into `AWaves.css`
- `docs/architecture.md` – Architecture notes with a Mermaid diagram

## Highlight: LiquidMetalEffect

One notable module is `LiquidMetalEffect.js`. It periodically draws an SVG ellipse that rises and fades, adding a glitchy stroke dash pattern for a "liquid" feel:

```javascript
// Schedule the next pulse at random intervals
scheduleNextPulse() {
  const interval = Math.random() * (this.maxInterval - this.minInterval) + this.minInterval;
  this.timeoutId = setTimeout(() => {
    this.triggerPulse();
    this.scheduleNextPulse();
  }, interval);
}
```

The effect is lightweight yet adds dramatic atmosphere when the visualization is active.

## Status

This repository is experimental and not a polished product. Expect rough edges and unfinished features. The code may serve as a reference or starting point for further audiovisual experiments.
