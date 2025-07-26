# Architecture

```mermaid
graph TD
    A[User Interaction] --> B[AWaves Web Component]
    B --> C[Emitter Events]
    B --> D[LiquidMetalEffect]
    B --> E[InteractiveAudioControl]
    E --> F[Audio Effects (Pizzicato, XA Effects)]
    E --> G[Audio API]
```

The project centers on the **AWaves** custom element which draws an SVG grid of lines. Movement is driven by perlin noise and optional audio data. Other modules such as `InteractiveAudioControl` hook into this element to manipulate audio playback and apply effects.
