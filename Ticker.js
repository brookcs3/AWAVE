import Emitter from './Emitter.js';

class Ticker {
  static isRunning = false;
  static callbacks = [];
  static nextTickCallbacks = [];
  static lastTime = 0;

  /**
   * Initialize ticker
   */
  static init() {
    if (this.isRunning) return;
    
    this.isRunning = true;
    this.lastTime = performance.now();
    
    requestAnimationFrame(this.tick.bind(this));
  }

  /**
   * Tick handler
   */
  static tick(time) {
    const delta = time - this.lastTime;
    this.lastTime = time;

    // Process next tick callbacks
    if (this.nextTickCallbacks.length > 0) {
      const callbacks = [...this.nextTickCallbacks];
      this.nextTickCallbacks = [];
      
      callbacks.forEach(({ callback, context }) => {
        callback.call(context || window);
      });
    }

    // Emit tick event
    Emitter.emit('tick', time, delta);
    
    // Continue animation loop
    requestAnimationFrame(this.tick.bind(this));
  }

  /**
   * Add callback to next tick
   */
  static nextTick(callback, context) {
    if (!callback) return;
    
    this.nextTickCallbacks.push({ callback, context });
    
    if (!this.isRunning) {
      this.init();
    }
  }
}

export default Ticker;