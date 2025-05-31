class Emitter {
  static events = {};

  // Register an event handler
  static on(name, callback, context, once = false) {
    if (!this.events[name]) {
      this.events[name] = [];
    }

    let exists = false;
    this.events[name].forEach((object) => {
      if (object.cb === callback && object.context === context) {
        exists = true;
      }
    });
    if (exists) {
      return;
    }

    this.events[name].push({
      cb: callback,
      context: context,
      once: once
    });
  }

  // Register a one-time event handler
  static once(name, callback, context) {
    this.on(name, callback, context, true);
  }

  // Emit an event
  static emit(name, ...args) {
    if (this.events[name]) {
      this.events[name].forEach((object, index) => {
        object.cb.apply(object.context, args);
        if (object.once) {
          delete this.events[name][index];
        }
      });
    }
  }

  // Remove an event handler
  static off(name, callback, context) {
    if (this.events[name]) {
      this.events[name].forEach((object, index) => {
        if (object.cb === callback && object.context === context) {
          delete this.events[name][index];
        }
      });
    }
  }
}

// Make Emitter globally available
export default Emitter;

