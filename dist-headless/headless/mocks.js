"use strict";
/**
 * Browser API mocks for running osrs-sdk in Node.js headless mode.
 * This file must be imported BEFORE any osrs-sdk imports.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.installMocks = installMocks;
// Mock OffscreenCanvas
class MockOffscreenCanvas {
    constructor(width, height) {
        this.width = width;
        this.height = height;
    }
    getContext(_type) {
        return new MockCanvasRenderingContext2D();
    }
    transferToImageBitmap() {
        return {};
    }
}
// Mock CanvasRenderingContext2D
class MockCanvasRenderingContext2D {
    constructor() {
        this.fillStyle = '';
        this.strokeStyle = '';
        this.lineWidth = 1;
        this.font = '';
        this.textAlign = 'left';
        this.textBaseline = 'top';
        this.globalAlpha = 1;
        this.globalCompositeOperation = 'source-over';
    }
    fillRect(_x, _y, _w, _h) { }
    strokeRect(_x, _y, _w, _h) { }
    clearRect(_x, _y, _w, _h) { }
    fillText(_text, _x, _y) { }
    strokeText(_text, _x, _y) { }
    measureText(_text) {
        return { width: 0 };
    }
    beginPath() { }
    closePath() { }
    moveTo(_x, _y) { }
    lineTo(_x, _y) { }
    arc(_x, _y, _r, _start, _end) { }
    fill() { }
    stroke() { }
    save() { }
    restore() { }
    translate(_x, _y) { }
    rotate(_angle) { }
    scale(_x, _y) { }
    drawImage(..._args) { }
    createLinearGradient(_x0, _y0, _x1, _y1) {
        return new MockGradient();
    }
    createRadialGradient(_x0, _y0, _r0, _x1, _y1, _r1) {
        return new MockGradient();
    }
    setTransform(..._args) { }
    resetTransform() { }
    clip() { }
    rect(_x, _y, _w, _h) { }
    ellipse(..._args) { }
    quadraticCurveTo(_cpx, _cpy, _x, _y) { }
    bezierCurveTo(_cp1x, _cp1y, _cp2x, _cp2y, _x, _y) { }
    arcTo(_x1, _y1, _x2, _y2, _radius) { }
    isPointInPath(_x, _y) {
        return false;
    }
    getImageData(_sx, _sy, _sw, _sh) {
        return { data: new Uint8ClampedArray(0), width: 0, height: 0 };
    }
    putImageData(_imageData, _dx, _dy) { }
}
class MockGradient {
    addColorStop(_offset, _color) { }
}
// Mock HTMLCanvasElement
class MockHTMLCanvasElement {
    constructor() {
        this.width = 800;
        this.height = 600;
    }
    getContext(_type) {
        return new MockCanvasRenderingContext2D();
    }
    toDataURL() {
        return '';
    }
    toBlob(_callback) {
        _callback(null);
    }
}
// Mock HTMLImageElement
class MockHTMLImageElement {
    constructor() {
        this.src = '';
        this.width = 32;
        this.height = 32;
        this.complete = true;
        this.onload = null;
        this.onerror = null;
        this.listeners = new Map();
        // Simulate async load completion
        setTimeout(() => {
            this.complete = true;
            if (this.onload)
                this.onload();
            this.dispatchEvent('load');
        }, 0);
    }
    addEventListener(event, handler) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(handler);
    }
    removeEventListener(event, handler) {
        const handlers = this.listeners.get(event);
        if (handlers) {
            const index = handlers.indexOf(handler);
            if (index >= 0)
                handlers.splice(index, 1);
        }
    }
    dispatchEvent(event) {
        const handlers = this.listeners.get(event);
        if (handlers) {
            handlers.forEach(h => h());
        }
    }
}
// Mock Audio
class MockAudio {
    constructor() {
        this.src = '';
        this.volume = 1;
        this.currentTime = 0;
        this.paused = true;
    }
    play() {
        return Promise.resolve();
    }
    pause() { }
    load() { }
    cloneNode() {
        return new MockAudio();
    }
}
// Mock document
const mockDocument = {
    createElement(tagName) {
        if (tagName === 'canvas')
            return new MockHTMLCanvasElement();
        if (tagName === 'img')
            return new MockHTMLImageElement();
        if (tagName === 'audio')
            return new MockAudio();
        return {};
    },
    getElementById(_id) {
        return null;
    },
    querySelector(_selector) {
        return null;
    },
    querySelectorAll(_selector) {
        return [];
    },
    body: {
        appendChild(_child) { },
        removeChild(_child) { },
    },
    addEventListener(_event, _handler) { },
    removeEventListener(_event, _handler) { },
};
// Mock localStorage with proper storage
const mockLocalStorageData = {};
const mockLocalStorage = {
    getItem(key) {
        return mockLocalStorageData[key] ?? null;
    },
    setItem(key, value) {
        mockLocalStorageData[key] = value;
    },
    removeItem(key) {
        delete mockLocalStorageData[key];
    },
    clear() {
        Object.keys(mockLocalStorageData).forEach(k => delete mockLocalStorageData[k]);
    },
};
// Mock window
const mockWindow = {
    requestAnimationFrame(_callback) {
        // Don't actually schedule - headless mode uses manual ticking
        return 0;
    },
    cancelAnimationFrame(_id) { },
    addEventListener(_event, _handler) { },
    removeEventListener(_event, _handler) { },
    innerWidth: 800,
    innerHeight: 600,
    devicePixelRatio: 1,
    localStorage: mockLocalStorage,
    performance: {
        now() {
            return Date.now();
        },
    },
};
// Install mocks on global object
function installMocks() {
    const g = global;
    // self is used by webpack bundles
    g.self = g;
    g.document = mockDocument;
    g.window = mockWindow;
    g.HTMLCanvasElement = MockHTMLCanvasElement;
    g.OffscreenCanvas = MockOffscreenCanvas;
    g.OffscreenCanvasRenderingContext2D = MockCanvasRenderingContext2D;
    g.Image = MockHTMLImageElement;
    g.HTMLImageElement = MockHTMLImageElement;
    g.Audio = MockAudio;
    g.requestAnimationFrame = mockWindow.requestAnimationFrame;
    g.cancelAnimationFrame = mockWindow.cancelAnimationFrame;
    g.localStorage = mockWindow.localStorage;
    // Mock navigator - use Object.defineProperty since navigator may be read-only
    try {
        Object.defineProperty(global, 'navigator', {
            value: {
                userAgent: 'node',
                platform: 'node',
            },
            writable: true,
            configurable: true,
        });
    }
    catch (e) {
        // Navigator already exists and can't be overridden, that's fine
    }
}
// Auto-install on import
installMocks();
// Mock Viewport to satisfy osrs-sdk internals
// Must be set up after mocks are installed, using dynamic require
class HeadlessViewport {
    tick() { }
    draw(_world) { }
    reset() { }
    setPlayer(_player) { }
    calculateViewport() { }
    getViewport(_tickPercent) {
        return { viewportX: 0, viewportY: 0 };
    }
    initialise() {
        return Promise.resolve();
    }
}
// Set the mock viewport singleton (using require to avoid loading osrs-sdk before mocks)
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { Viewport } = require('osrs-sdk');
Viewport.viewport = new HeadlessViewport();
