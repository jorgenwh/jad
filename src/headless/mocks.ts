/**
 * Browser API mocks for running osrs-sdk in Node.js headless mode.
 * This file must be imported BEFORE any osrs-sdk imports.
 */

// Mock OffscreenCanvas
class MockOffscreenCanvas {
  width: number;
  height: number;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
  }

  getContext(_type: string): MockCanvasRenderingContext2D {
    return new MockCanvasRenderingContext2D();
  }

  transferToImageBitmap(): unknown {
    return {};
  }
}

// Mock CanvasRenderingContext2D
class MockCanvasRenderingContext2D {
  fillStyle: string = '';
  strokeStyle: string = '';
  lineWidth: number = 1;
  font: string = '';
  textAlign: string = 'left';
  textBaseline: string = 'top';
  globalAlpha: number = 1;
  globalCompositeOperation: string = 'source-over';

  fillRect(_x: number, _y: number, _w: number, _h: number): void {}
  strokeRect(_x: number, _y: number, _w: number, _h: number): void {}
  clearRect(_x: number, _y: number, _w: number, _h: number): void {}
  fillText(_text: string, _x: number, _y: number): void {}
  strokeText(_text: string, _x: number, _y: number): void {}
  measureText(_text: string): { width: number } {
    return { width: 0 };
  }
  beginPath(): void {}
  closePath(): void {}
  moveTo(_x: number, _y: number): void {}
  lineTo(_x: number, _y: number): void {}
  arc(_x: number, _y: number, _r: number, _start: number, _end: number): void {}
  fill(): void {}
  stroke(): void {}
  save(): void {}
  restore(): void {}
  translate(_x: number, _y: number): void {}
  rotate(_angle: number): void {}
  scale(_x: number, _y: number): void {}
  drawImage(..._args: unknown[]): void {}
  createLinearGradient(_x0: number, _y0: number, _x1: number, _y1: number): MockGradient {
    return new MockGradient();
  }
  createRadialGradient(_x0: number, _y0: number, _r0: number, _x1: number, _y1: number, _r1: number): MockGradient {
    return new MockGradient();
  }
  setTransform(..._args: unknown[]): void {}
  resetTransform(): void {}
  clip(): void {}
  rect(_x: number, _y: number, _w: number, _h: number): void {}
  ellipse(..._args: unknown[]): void {}
  quadraticCurveTo(_cpx: number, _cpy: number, _x: number, _y: number): void {}
  bezierCurveTo(_cp1x: number, _cp1y: number, _cp2x: number, _cp2y: number, _x: number, _y: number): void {}
  arcTo(_x1: number, _y1: number, _x2: number, _y2: number, _radius: number): void {}
  isPointInPath(_x: number, _y: number): boolean {
    return false;
  }
  getImageData(_sx: number, _sy: number, _sw: number, _sh: number): { data: Uint8ClampedArray; width: number; height: number } {
    return { data: new Uint8ClampedArray(0), width: 0, height: 0 };
  }
  putImageData(_imageData: unknown, _dx: number, _dy: number): void {}
}

class MockGradient {
  addColorStop(_offset: number, _color: string): void {}
}

// Mock HTMLCanvasElement
class MockHTMLCanvasElement {
  width: number = 800;
  height: number = 600;

  getContext(_type: string): MockCanvasRenderingContext2D {
    return new MockCanvasRenderingContext2D();
  }

  toDataURL(): string {
    return '';
  }

  toBlob(_callback: (blob: unknown) => void): void {
    _callback(null);
  }
}

// Mock HTMLImageElement
class MockHTMLImageElement {
  src: string = '';
  width: number = 32;
  height: number = 32;
  complete: boolean = true;
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  private listeners: Map<string, Array<() => void>> = new Map();

  constructor() {
    // Simulate async load completion
    setTimeout(() => {
      this.complete = true;
      if (this.onload) this.onload();
      this.dispatchEvent('load');
    }, 0);
  }

  addEventListener(event: string, handler: () => void): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(handler);
  }

  removeEventListener(event: string, handler: () => void): void {
    const handlers = this.listeners.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index >= 0) handlers.splice(index, 1);
    }
  }

  dispatchEvent(event: string): void {
    const handlers = this.listeners.get(event);
    if (handlers) {
      handlers.forEach(h => h());
    }
  }
}

// Mock Audio
class MockAudio {
  src: string = '';
  volume: number = 1;
  currentTime: number = 0;
  paused: boolean = true;

  play(): Promise<void> {
    return Promise.resolve();
  }
  pause(): void {}
  load(): void {}

  cloneNode(): MockAudio {
    return new MockAudio();
  }
}

// Mock document
const mockDocument = {
  createElement(tagName: string): unknown {
    if (tagName === 'canvas') return new MockHTMLCanvasElement();
    if (tagName === 'img') return new MockHTMLImageElement();
    if (tagName === 'audio') return new MockAudio();
    return {};
  },
  getElementById(_id: string): unknown {
    return null;
  },
  querySelector(_selector: string): unknown {
    return null;
  },
  querySelectorAll(_selector: string): unknown[] {
    return [];
  },
  body: {
    appendChild(_child: unknown): void {},
    removeChild(_child: unknown): void {},
  },
  addEventListener(_event: string, _handler: unknown): void {},
  removeEventListener(_event: string, _handler: unknown): void {},
};

// Mock localStorage with proper storage
const mockLocalStorageData: Record<string, string> = {};
const mockLocalStorage = {
  getItem(key: string): string | null {
    return mockLocalStorageData[key] ?? null;
  },
  setItem(key: string, value: string): void {
    mockLocalStorageData[key] = value;
  },
  removeItem(key: string): void {
    delete mockLocalStorageData[key];
  },
  clear(): void {
    Object.keys(mockLocalStorageData).forEach(k => delete mockLocalStorageData[k]);
  },
};

// Mock window
const mockWindow = {
  requestAnimationFrame(_callback: FrameRequestCallback): number {
    // Don't actually schedule - headless mode uses manual ticking
    return 0;
  },
  cancelAnimationFrame(_id: number): void {},
  addEventListener(_event: string, _handler: unknown): void {},
  removeEventListener(_event: string, _handler: unknown): void {},
  innerWidth: 800,
  innerHeight: 600,
  devicePixelRatio: 1,
  localStorage: mockLocalStorage,
  performance: {
    now(): number {
      return Date.now();
    },
  },
};

// Install mocks on global object
export function installMocks(): void {
  const g = global as unknown as Record<string, unknown>;

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
  } catch (e) {
    // Navigator already exists and can't be overridden, that's fine
  }
}

// Auto-install on import
installMocks();

// Mock Viewport to satisfy osrs-sdk internals
// Must be set up after mocks are installed, using dynamic require
class HeadlessViewport {
  tick(): void {}
  draw(_world: unknown): void {}
  reset(): void {}
  setPlayer(_player: unknown): void {}
  calculateViewport(): void {}
  getViewport(_tickPercent: number): { viewportX: number; viewportY: number } {
    return { viewportX: 0, viewportY: 0 };
  }
  initialise(): Promise<void> {
    return Promise.resolve();
  }
}

// Set the mock viewport singleton (using require to avoid loading osrs-sdk before mocks)
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { Viewport, Assets } = require('osrs-sdk');
(Viewport as unknown as { viewport: HeadlessViewport }).viewport = new HeadlessViewport();

// Mock Assets.getAssetUrl to prevent network fetches in headless mode
// The original implementation fetches from oldschool-cdn.com which causes timeouts
const originalGetAssetUrl = Assets.getAssetUrl.bind(Assets);
Assets.getAssetUrl = function(asset: string): string {
  const url = `https://oldschool-cdn.com/${asset}`;
  // Mark as already loaded to prevent any fetch attempts
  Assets.loadedAssets[url] = true;
  // Don't add to loadingAssetUrls, don't increment assetCount, don't fetch
  return url;
};

// Ensure checkAssetsLoaded always reports complete (no pending loads)
Assets.loadingAssetUrls = [];
Assets.assetCount = 0;
