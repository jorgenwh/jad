/**
 * Bootstrap file that sets up mocks before anything else loads.
 * This must be the entry point - it sets up globals then requires main.
 * Dreamt up by Claude
 *
 * Uses require() instead of import to control execution order.
 */

// Redirect ALL console output to stderr to keep stdout clean for JSON protocol
// Must happen before ANY other code runs
const _stderr = process.stderr;
const writeToStderr = (...args: unknown[]) => {
    _stderr.write(args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' ') + '\n');
};
console.log = writeToStderr;
console.info = writeToStderr;
console.warn = writeToStderr;
console.debug = writeToStderr;
// Keep console.error as-is since it already goes to stderr

// Set up self/window/document IMMEDIATELY before any other requires
const g = global as unknown as Record<string, unknown>;
g.self = g;
g.window = g;
g.document = {
    createElement: () => ({}),
    getElementById: () => null,
    querySelector: () => null,
    querySelectorAll: () => [],
    body: { appendChild: () => {}, removeChild: () => {} },
    addEventListener: () => {},
    removeEventListener: () => {},
};
const storageData: Record<string, string> = {};
g.localStorage = {
    getItem: (key: string) => storageData[key] ?? null,
    setItem: (key: string, val: string) => { storageData[key] = val; },
    removeItem: (key: string) => { delete storageData[key]; },
    clear: () => { Object.keys(storageData).forEach(k => delete storageData[k]); },
};
g.requestAnimationFrame = () => 0;
g.cancelAnimationFrame = () => {};

// Now require the full mocks (which adds more complete implementations)
require('./mocks');

// Then require main to start the application
require('./main');
