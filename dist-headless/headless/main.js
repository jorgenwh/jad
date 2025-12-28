"use strict";
/**
 * Headless entry point for the Jad RL environment.
 * Runs the simulation without browser rendering.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
// Must import mocks FIRST before any osrs-sdk imports
require("./mocks");
const readline = __importStar(require("readline"));
const jad_region_1 = require("../jad-region");
const env_1 = require("./env");
// Create environment
const env = new env_1.HeadlessEnv(() => new jad_region_1.JadRegion());
// Set up stdio protocol
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false,
});
// Output function that writes directly to stdout (bypasses console.log redirect)
function output(data) {
    process.stdout.write(JSON.stringify(data) + '\n');
}
rl.on('line', (line) => {
    try {
        const msg = JSON.parse(line);
        let result;
        switch (msg.command) {
            case 'reset':
                result = env.reset();
                break;
            case 'step':
                result = env.step(msg.action);
                break;
            case 'close':
                rl.close();
                process.exit(0);
                break;
            default:
                result = { error: `Unknown command: ${msg.command}` };
        }
        output(result);
    }
    catch (err) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        output({ error: errorMessage });
    }
});
// Handle clean shutdown
process.on('SIGINT', () => {
    rl.close();
    process.exit(0);
});
process.on('SIGTERM', () => {
    rl.close();
    process.exit(0);
});
// Signal that we're ready
console.error('Headless Jad environment ready');
