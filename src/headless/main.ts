/**
 * Headless entry point for the Jad RL environment.
 * Runs the simulation without browser rendering.
 */

// Must import mocks FIRST before any osrs-sdk imports
import './mocks';

import * as readline from 'readline';
import { JadRegion } from '../jad-region';
import { HeadlessEnv } from './env';

// Create environment
const env = new HeadlessEnv(() => new JadRegion());

// Set up stdio protocol
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false,
});

// Output function that writes directly to stdout (bypasses console.log redirect)
function output(data: unknown): void {
  process.stdout.write(JSON.stringify(data) + '\n');
}

rl.on('line', (line: string) => {
  try {
    const msg = JSON.parse(line);
    let result: unknown;

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
  } catch (err) {
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
