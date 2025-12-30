/**
 * Headless entry point for the Jad RL environment.
 * Runs the simulation without browser rendering.
 *
 * Configuration via environment variables:
 * - JAD_COUNT: Number of Jads (1-6, default: 1)
 * - HEALERS_PER_JAD: Number of healers per Jad (0-5, default: 3)
 */

// Must import mocks FIRST before any osrs-sdk imports
import './mocks';

import * as readline from 'readline';
import { JadRegion } from '../jad-region';
import { HeadlessEnv, getActionCount } from './env';
import { parseConfig, JadConfig } from '../config';

// Parse config from environment
const config = parseConfig();
console.error(`Jad config: ${config.jadCount} Jad(s), ${config.healersPerJad} healers per Jad`);
console.error(`Action count: ${getActionCount(config)}`);

// Create environment with config
const env = new HeadlessEnv((cfg: JadConfig) => new JadRegion(cfg), config);

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
