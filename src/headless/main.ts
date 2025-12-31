// Must import mocks FIRST before any osrs-sdk imports
import './mocks';

import * as readline from 'readline';
import { Settings } from 'osrs-sdk';
import { JadRegion, JadConfig } from '../core';
import { HeadlessEnv, EnvConfig, StepResult } from './env';

// Initialize settings from (mock) storage
Settings.readFromStorage();

function parseJadConfig(): JadConfig {
    const jadCount = parseInt(process.env.JAD_COUNT || '1', 10);
    const healersPerJad = parseInt(process.env.HEALERS_PER_JAD || '3', 10);

    if (jadCount < 1 || jadCount > 6) {
        throw new Error(`JAD_COUNT must be 1-6, got ${jadCount}`);
    }
    if (healersPerJad < 0 || healersPerJad > 5) {
        throw new Error(`HEALERS_PER_JAD must be 0-5, got ${healersPerJad}`);
    }

    return { jadCount, healersPerJad };
}

function parseEnvConfig(): EnvConfig {
    const rewardFunc = process.env.REWARD_FUNC || 'default';
    return { rewardFunc };
}

const jadConfig = parseJadConfig();
const envConfig = parseEnvConfig();
const env = new HeadlessEnv((cfg: JadConfig) => new JadRegion(cfg), jadConfig, envConfig);

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
        let result: StepResult;

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
                output({ error: `Unknown command: ${msg.command}` });
                return;
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
