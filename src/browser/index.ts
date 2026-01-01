import { JadRegion, JadConfig } from '../core';
import { AgentController } from './agent-controller';
import { Settings, World, Trainer, Viewport, ImageLoader, MapController, Assets } from 'osrs-sdk';

Settings.readFromStorage();

// Parse config from URL params (e.g., ?jads=3&healers=3)
const urlParams = new URLSearchParams(window.location.search);
const jadCount = parseInt(urlParams.get('jads') || '1', 10);
const healersPerJad = parseInt(urlParams.get('healers') || '3', 10);
const config: JadConfig = {
    jadCount: Math.max(1, Math.min(6, jadCount)),
    healersPerJad: Math.max(0, Math.min(5, healersPerJad)),
};
console.log(`Config: ${config.jadCount} Jad(s), ${config.healersPerJad} healers per Jad`);

const region = new JadRegion(config);
const world = new World();
region.world = world;
world.addRegion(region);

const { player } = region.initialiseRegion();

Viewport.setupViewport(region);
Viewport.viewport.setPlayer(player);

Trainer.setPlayer(player);

// Agent controller for AI play
let agentController: AgentController | null = null;

// Check URL params for agent mode
const agentMode = urlParams.get('agent') === 'true';

if (agentMode) {
    console.log('Agent mode enabled - will connect to WebSocket server');
    agentController = new AgentController(player, region, config);
}

let imagesReady = false;
let assetsReady = false;
let started = false;

function checkStart() {
    // Ensure both images and assets are ready
    if (!imagesReady || !assetsReady) {
        return;
    }
    // Prevent multiple starts
    if (started) {
        return;
    }

    started = true;
    console.log('All assets loaded, starting game...');

    if (agentController) {
        // Connect to agent server before starting
        agentController.connect().then(() => {
            console.log('Agent connected, starting game loop');
            startGameWithAgent();
        }).catch((err) => {
            console.error('Failed to connect to agent server:', err);
            console.log('Starting without agent...');
            // Start normally as a fallback if agent connection fails
            world.startTicking();
        });
    } else {
        // No agent, start normally
        world.startTicking();
    }
}

function startGameWithAgent() {
    // Hook into the world's tick event to run agent each tick
    const tickWorld = world.tickWorld.bind(world);

    world.tickWorld = (ticks: number) => {
        tickWorld(ticks);  // Tick happens FIRST
        if (agentController) {
            agentController.tick();  // Then observe
        }
    };

    // Use the normal game loop (handles rendering + ticking properly)
    world.startTicking();
}

ImageLoader.onAllImagesLoaded(() => {
    console.log('Images loaded');
    MapController.controller.updateOrbsMask(player.currentStats, player.stats);
    imagesReady = true;
    checkStart();
});

const imageCheckInterval = setInterval(() => {
    ImageLoader.checkImagesLoaded(imageCheckInterval);
}, 50);

Assets.onAllAssetsLoaded(() => {
    Viewport.viewport.initialise().then(() => {
        console.log('Assets preloaded');
        assetsReady = true;
        checkStart();
    });
});

const assetsCheckInterval = setInterval(() => {
    Assets.checkAssetsLoaded(assetsCheckInterval);
}, 50);
