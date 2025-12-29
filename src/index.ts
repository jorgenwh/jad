import { JadRegion } from './jad-region';
import { AgentController } from './agent-controller';
import { Settings, World, Trainer, Viewport, ImageLoader, MapController, Assets } from 'osrs-sdk';

Settings.readFromStorage();

const region = new JadRegion();
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
const urlParams = new URLSearchParams(window.location.search);
const agentMode = urlParams.get('agent') === 'true';

if (agentMode) {
  console.log('Agent mode enabled - will connect to WebSocket server');
  agentController = new AgentController(player, region.jad);
}

let imagesReady = false;
let assetsReady = false;
let started = false;

function checkStart() {
  if (!started && imagesReady && assetsReady) {
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
        world.startTicking();
      });
    } else {
      world.startTicking();
    }
  }
}

function startGameWithAgent() {
  // Hook into the world's tick event to run agent each tick
  const originalTick = world.tickWorld.bind(world);
  world.tickWorld = (ticks: number) => {
    originalTick(ticks);  // Tick happens FIRST
    if (agentController) {
      agentController.tick();  // Then observe (state AFTER tick, matching headless)
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
