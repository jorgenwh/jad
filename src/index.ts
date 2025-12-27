import { JadRegion } from './jad-region';
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

let imagesReady = false;
let assetsReady = false;
let started = false;

function checkStart() {
  if (!started && imagesReady && assetsReady) {
    started = true;
    console.log('All assets loaded, starting game...');
    world.startTicking();
  }
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
