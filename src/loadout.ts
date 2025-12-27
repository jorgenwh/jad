import {
  BastionPotion,
  DizanasQuiver,
  DragonArrows,
  MasoriBodyF,
  MasoriChapsF,
  MasoriMaskF,
  NecklaceOfAnguish,
  PegasianBoots,
  RingOfSufferingImbued,
  SaradominBrew,
  SuperRestore,
  TwistedBow,
  UnitEquipment,
  ZaryteVambraces,
} from "osrs-sdk";

export const getLoadout = () => {
  const equipment = new UnitEquipment();
  equipment.weapon = new TwistedBow();
  equipment.helmet = new MasoriMaskF();
  equipment.necklace = new NecklaceOfAnguish();
  equipment.cape = new DizanasQuiver();
  equipment.ammo = new DragonArrows();
  equipment.chest = new MasoriBodyF();
  equipment.legs = new MasoriChapsF();
  equipment.feet = new PegasianBoots();
  equipment.gloves = new ZaryteVambraces();
  equipment.ring = new RingOfSufferingImbued();

  return {
    equipment,
    inventory: [
      new BastionPotion(),
      new SuperRestore(),
      new SaradominBrew(),
    ],
  };
};
