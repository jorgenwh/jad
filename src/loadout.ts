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
  TorvaFullhelm,
  TorvaPlatebody,
  TorvaPlatelegs,
  InfernalCape,
  PrimordialBoots,
  FerociousGloves,
  BladeOfSaeldor,
  AvernicDefender,
  AmuletOfTorture,
  SuperCombatPotion,
} from "osrs-sdk";

export const getRangedLoadout = () => {
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

export const getMeleeLoadout = () => {
  const equipment = new UnitEquipment();
  equipment.weapon = new BladeOfSaeldor();
  equipment.offhand = new AvernicDefender();
  // equipment.helmet = new TorvaFullhelm();
  equipment.necklace = new AmuletOfTorture();
  equipment.cape = new InfernalCape();
  // equipment.chest = new TorvaPlatebody();
  // equipment.legs = new TorvaPlatelegs();
  equipment.feet = new PrimordialBoots();
  equipment.gloves = new FerociousGloves();
  equipment.ring = new RingOfSufferingImbued();

  return {
    equipment,
    inventory: [
      new SuperCombatPotion(),
      new SuperRestore(),
      new SaradominBrew(),
    ],
  };
};
