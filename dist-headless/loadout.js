"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getLoadout = void 0;
const osrs_sdk_1 = require("osrs-sdk");
const getLoadout = () => {
    const equipment = new osrs_sdk_1.UnitEquipment();
    equipment.weapon = new osrs_sdk_1.TwistedBow();
    equipment.helmet = new osrs_sdk_1.MasoriMaskF();
    equipment.necklace = new osrs_sdk_1.NecklaceOfAnguish();
    equipment.cape = new osrs_sdk_1.DizanasQuiver();
    equipment.ammo = new osrs_sdk_1.DragonArrows();
    equipment.chest = new osrs_sdk_1.MasoriBodyF();
    equipment.legs = new osrs_sdk_1.MasoriChapsF();
    equipment.feet = new osrs_sdk_1.PegasianBoots();
    equipment.gloves = new osrs_sdk_1.ZaryteVambraces();
    equipment.ring = new osrs_sdk_1.RingOfSufferingImbued();
    return {
        equipment,
        inventory: [
            new osrs_sdk_1.BastionPotion(),
            new osrs_sdk_1.SuperRestore(),
            new osrs_sdk_1.SaradominBrew(),
        ],
    };
};
exports.getLoadout = getLoadout;
