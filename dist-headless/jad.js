"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Jad = void 0;
const osrs_sdk_1 = require("osrs-sdk");
const JadModel = osrs_sdk_1.Assets.getAssetUrl("models/7700_33012.glb");
const JadRangeProjectileModel = osrs_sdk_1.Assets.getAssetUrl("models/jad_range.glb");
const JadMageProjectileModel1 = osrs_sdk_1.Assets.getAssetUrl("models/jad_mage_front.glb");
const JadMageProjectileModel2 = osrs_sdk_1.Assets.getAssetUrl("models/jad_mage_middle.glb");
const JadMageProjectileModel3 = osrs_sdk_1.Assets.getAssetUrl("models/jad_mage_rear.glb");
const HitSound = osrs_sdk_1.Assets.getAssetUrl("assets/sounds/dragon_hit_410.ogg");
const JAD_PROJECTILE_DELAY = 3;
class JadMagicWeapon extends osrs_sdk_1.MagicWeapon {
    attack(from, to, bonuses = {}) {
        osrs_sdk_1.DelayedAction.registerDelayedAction(new osrs_sdk_1.DelayedAction(() => {
            const overhead = to.prayerController?.matchFeature("magic");
            from.attackFeedback = osrs_sdk_1.AttackIndicators.HIT;
            if (overhead) {
                from.attackFeedback = osrs_sdk_1.AttackIndicators.BLOCKED;
            }
            super.attack(from, to, bonuses);
        }, JAD_PROJECTILE_DELAY));
        return true;
    }
    registerProjectile(from, to) {
        to.addProjectile(new osrs_sdk_1.Projectile(this, this.damage, from, to, "magic", {
            reduceDelay: JAD_PROJECTILE_DELAY,
            motionInterpolator: new osrs_sdk_1.ArcProjectileMotionInterpolator(1),
            color: "#FFAA00",
            size: 2,
            visualHitEarlyTicks: -1,
            models: [
                JadMageProjectileModel1,
                JadMageProjectileModel2,
                JadMageProjectileModel3,
            ],
            modelScale: 1 / 128,
        }));
    }
}
class JadRangeWeapon extends osrs_sdk_1.RangedWeapon {
    attack(from, to, bonuses = {}) {
        osrs_sdk_1.DelayedAction.registerDelayedAction(new osrs_sdk_1.DelayedAction(() => {
            const overhead = to.prayerController?.matchFeature("range");
            from.attackFeedback = osrs_sdk_1.AttackIndicators.HIT;
            if (overhead) {
                from.attackFeedback = osrs_sdk_1.AttackIndicators.BLOCKED;
            }
            super.attack(from, to, bonuses);
        }, JAD_PROJECTILE_DELAY));
        return true;
    }
    registerProjectile(from, to) {
        to.addProjectile(new osrs_sdk_1.Projectile(this, this.damage, from, to, "range", {
            reduceDelay: JAD_PROJECTILE_DELAY,
            model: JadRangeProjectileModel,
            modelScale: 1 / 128,
            visualHitEarlyTicks: -1,
            motionInterpolator: new osrs_sdk_1.FollowTargetInterpolator(),
        }));
    }
}
class Jad extends osrs_sdk_1.Mob {
    constructor(region, location, options) {
        super(region, location, options);
        this.autoRetaliate = true;
    }
    mobName() {
        return osrs_sdk_1.EntityNames.JAL_TOK_JAD;
    }
    get combatLevel() {
        return 900;
    }
    get size() {
        return 5;
    }
    get attackRange() {
        return 50;
    }
    get attackSpeed() {
        return 8;
    }
    get flinchDelay() {
        return 2;
    }
    get clickboxRadius() {
        return 2.5;
    }
    get clickboxHeight() {
        return 4;
    }
    setStats() {
        this.weapons = {
            stab: new osrs_sdk_1.MeleeWeapon(),
            magic: new JadMagicWeapon(),
            range: new JadRangeWeapon(),
        };
        this.stats = {
            hitpoint: 350,
            attack: 750,
            strength: 1020,
            defence: 480,
            range: 1020,
            magic: 510,
        };
        this.currentStats = JSON.parse(JSON.stringify(this.stats));
    }
    get bonuses() {
        return {
            attack: {
                stab: 0,
                slash: 0,
                crush: 0,
                magic: 100,
                range: 80,
            },
            defence: {
                stab: 0,
                slash: 0,
                crush: 0,
                magic: 0,
                range: 0,
            },
            other: {
                meleeStrength: 0,
                rangedStrength: 80,
                magicDamage: 1.75,
                prayer: 0,
            },
        };
    }
    attackStyleForNewAttack() {
        return osrs_sdk_1.Random.get() < 0.5 ? "range" : "magic";
    }
    canMeleeIfClose() {
        return "stab";
    }
    magicMaxHit() {
        return 113;
    }
    hitSound() {
        return new osrs_sdk_1.Sound(HitSound, 0.1);
    }
    attack() {
        super.attack();
        this.attackFeedback = osrs_sdk_1.AttackIndicators.NONE;
        return true;
    }
    create3dModel() {
        return osrs_sdk_1.GLTFModel.forRenderable(this, JadModel);
    }
    get attackAnimationId() {
        switch (this.attackStyle) {
            case "magic":
                return 2;
            case "range":
                return 3;
            default:
                return 4;
        }
    }
    get deathAnimationId() {
        return 6;
    }
}
exports.Jad = Jad;
