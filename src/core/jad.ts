import {
    Assets,
    AttackBonuses,
    AttackIndicators,
    Collision,
    DelayedAction,
    EntityNames,
    GLTFModel,
    Location,
    MagicWeapon,
    MeleeWeapon,
    Mob,
    Projectile,
    ArcProjectileMotionInterpolator,
    FollowTargetInterpolator,
    Random,
    RangedWeapon,
    Region,
    Sound,
    Unit,
    UnitBonuses,
    UnitOptions,
} from 'osrs-sdk';

import { YtHurKot } from './healer';
import type { JadRegion } from './jad-region';
const JadModel = Assets.getAssetUrl("models/7700_33012.glb");
const JadRangeProjectileModel = Assets.getAssetUrl("models/jad_range.glb");
const JadMageProjectileModel1 = Assets.getAssetUrl("models/jad_mage_front.glb");
const JadMageProjectileModel2 = Assets.getAssetUrl("models/jad_mage_middle.glb");
const JadMageProjectileModel3 = Assets.getAssetUrl("models/jad_mage_rear.glb");
const HitSound = Assets.getAssetUrl("assets/sounds/dragon_hit_410.ogg");

export const JAD_PROJECTILE_DELAY = 3;


class JadMagicWeapon extends MagicWeapon {
    override attack(from: Mob, to: Unit, bonuses: AttackBonuses = {}): boolean {
        DelayedAction.registerDelayedAction(
            new DelayedAction(() => {
                const overhead = to.prayerController?.matchFeature("magic");
                from.attackFeedback = AttackIndicators.HIT;
                if (overhead) {
                    from.attackFeedback = AttackIndicators.BLOCKED;
                }
                super.attack(from, to, bonuses);
            }, JAD_PROJECTILE_DELAY),
        );
        return true;
    }

    registerProjectile(from: Unit, to: Unit) {
        to.addProjectile(
            new Projectile(this, this.damage, from, to, "magic", {
                reduceDelay: JAD_PROJECTILE_DELAY,
                motionInterpolator: new ArcProjectileMotionInterpolator(1),
                color: "#FFAA00",
                size: 2,
                visualHitEarlyTicks: -1,
                models: [
                    JadMageProjectileModel1,
                    JadMageProjectileModel2,
                    JadMageProjectileModel3,
                ],
                modelScale: 1 / 128,
            }),
        );
    }
}

class JadRangeWeapon extends RangedWeapon {
    attack(from: Mob, to: Unit, bonuses: AttackBonuses = {}): boolean {
        DelayedAction.registerDelayedAction(
            new DelayedAction(() => {
                const overhead = to.prayerController?.matchFeature("range");
                from.attackFeedback = AttackIndicators.HIT;
                if (overhead) {
                    from.attackFeedback = AttackIndicators.BLOCKED;
                }
                super.attack(from, to, bonuses);
            }, JAD_PROJECTILE_DELAY),
        );
        return true;
    }

    registerProjectile(from: Unit, to: Unit) {
        to.addProjectile(
            new Projectile(this, this.damage, from, to, "range", {
                reduceDelay: JAD_PROJECTILE_DELAY,
                model: JadRangeProjectileModel,
                modelScale: 1 / 128,
                visualHitEarlyTicks: -1,
                motionInterpolator: new FollowTargetInterpolator(),
            }),
        );
    }
}

export class Jad extends Mob {
    private hasProccedHealers = false;
    private healerCount: number;
    private jadIndex: number;
    private _attackSpeed: number;

    constructor(
        region: Region,
        location: Location,
        jadIndex: number = 0,
        healerCount: number = 3,
        attackSpeed: number = 8,
        options?: UnitOptions
    ) {
        super(region, location, options);
        this.jadIndex = jadIndex;
        this.healerCount = healerCount;
        this._attackSpeed = attackSpeed;
        this.autoRetaliate = true;
    }

    getJadIndex(): number {
        return this.jadIndex;
    }

    mobName() {
        return EntityNames.JAL_TOK_JAD;
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
        return this._attackSpeed;
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
            stab: new MeleeWeapon(),
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

    get bonuses(): UnitBonuses {
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
        return Random.get() < 0.5 ? "range" : "magic";
    }

    canMeleeIfClose() {
        return "stab" as const;
    }

    magicMaxHit() {
        return 113;
    }

    hitSound() {
        return new Sound(HitSound, 0.1);
    }

    attack() {
        super.attack();
        this.attackFeedback = AttackIndicators.NONE;
        return true;
    }

    /**
     * Called when Jad takes damage
     * Spawns healers when HP drops below 50%
     */
    damageTaken() {
        // Already procced healers
        if (this.hasProccedHealers) {
            return;
        }
        // Above 50% HP
        if (this.currentStats.hitpoint >= this.stats.hitpoint / 2) {
            return;
        }

        this.hasProccedHealers = true;

        const healers: YtHurKot[] = [];

        for (let i = 0; i < this.healerCount; i++) {
            // Find a valid spawn position around Jad
            let xOff: number;
            let yOff: number;

            do {
                xOff = Math.floor(Random.get() * 11) - 5;
                yOff = Math.floor(Random.get() * 15) - 5 - this.size;
            } while (
                Collision.collidesWithAnyMobs(
                    this.region,
                    this.location.x + xOff,
                    this.location.y + yOff,
                    1,
                    this
                )
            );

            const healer = new YtHurKot(
                this.region,
                { x: this.location.x + xOff, y: this.location.y + yOff },
                { aggro: this },
            );
            this.region.addMob(healer);
            healers.push(healer);
        }

        (this.region as JadRegion).registerHealers(this.jadIndex, healers);
    }

    create3dModel() {
        return GLTFModel.forRenderable(this, JadModel);
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

    override get deathAnimationId() {
        return 6;
    }
}
