import {
  Weapon,
  Unit,
  AttackBonuses,
  ProjectileOptions,
  Random,
  Mob,
  Location,
  Region,
  UnitOptions,
  Projectile,
  MeleeWeapon,
  UnitBonuses,
  UnitTypes,
  EntityNames,
  ImageLoader,
} from 'osrs-sdk';

// Image import - only works in browser with webpack, returns empty string in headless
let HurKotImage: string = '';
try {
  // This will work with webpack in browser
  HurKotImage = require('./assets/images/Yt-HurKot.png').default || require('./assets/images/Yt-HurKot.png');
} catch {
  // Headless mode - no image needed
  HurKotImage = '';
}

/**
 * Healing weapon used by YtHurKot to heal Jad.
 * Deals negative damage (healing) of 0-19 HP per hit.
 */
class HealWeapon extends Weapon {
  calculateHitDelay(distance: number) {
    return 3;
  }

  static isMeleeAttackStyle(style: string) {
    return true;
  }

  attack(from: Unit, to: Unit, bonuses: AttackBonuses = {}, options: ProjectileOptions): boolean {
    // Negative damage = healing
    this.damage = -Math.floor(Random.get() * 20);
    this.registerProjectile(from, to, bonuses, options);
    return true;
  }
}

/**
 * Jad's healer mob. Spawns when Jad reaches 50% HP.
 * Heals Jad until attacked by the player, then retaliates.
 */
export class YtHurKot extends Mob {
  myJad: Unit;

  constructor(region: Region, location: Location, options: UnitOptions) {
    super(region, location, options);
    this.myJad = this.aggro as Unit;
  }

  mobName() {
    return EntityNames.YT_HUR_KOT;
  }

  attackStep() {
    super.attackStep();

    // Healers die when Jad dies
    if (this.myJad.isDying()) {
      this.dead();
    }
  }

  shouldChangeAggro(projectile: Projectile) {
    return this.aggro != projectile.from && this.autoRetaliate;
  }

  get combatLevel() {
    return 141;
  }

  setStats() {
    this.stunned = 1; // Start stunned for 1 tick

    this.weapons = {
      heal: new HealWeapon(),
      crush: new MeleeWeapon(),
    };

    this.stats = {
      attack: 165,
      strength: 125,
      defence: 100,
      range: 150,
      magic: 150,
      hitpoint: 90,
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
        magic: 130,
        range: 130,
      },
      other: {
        meleeStrength: 0,
        rangedStrength: 0,
        magicDamage: 0,
        prayer: 0,
      },
    };
  }

  get attackSpeed() {
    return 4;
  }

  // Heal Jad if targeting Jad, attack player if targeting player
  attackStyleForNewAttack() {
    return this.aggro?.type === UnitTypes.PLAYER ? 'crush' : 'heal';
  }

  get attackRange() {
    return 1;
  }

  get size() {
    return 1;
  }

  get image() {
    return HurKotImage;
  }

  get color() {
    return '#ACFF56'; // Light green color for healers
  }

  attackAnimation(tickPercent: number, context: OffscreenCanvasRenderingContext2D) {
    context.transform(1, 0, Math.sin(-tickPercent * Math.PI * 2) / 2, 1, 0, 0);
  }
}
