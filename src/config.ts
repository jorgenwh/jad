/**
 * Configuration for multi-Jad environment.
 */

export interface JadConfig {
  /** Number of Jads (1-6) */
  jadCount: number;
  /** Number of healers per Jad (default: 3) */
  healersPerJad: number;
}

export const DEFAULT_CONFIG: JadConfig = {
  jadCount: 1,
  healersPerJad: 3,
};

/**
 * Get action count for given Jad configuration.
 * Actions: DO_NOTHING + N*AGGRO_JAD + N*3*AGGRO_HEALER + 7 prayers/potions
 */
export function getActionCount(config: JadConfig): number {
  return 1 + config.jadCount + config.jadCount * config.healersPerJad + 7;
}

/**
 * Parse configuration from CLI args or environment.
 */
export function parseConfig(): JadConfig {
  const jadCount = parseInt(process.env.JAD_COUNT || '1', 10);
  const healersPerJad = parseInt(process.env.HEALERS_PER_JAD || '3', 10);

  // Validate
  if (jadCount < 1 || jadCount > 6) {
    throw new Error(`JAD_COUNT must be 1-6, got ${jadCount}`);
  }
  if (healersPerJad < 0 || healersPerJad > 5) {
    throw new Error(`HEALERS_PER_JAD must be 0-5, got ${healersPerJad}`);
  }

  return { jadCount, healersPerJad };
}
