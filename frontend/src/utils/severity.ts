/**
 * Standard Air Quality Warning Severity Color System for Dark Theme
 * 
 * Consistent with:
 * - Primary accent: teal/turquoise (#14D6C5)
 * - Good: green (#10B981)
 * - Moderate: amber (#F59E0B)
 * - Light/Unhealthy for Sensitive: orange (#F97316)
 * - Moderate/Unhealthy: red (#EF4444)
 * - Heavy/Very Unhealthy: purple (#8B5CF6)
 * - Severe/Hazardous: deep red (#DC2626)
 */

export interface SeverityTierStyle {
  tierKey: 'good' | 'moderate' | 'light' | 'high' | 'severe' | 'hazardous';
  grade: string;
  name: string;
  label: string;
  bg: string;
  text: string;
  border: string;
  dot: string;
  chartColor: string;
}

export const SEVERITY_CONFIG: Record<string, SeverityTierStyle> = {
  good: {
    tierKey: 'good',
    grade: 'Grade I',
    name: 'Good',
    label: 'Good (Grade I)',
    bg: 'rgba(16, 185, 129, 0.12)',
    text: '#34D399',
    border: 'rgba(16, 185, 129, 0.3)',
    dot: '#10B981',
    chartColor: '#10B981',
  },
  moderate: {
    tierKey: 'moderate',
    grade: 'Grade II',
    name: 'Moderate',
    label: 'Moderate (Grade II)',
    bg: 'rgba(245, 158, 11, 0.12)',
    text: '#FBBF24',
    border: 'rgba(245, 158, 11, 0.3)',
    dot: '#F59E0B',
    chartColor: '#F59E0B',
  },
  light: {
    tierKey: 'light',
    grade: 'Grade III',
    name: 'Light',
    label: 'Light Pollution (Grade III)',
    bg: 'rgba(249, 115, 22, 0.12)',
    text: '#FB923C',
    border: 'rgba(249, 115, 22, 0.3)',
    dot: '#F97316',
    chartColor: '#F97316',
  },
  high: {
    tierKey: 'high',
    grade: 'Grade IV',
    name: 'Moderate',
    label: 'Moderate Pollution (Grade IV)',
    bg: 'rgba(239, 68, 68, 0.15)',
    text: '#F87171',
    border: 'rgba(239, 68, 68, 0.35)',
    dot: '#EF4444',
    chartColor: '#EF4444',
  },
  severe: {
    tierKey: 'severe',
    grade: 'Grade V',
    name: 'Heavy',
    label: 'Heavy Pollution (Grade V)',
    bg: 'rgba(139, 92, 246, 0.15)',
    text: '#A78BFA',
    border: 'rgba(139, 92, 246, 0.35)',
    dot: '#8B5CF6',
    chartColor: '#8B5CF6',
  },
  hazardous: {
    tierKey: 'hazardous',
    grade: 'Grade VI',
    name: 'Severe',
    label: 'Severe / Emergency (Grade VI)',
    bg: 'rgba(220, 38, 38, 0.2)',
    text: '#FCA5A5',
    border: 'rgba(220, 38, 38, 0.5)',
    dot: '#DC2626',
    chartColor: '#DC2626',
  },
};

export const getSeverityStyle = (tierOrLabel?: string, explicitColorHex?: string): SeverityTierStyle => {
  if (!tierOrLabel) return SEVERITY_CONFIG.good;

  const l = tierOrLabel.toLowerCase();
  let base: SeverityTierStyle;

  if (l.includes('grade vi') || l.includes('severe') || l.includes('hazardous')) {
    base = SEVERITY_CONFIG.hazardous;
  } else if (l.includes('grade v') || l.includes('heavy') || l.includes('very unhealthy')) {
    base = SEVERITY_CONFIG.severe;
  } else if (l.includes('grade iv') || l.includes('unhealthy') || l.includes('tier 4')) {
    base = SEVERITY_CONFIG.high;
  } else if (l.includes('grade iii') || l.includes('light') || l.includes('sensitive') || l.includes('tier 3')) {
    base = SEVERITY_CONFIG.light;
  } else if (l.includes('grade ii') || l.includes('moderate') || l.includes('tier 2')) {
    base = SEVERITY_CONFIG.moderate;
  } else {
    base = SEVERITY_CONFIG.good;
  }

  if (explicitColorHex) {
    return {
      ...base,
      dot: explicitColorHex,
      border: `${explicitColorHex}4D`,
      bg: `${explicitColorHex}1A`,
      text: explicitColorHex,
    };
  }

  return base;
};
