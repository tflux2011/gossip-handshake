/**
 * Minimalistic SVG icon set for the Gossip Handshake Protocol POC.
 *
 * All icons accept `size` (default 20) and `className` props.
 * No emoji usage — clean, consistent line icons.
 */

const defaults = { size: 20, strokeWidth: 1.8 }

function wrap(props, children) {
  const s = props.size || defaults.size
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={s}
      height={s}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={props.strokeWidth || defaults.strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={props.className || ''}
      aria-hidden="true"
      style={props.style}
    >
      {children}
    </svg>
  )
}

/* ── Domain icons ─────────────────────────────── */

/** Wheat stalk — Agronomy */
export function WheatIcon(props) {
  return wrap(props, <>
    <path d="M12 21V10" />
    <path d="M6 12s2-3 6-3" />
    <path d="M18 12s-2-3-6-3" />
    <path d="M7 8s1.5-2.5 5-3" />
    <path d="M17 8s-1.5-2.5-5-3" />
    <path d="M8.5 5s1-2 3.5-2" />
    <path d="M15.5 5s-1-2-3.5-2" />
  </>)
}

/** Stethoscope — Veterinary */
export function StethoscopeIcon(props) {
  return wrap(props, <>
    <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3" />
    <path d="M8 15v1a6 6 0 0 0 6 6 6 6 0 0 0 6-6v-4" />
    <circle cx="20" cy="10" r="2" />
  </>)
}

/** Water drop — Irrigation */
export function DropletIcon(props) {
  return wrap(props, <>
    <path d="M12 22a7 7 0 0 0 7-7c0-4-3.5-7.5-7-11-3.5 3.5-7 7-7 11a7 7 0 0 0 7 7z" />
  </>)
}

/** Layers stack — Soil Science */
export function LayersIcon(props) {
  return wrap(props, <>
    <polygon points="12 2 2 7 12 12 22 7 12 2" />
    <polyline points="2 17 12 22 22 17" />
    <polyline points="2 12 12 17 22 12" />
  </>)
}

/** Fish — Aquaculture */
export function FishIcon(props) {
  return wrap(props, <>
    <path d="M6.5 12c.94-3.46 4.94-6 8.5-6 3.56 0 3 3 3 6s.56 6-3 6c-3.56 0-7.56-2.54-8.5-6z" />
    <path d="M6.5 12H2" />
    <path d="M18 19l2-2-2-2" />
    <circle cx="14.5" cy="10.5" r="0.8" fill="currentColor" stroke="none" />
  </>)
}

/* ── UI icons ─────────────────────────────────── */

/** Shuffle arrows — Router */
export function ShuffleIcon(props) {
  return wrap(props, <>
    <polyline points="16 3 21 3 21 8" />
    <line x1="4" y1="20" x2="21" y2="3" />
    <polyline points="21 16 21 21 16 21" />
    <line x1="15" y1="15" x2="21" y2="21" />
    <line x1="4" y1="4" x2="9" y2="9" />
  </>)
}

/** Arrow up — Send message */
export function SendIcon(props) {
  return wrap(props, <>
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </>)
}

/** Clock — Timer / Loading */
export function ClockIcon(props) {
  return wrap(props, <>
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </>)
}

/** Zap — Fast / Generation time */
export function ZapIcon(props) {
  return wrap(props, <>
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
  </>)
}

/** Radio — Signal / Router type */
export function RadioIcon(props) {
  return wrap(props, <>
    <path d="M16.2 3.8a9 9 0 0 1 0 16.4" />
    <path d="M7.8 20.2a9 9 0 0 1 0-16.4" />
    <path d="M14.1 6a5 5 0 0 1 0 12" />
    <path d="M9.9 18a5 5 0 0 1 0-12" />
    <circle cx="12" cy="12" r="1" fill="currentColor" />
  </>)
}

/** Triangle alert — Warning */
export function AlertIcon(props) {
  return wrap(props, <>
    <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </>)
}

/** Handshake — Protocol logo */
export function HandshakeIcon(props) {
  return wrap(props, <>
    <path d="M11 17l-1-1 4-4a1 1 0 0 0-1.42-1.42L8 15" />
    <path d="M16 7l-4 4" />
    <path d="M20 4l-3 3" />
    <path d="M4 20l3-3" />
    <path d="M6.5 12.5l-2 2a2 2 0 1 0 2.8 2.8l2-2" />
    <path d="M14.5 6.5l2-2a2 2 0 1 1 2.8 2.8l-2 2" />
    <path d="M7 17l1 1" />
  </>)
}

/** Checkmark — Selected / Success */
export function CheckIcon(props) {
  return wrap(props, <>
    <polyline points="20 6 9 17 4 12" />
  </>)
}

/** Key — Keyword router */
export function KeyIcon(props) {
  return wrap(props, <>
    <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.78 7.78 5.5 5.5 0 0 1 7.78-7.78zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4" />
  </>)
}

/** Crosshair / Target — Cosine router */
export function TargetIcon(props) {
  return wrap(props, <>
    <circle cx="12" cy="12" r="10" />
    <circle cx="12" cy="12" r="6" />
    <circle cx="12" cy="12" r="2" />
  </>)
}

/** Loader — Spinner */
export function LoaderIcon(props) {
  return wrap({ ...props, className: `${props.className || ''} icon-spin` }, <>
    <line x1="12" y1="2" x2="12" y2="6" />
    <line x1="12" y1="18" x2="12" y2="22" />
    <line x1="4.93" y1="4.93" x2="7.76" y2="7.76" />
    <line x1="16.24" y1="16.24" x2="19.07" y2="19.07" />
    <line x1="2" y1="12" x2="6" y2="12" />
    <line x1="18" y1="12" x2="22" y2="12" />
    <line x1="4.93" y1="19.07" x2="7.76" y2="16.24" />
    <line x1="16.24" y1="7.76" x2="19.07" y2="4.93" />
  </>)
}

/**
 * Domain icon lookup — maps domain key to the right icon component.
 * Use this wherever you previously used emoji domain icons.
 */
export const DOMAIN_ICONS = {
  agronomy:     WheatIcon,
  veterinary:   StethoscopeIcon,
  irrigation:   DropletIcon,
  soil_science: LayersIcon,
  aquaculture:  FishIcon,
}

export function DomainIcon({ domain, ...rest }) {
  const Icon = DOMAIN_ICONS[domain]
  return Icon ? <Icon {...rest} /> : null
}

/** Chevron down — collapsible sections */
export function ChevronIcon(props) {
  return wrap(props, <>
    <polyline points="6 9 12 15 18 9" />
  </>)
}
