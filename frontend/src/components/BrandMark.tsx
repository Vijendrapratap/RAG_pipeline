/**
 * Inline brand mark — a small saffron diya (lamp-flame), rendered as inline SVG
 * so it never depends on an external asset (no /logo.png|svg fetch that can 404
 * or get cached stale). Used for the sidebar, login, welcome screen, and chat
 * avatar. Sizing/background come from the className the caller passes
 * (.brand-mark / .welcome-mark / .bubble-avatar + their --logo variants).
 */
interface Props {
  className?: string;
  title?: string;
  decorative?: boolean;
}

export function BrandMark({ className, title = "Vishvas Foundation", decorative }: Props) {
  return (
    <span
      className={className}
      role={decorative ? undefined : "img"}
      aria-label={decorative ? undefined : title}
      aria-hidden={decorative ? true : undefined}
    >
      {/* Layered saffron diya flame — solid fills (no gradients / shared ids),
          so it renders identically everywhere. */}
      <svg viewBox="0 0 64 64" className="brand-mark-svg" aria-hidden="true">
        {/* diya bowl */}
        <path d="M13 45 Q32 53 51 45 Q45 60 32 60 Q19 60 13 45 Z" fill="#9A3D18" />
        {/* outer flame */}
        <path d="M32 5 C 45 21, 47 34, 32 49 C 17 34, 19 21, 32 5 Z" fill="#E2641F" />
        {/* mid flame */}
        <path d="M32 13 C 41 25, 42 35, 32 46 C 22 35, 23 25, 32 13 Z" fill="#F59A2E" />
        {/* inner flame */}
        <path d="M32 24 C 38 31, 38 39, 32 45 C 26 39, 26 31, 32 24 Z" fill="#FFE08A" />
      </svg>
    </span>
  );
}
