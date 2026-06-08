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
      <svg viewBox="0 0 128 128" className="brand-mark-svg" aria-hidden="true">
        <defs>
          <linearGradient id="vf-flame" x1="0" y1="1" x2="0" y2="0">
            <stop offset="0" stopColor="#C8401A" />
            <stop offset="0.5" stopColor="#F2832A" />
            <stop offset="1" stopColor="#FFC83D" />
          </linearGradient>
        </defs>
        {/* flame */}
        <path d="M64 18 C 80 40, 84 58, 64 78 C 44 58, 48 40, 64 18 Z" fill="url(#vf-flame)" />
        {/* inner highlight */}
        <path d="M64 40 C 71 52, 72 62, 64 72 C 56 62, 57 52, 64 40 Z" fill="#FFE9A8" opacity="0.9" />
        {/* diya bowl */}
        <path d="M34 84 Q 64 92 94 84 Q 86 104 64 104 Q 42 104 34 84 Z" fill="#B6471C" />
      </svg>
    </span>
  );
}
