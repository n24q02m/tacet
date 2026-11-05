## 2024-05-24 - Scrollable Region Focus
**Learning:** Containers with overflow-x: auto trap content for keyboard users on narrow viewports if they are not explicitly focusable.
**Action:** Add tabindex="0", role="region", an aria-label, and a :focus-visible outline to overflow containers to ensure keyboard accessibility.
