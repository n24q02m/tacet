## 2024-09-08 - Accessible Scrollable Regions and Focus States
**Learning:** Scrollable overflow containers (like tables) require explicit keyboard support (`tabindex="0"`, `role="region"`, `aria-label`) to be accessible. Relying solely on brightness changes for focus states is insufficient for WCAG contrast guidelines.
**Action:** Always add proper ARIA labeling and tabindex to scrollable regions, and use distinct `outline` with `outline-offset` for interactive elements' focus states.
