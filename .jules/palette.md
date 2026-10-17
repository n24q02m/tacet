## 2024-10-11 - Make horizontally scrollable tables keyboard accessible
**Learning:** Regions with overflow-x: auto (like data tables) are inaccessible to keyboard-only users unless explicitly made focusable. Without it, users cannot scroll to see hidden content.
**Action:** Always add tabindex="0", a descriptive aria-label, role="region", and a clear :focus-visible outline to horizontally scrollable containers.
