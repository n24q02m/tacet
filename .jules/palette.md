## 2024-09-25 - Scrollable regions require keyboard accessibility
**Learning:** Containers with overflow-x: auto (like tables) are inaccessible to keyboard users unless they can receive focus.
**Action:** Always add tabindex="0", role="region", an aria-label, and a focus-visible outline to scrollable containers.
