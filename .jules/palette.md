## 2024-05-24 - Keyboard Accessibility for Scrollable Containers
**Learning:** Scrollable overflow containers (like data tables) are inaccessible to keyboard-only users unless they can receive focus.
**Action:** Always add tabindex="0", role="region", and an appropriate aria-label to containers with overflow-x: auto, and ensure they have distinct :focus-visible styles.
