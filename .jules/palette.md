## 2025-02-12 - Keyboard-accessible scrolling regions
**Learning:** Containers with overflow-x: auto (like the .scroller class for responsive tables) trap hidden content from keyboard-only users unless the container is explicitly focusable.
**Action:** Always add tabindex="0", role="region", and an aria-label to custom scrollable wrappers, along with a :focus-visible state.
