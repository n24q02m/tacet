## 2024-05-23 - Keyboard accessibility for scrollable regions and focus outlines
**Learning:** Scrollable overflow containers (like tables) hide content from keyboard users unless explicitly made focusable. Focus visibility must also have distinct contrast.
**Action:** Always add `tabindex="0"`, `role="region"`, and an `aria-label`/`aria-labelledby` to `.scroller` components, and ensure interactive elements have clear `outline` and `outline-offset` overrides for `:focus-visible`.
