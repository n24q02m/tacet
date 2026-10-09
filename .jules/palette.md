## 2024-05-23 - Keyboard accessibility for scrollable regions
**Learning:** Responsive tables wrapped in `overflow-x: auto` containers are completely inaccessible to keyboard users when content overflows, because the container itself isn't focusable for arrow-key scrolling.
**Action:** Always add `tabindex="0"`, `role="region"`, and an `aria-label` to overflow containers, and provide a clear `:focus-visible` outline to ensure sighted keyboard users know they have focus.
