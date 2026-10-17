## 2024-05-18 - Scrollable containers need keyboard access
**Learning:** Containers with `overflow-x: auto` (like data tables) are inaccessible to keyboard users unless explicitly given a `tabindex="0"`.
**Action:** Always add `tabindex="0"`, `role="region"`, and an `aria-label` to scrollable containers, along with a `:focus-visible` outline.
