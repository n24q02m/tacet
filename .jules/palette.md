## 2025-02-27 - Scrollable table accessibility

**Learning:** Tables wrapped in containers with `overflow-x: auto` (like `.scroller`) are inaccessible to keyboard users because they cannot be scrolled without a mouse.
**Action:** Always add `tabindex="0"`, `role="region"`, and a descriptive `aria-label` to scrollable containers, along with a `:focus-visible` outline in CSS to indicate when the container has keyboard focus.
