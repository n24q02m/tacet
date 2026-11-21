## 2024-05-28 - Keyboard Accessible Scrollable Regions
**Learning:** Found a scrollable table container (`overflow-x: auto`) without a `tabindex`. Keyboard users cannot scroll overflowing content unless the container itself is focusable.
**Action:** Always add `tabindex="0"`, `role="region"`, and a descriptive `aria-label` to custom scrollable containers, along with a distinct `:focus-visible` outline.
