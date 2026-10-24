## 2024-10-24 - Accessible scrollable regions and toggle states
**Learning:** Elements with `overflow: auto` that contain content like tables need explicit `tabindex="0"` and `role="region"` with an `aria-label` so keyboard users can focus and scroll them, and screen readers can identify the region. Theme toggle buttons must maintain an accurate `aria-pressed` state dynamically.
**Action:** Always add focusable regions for scrolling containers that don't inherently take focus, and ensure JavaScript updates ARIA states alongside visual classes.
