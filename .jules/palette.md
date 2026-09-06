## 2024-09-05 - Explicit focus states and accessible scrollable containers
**Learning:** The application lacked keyboard focus support for its scrollable overflow table containers, which hinders accessibility.
**Action:** Always apply explicit `outline` properties with `outline-offset` for `:focus-visible` states, and explicitly add `tabindex="0"`, `role="region"`, and a context-appropriate `aria-label` to overflow containers to ensure keyboard navigability.
