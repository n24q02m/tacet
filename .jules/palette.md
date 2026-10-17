## 2024-09-23 - Interactive Element and Scrollable Region Focus Accessibility
**Learning:** Relying solely on brightness changes for button focus is insufficient for WCAG contrast guidelines. Furthermore, scrollable overflow regions (`overflow-x: auto`) require explicit `tabindex="0"`, `role="region"`, and an `aria-label` so keyboard users can discover and scroll them.
**Action:** Always provide explicit `outline` properties with `outline-offset` for `:focus-visible` states, and explicitly configure scrollable containers for keyboard accessibility.
