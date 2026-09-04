## 2024-05-18 - Scrollable Region Accessibility
**Learning:** Overflow containers (like `.scroller` for tables) require explicit tab indexing and aria-labels so keyboard users can focus and scroll them without relying on a mouse.
**Action:** Always ensure `.scroller` or similar overflow wrappers have `tabindex="0"`, `role="region"`, and an appropriate `aria-label`. Include visible `:focus-visible` outlines to guide the user's tab flow.
