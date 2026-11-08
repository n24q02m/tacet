## 2024-05-24 - Accessible Scroll Containers
**Learning:** The `.scroller` pattern used for responsive tables in this app's design system creates an unscrollable region for keyboard users if it lacks focusability.
**Action:** When using the `.scroller` utility class, always pair it with `tabindex="0"`, `role="region"`, and an `aria-label`, plus ensure `:focus-visible` styles are defined.
