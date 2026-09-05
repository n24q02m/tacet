## 2024-09-04 - Improve keyboard accessibility for focus states and scrolling
**Learning:** Scrollable overflow containers (like tables) require explicit `tabindex="0"`, `role="region"`, and an `aria-label` to allow keyboard users to navigate and scroll them. Also, focus states using only brightness filters are insufficient for WCAG contrast guidelines; distinct outlines with `outline-offset` are required.
**Action:** Always verify keyboard nav for `.scroller` classes and use explicit `outline: 2px solid var(--accent); outline-offset: 2px;` for global `:focus-visible` styles.
