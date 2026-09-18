## 2024-05-24 - Explicit Focus Styles
**Learning:** Custom components like buttons, links, and scrollable tables relying on subtle or default browser focus states fail WCAG visibility guidelines, and scrollable regions need explicit labels/tabindex for keyboard users.
**Action:** Always define explicit `outline` and `outline-offset` properties in `global.css` for all interactive elements to ensure clear keyboard navigation visibility, and add `tabindex="0"`, `role="region"`, and `aria-label` to overflow scroll containers.
