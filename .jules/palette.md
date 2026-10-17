## 2024-05-24 - Improve keyboard accessibility for overflow tables
**Learning:** Scrollable responsive containers (`overflow-x: auto`) for tables are entirely skipped by keyboard navigation, making their hidden content inaccessible to keyboard users.
**Action:** Always add `tabindex="0"`, `role="region"`, and a descriptive `aria-label` to these `.scroller` containers so they can receive focus and be scrolled via keyboard arrows.
