## 2025-02-24 - Scrollable Region Accessibility
**Learning:** Tables in scrollable overflow containers (`overflow-x: auto`) require explicit `tabindex="0"`, `role="region"`, and an ARIA label to allow keyboard-only users to scroll and navigate the hidden content.
**Action:** When creating `.scroller` wrappers for tables, always apply `tabindex="0"`, `role="region"`, and an `aria-labelledby` linking to the section heading, along with a visible focus outline.
