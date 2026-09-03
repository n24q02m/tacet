## 2025-03-03 - Accessible Scrollable Regions
**Learning:** Overflow containers require an explicit `tabindex="0"`, `role="region"`, and an `aria-label` for keyboard users to be able to scroll them. Standard focus states for buttons only relying on brightness changes are insufficient for WCAG contrast guidelines.
**Action:** Always verify keyboard scrollability on overflow elements and ensure `focus-visible` outlines are distinct and use high-contrast properties like `outline`.
