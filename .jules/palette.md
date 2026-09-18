## 2024-05-24 - Interactive Elements Focus and Scrollable Regions
**Learning:** Scrollable containers (`overflow-x: auto`) are not keyboard focusable by default, causing data (like tables) to be inaccessible to keyboard-only users on smaller screens. Additionally, relying on `filter: brightness` for focus indicators does not meet WCAG contrast requirements for visibility.
**Action:** Always add `tabindex="0"`, `role="region"`, and an `aria-label` to overflow containers. Replace subtle hover-style focus states with distinct `outline` and `outline-offset` properties.
