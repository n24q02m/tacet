## 2024-05-14 - Keyboard Accessibility for Overflow Containers
**Learning:** Found that overflow-x containers (like `.scroller`) for data tables require explicit tabindex="0", role="region", and aria-labelledby to be keyboard accessible.
**Action:** Always add keyboard accessibility attributes to scrollable containers and ensure a visible focus state with proper contrast and outline-offset.
