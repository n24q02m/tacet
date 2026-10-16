## 2024-10-16 - Accessible Scrollable Tables
**Learning:** Found that the app uses a `.scroller` div wrapper around tables for responsive overflow (`overflow-x: auto`), but it lacked keyboard focusability. Screen reader and keyboard-only users could not independently scroll the table content if it overflowed the container.
**Action:** Always add `tabindex="0"`, `role="region"`, and an `aria-label` to custom scrollable containers to ensure they are keyboard focusable and identifiable by assistive technology.
