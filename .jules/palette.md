## 2024-05-24 - Scrollable Container Keyboard Accessibility
**Learning:** Scrollable overflow containers (like tables) lack keyboard scrollability by default, making them inaccessible for keyboard users when horizontal overflow occurs.
**Action:** Always explicitly add `tabindex="0"`, `role="region"`, and `aria-label` to scrollable containers to enable keyboard navigation and screen reader support.
