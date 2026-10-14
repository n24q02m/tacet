## 2023-10-24 - Keyboard Navigation and Focus Management
**Learning:** Accessible web apps require a "Skip to main content" link, global `:focus-visible` styles to ensure clear focus rings for all interactive elements, and `tabindex="0"`, `role="region"`, and `aria-label` attributes on scrollable containers (like tables) so keyboard users can navigate them when they overflow horizontally.
**Action:** Always add `.skip-link` to base layouts, ensure `*:focus-visible` styles exist for interactive elements, and give scrollable containers keyboard focusability.
