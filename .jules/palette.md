## 2024-05-24 - Accessible Scroll Containers and Focus States
**Learning:** Using subtle brightness changes for focus states fails WCAG contrast guidelines, and scrollable containers without explicit focus management cannot be navigated by keyboard users.
**Action:** Always add `tabindex="0"`, `role="region"`, and contextual `aria-label`s to scroll containers, and use explicit `outline` with `outline-offset` for interactive element focus states.
