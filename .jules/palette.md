## 2024-05-24 - Focus States and Accessibility
**Learning:** Implicit browser focus outlines are often insufficient or inconsistent, and color-only state changes (like text color on focus) violate accessibility guidelines for discernible focus rings. Adding explicit `outline` and `outline-offset` provides a much clearer, universally accessible keyboard navigation experience.
**Action:** Always explicitly define `:focus-visible` styles with high-contrast outlines and appropriate offsets for interactive elements (links, buttons) rather than relying on browser defaults or subtle color shifts.
