## 2024-10-02 - Smooth transitions and explicit focus outlines
**Learning:** By default, state changes on hover and focus feel abrupt. Relying on default browser focus outlines often leads to poor visibility and inconsistent branding. Additionally, some components (like `.theme-toggle`) were entirely missing a `:focus-visible` state.
**Action:** Always add `transition` properties (e.g., `0.15s ease`) to interactive elements, and explicitly define `:focus-visible` outlines using the theme's accent color to ensure keyboard accessibility is both highly visible and on-brand.
