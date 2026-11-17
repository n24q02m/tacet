## 2024-05-24 - Accessibility improvements for keyboard users
**Learning:** The navigation theme toggle button lacked visual focus indication for keyboard users, and there was no mechanism to bypass the navigation directly to the main content.
**Action:** When adding standard interactive elements, explicitly ensure they define `:focus-visible` styles matching `:hover`, and provide a hidden but accessible skip-link to standardise main content access for screen readers and keyboard power-users.
