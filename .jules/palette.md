## 2024-05-24 - Keyboard Accessibility for Scrollable Containers
**Learning:** Scrollable overflow containers (like tables) require an explicit tabindex="0", role="region", and aria-label so keyboard users can navigate and scroll them.
**Action:** Always inspect overflow containers and add these attributes with context-appropriate labels.
