## 2024-05-18 - [Keyboard Accessible Scroll Containers]
**Learning:** Scrollable overflow containers require explicit role="region", tabindex="0", and aria-label to be navigable via keyboard, and subtle focus states like brightness adjustments are insufficient for WCAG compliance.
**Action:** Always pair overflow containers with focusable attributes and apply explicit outline and outline-offset for :focus-visible states across all interactive elements.
