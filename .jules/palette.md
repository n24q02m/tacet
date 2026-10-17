## 2024-10-01 - Add Skip-to-Content Link & Global Focus Indicators
**Learning:** Users relying on keyboard navigation need explicit and fast ways to bypass repeated navigation structures, and default focus indicators are often inadequate. Standardizing `:focus-visible` globally and providing an invisible-until-focused `.skip-link` instantly elevates baseline accessibility.
**Action:** Always include a `.skip-link` as the first actionable element in the `<body>` and use global `:focus-visible` to guarantee all interactive elements indicate focus correctly.
