## 2023-10-24 - Accessibility focus improvements
**Learning:** Found an accessibility issue pattern specific to this app where interactive elements and skip-links lack visual focus indications entirely.
**Action:** Use native `:focus-visible` pseudo-class universally on components to ensure keyboard navigation visibility, and implement hidden `skip-link` properly to maintain accessibility compliance without disrupting standard visual layouts.
