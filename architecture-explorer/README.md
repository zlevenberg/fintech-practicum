# Architecture Explorer

Standalone interactive diagram of the `parts-inflation` codebase. **Does not modify** any application code.

## Open

```bash
open architecture-explorer/index.html
```

Or from this folder, serve locally:

```bash
python3 -m http.server 8765
# then visit http://localhost:8765
```

## Features

- **Data flow** view with animated edges — click stages to drill into sub-steps
- Detail panel with inputs/outputs, key functions, and KaTeX formulas
- **Formulas** atlas of every V2 methodology equation
- **Modules** map jumping back into the flow
- Pan (drag) · zoom (scroll) · Esc / breadcrumb to go up
