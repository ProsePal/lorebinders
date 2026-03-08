# Design Guidelines

LoreBinders is designed with a focus on clear information presentation and a consistent user experience across its CLI and generated reports.

## 🖥 Command-Line Interface (CLI)

The CLI is built using **Typer** and **Rich** to provide a polished terminal experience.

- **Progress Feedback**: All long-running operations (extraction, analysis, summarization) must provide visual progress bars via `rich.progress`.
- **Status Messages**: Informative status messages should be displayed at each stage of the pipeline.
- **Color Palette**: Use **Rich**'s standard color set consistently:
  - **Cyan**: For information and labels.
  - **Green**: For success messages and completed tasks.
  - **Yellow**: For warnings or pending actions.
  - **Red**: For errors.
  - **Bold**: For emphasis and primary headers.
- **Helpful Errors**: When the CLI fails, provide actionable error messages (e.g., missing environment variables, invalid file paths).

## 📄 Story Bible (PDF Report)

The generated PDF is the primary output for the end-user and must be professional and highly readable.

- **Typography**:
  - **Headers**: Clean, bold sans-serif (e.g., Helvetica Bold).
  - **Body**: Readable serif or sans-serif (e.g., Helvetica).
- **Organization**:
  - **Table of Contents**: Clearly link to each category and entity.
  - **Categorization**: Groups entities by their extracted category (Characters, Locations, etc.).
  - **Entity Detail**: Each entity gets its own section with its synthesized summary followed by its extracted traits in a table.
- **Visual Style**:
  - **Consistent Spacing**: Use ample white space between entities and sections.
  - **Tables**: Use professional, clean table styles for trait presentation.
  - **Page Layout**: Include clear headers and footers with the book title and author.

## 🔗 Interface Agnostic Design

While the CLI is the current primary interface, the core engine should remain decoupled from visual design specifics.

- **Structured Output**: The engine returns Pydantic models, allowing any frontend (web, mobile, desktop) to render the Story Bible state according to its own design system.
- **Hook-Driven Updates**: The progress callable mechanism should be the only way the engine communicates status, ensuring it remains generic and interface-independent.
