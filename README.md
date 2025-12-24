# ML Notes

A personal collection of machine learning notes organized as a navigable book-style website.

## Content Areas

| Source | Description |
|--------|-------------|
| **ESLR** | Notes from "Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman |
| **General** | Foundational machine learning concepts, algorithms, and techniques |
| **Jurafsky** | Notes from "Speech and Language Processing" by Jurafsky & Martin |
| **ProbML** | Probabilistic Machine Learning by Kevin Murphy |

## Quick Start

```bash
npm install
npm run build
```

Open `index.html` in your browser to view your notes.

## Adding New Notes

### 1. Add a markdown file

Create a new `.md` file in the appropriate source folder under `notes/`:

```
notes/
├── eslr/
├── general/
│   └── gen-09-neural-networks.md  ← new note
├── jurafsky/
└── probml/
```

**Naming convention:** Use `prefix-##-topic.md` format where:
- `prefix` matches the folder (e.g., `gen`, `eslr`, `probml`, `jfsky`)
- `##` is a number for ordering (e.g., `01`, `02`, `09`)
- `topic` describes the content

### 2. Write your note

```markdown
---
title: "Neural Networks"  # Optional: title override
---

# Neural Networks

Your content here...

## Section 1

Use **bold**, *italics*, and `code`.

## Math Support

Inline math: $E = mc^2$

Block math:
$$
\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}
$$
```

### 3. Rebuild the site

```bash
npm run build
```

The generator automatically:
- Extracts the title from frontmatter, first heading, or filename
- Orders notes by the number prefix
- Generates prev/next navigation
- Updates the sidebar

## Adding a New Source

1. Create a new folder under `notes/`:

```bash
mkdir notes/deeplearning
```

2. Add the source to `config.json`:

```json
{
  "sources": {
    "deeplearning": {
      "name": "Deep Learning",
      "shortName": "DL",
      "description": "Notes on neural networks and deep learning",
      "icon": "🔮",
      "color": "rose"
    }
  }
}
```

Available colors: `peach`, `mint`, `lavender`, `butter`, `rose`, `sage`

3. Add notes and rebuild:

```bash
npm run build
```

## Project Structure

```
notes/
├── config.json          # Source configuration
├── index.js             # Site generator
├── package.json
├── css/
│   └── style.css        # Styles (auto-regenerated)
├── notes/               # Your markdown files
│   ├── eslr/
│   ├── general/
│   ├── jurafsky/
│   └── probml/
└── [generated HTML files]
```

## Features

- **Book-style navigation**: Sidebar with collapsible sections
- **Previous/Next links**: Navigate sequentially through notes
- **Mobile responsive**: Hamburger menu on small screens
- **LaTeX support**: MathJax for mathematical notation
- **Pastel theme**: Modern, minimalistic design

## Hosting on GitHub Pages

1. Build the site: `npm run build`
2. Commit and push all files including generated HTML
3. In repository Settings → Pages, set source to your main branch
4. Your site will be at `https://username.github.io/notes/`

## License

MIT
