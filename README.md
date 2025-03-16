# Machine Learning Notes

This repository contains my personal notes on various machine learning topics. The notes are organized by subject area and presented as a static website for easy navigation.

## Content Areas

- **ESLR**: Notes from "Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman
- **General ML**: General machine learning concepts, algorithms, and techniques
- **NLP**: Notes from "Speech and Language Processing" by Jurafsky & Martin
- **Probability**: Fundamentals of probability for machine learning

## Compiling the Site

The site is generated using a simple Node.js static site generator:

1. Make sure you have Node.js installed
2. Install dependencies: `npm install`
3. Run the generator: `node index.js`
4. Open `index.html` in your browser to view the site

## Adding New Notes

To add new notes:

1. Create a Markdown file in the appropriate subdirectory under `notes/`
   - Example: `notes/general/gen-12-my-new-topic.md`

2. Use the following format for your notes:
   ```markdown
   # Your Note Title

   ## First Section
   
   Content here...
   
   ## Second Section
   
   More content...
   ```

3. Optionally, add YAML frontmatter at the top:
   ```markdown
   ---
   title: "Custom Title (Optional)"
   date: "2023-08-01"
   ---
   ```

4. Re-run `node index.js` to regenerate the site

The generator will extract the title from either:
- The frontmatter `title` field
- The first heading in the document
- The filename (as a fallback)

## Navigation

- The homepage lists all subject areas
- Each subject area has an index page listing all notes in that area
- You can navigate back to the home page from any note

## Prerequisites

- [Node.js](https://nodejs.org/) (v14 or newer)
- npm (comes with Node.js)

## Getting Started

1. Clone this repository or download the files
2. Place your markdown notes in a folder called `notes` in the project root

```
project-root/
├── index.js
├── package.json
└── notes/
    ├── folder1/
    │   ├── note1.md
    │   └── note2.md
    └── folder2/
        ├── note3.md
        └── note4.md
```

3. Install dependencies:

```bash
npm install
```

4. Build the static site:

```bash
npm run build
```

5. The built site will be in the `dist` directory
6. To preview locally:

```bash
npm run serve
```

Then open http://localhost:8080 in your browser.

## Hosting on GitHub Pages

1. Create a GitHub repository for your notes

2. After building the site, copy the contents of the `dist` directory to your repository's root directory:

```bash
cp -r dist/* .
```

3. Commit and push your changes:

```bash
git add .
git commit -m "Add static site"
git push
```

4. Go to your repository settings on GitHub
5. Scroll down to the "GitHub Pages" section
6. Set the source to the branch where you pushed the site (usually `main` or `master`)
7. Your site will be published at `https://yourusername.github.io/repository-name/`

## Customization

You can customize the site by editing the `config` object at the top of `index.js`:

```javascript
const config = {
  sourceDir: 'notes',        // Directory containing your markdown files
  outputDir: 'dist',         // Output directory for the static site
  siteTitle: 'My Notes',     // Site title
  dateFormat: { year: 'numeric', month: 'long', day: 'numeric' }
};
```

## Markdown Features

This generator supports standard markdown syntax plus:

- Front matter for custom titles and dates:
  ```
  ---
  title: My Custom Title
  date: 2023-01-01
  ---
  ```
- LaTeX math equations:
  - Inline: `$E = mc^2$`
  - Block: `$$E = mc^2$$`

## Technical Details

The generator uses:

- `marked` for Markdown parsing
- `gray-matter` for parsing YAML front matter
- `glob` for file discovery
- `mkdirp` for directory creation
- MathJax for rendering LaTeX equations

## License

MIT