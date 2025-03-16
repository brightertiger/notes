// index.js
const fs = require('fs');
const path = require('path');
const marked = require('marked');
const matter = require('gray-matter');
const glob = require('glob');
const mkdirp = require('mkdirp');

// Configuration
const config = {
    sourceDir: 'notes',
    outputDir: 'dist',
    siteTitle: 'My Notes',
    dateFormat: { year: 'numeric', month: 'long', day: 'numeric' }
};

// Create title from filename
function createTitleFromFilename(filename) {
    const basename = path.basename(filename, path.extname(filename));
    return basename
        .replace(/-/g, ' ')
        .replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Get folder name
function getFolderName(folderPath) {
    return path.basename(folderPath);
}

// Get folder display name
function getFolderDisplayName(folderPath) {
    const name = getFolderName(folderPath);
    return name
        .replace(/-/g, ' ')
        .replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Load and process markdown file
function processMarkdownFile(filePath) {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const { content, data } = matter(fileContent);
    const html = marked.parse(content);

    const title = data.title || createTitleFromFilename(filePath);
    const date = data.date ? new Date(data.date).toLocaleDateString('en-US', config.dateFormat) : null;

    return {
        title,
        date,
        html,
        path: filePath,
        relativePath: path.relative(config.sourceDir, filePath),
        outputPath: path.join(
            config.outputDir,
            path.relative(config.sourceDir, filePath).replace(/\.md$/, '.html')
        )
    };
}

// Generate HTML page for a note
function generateNotePage(note) {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${note.title} | ${config.siteTitle}</title>
  <link rel="stylesheet" href="/css/style.css">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
  <!-- MathJax for LaTeX support -->
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <script>
    MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      },
      options: {
        enableMenu: false
      }
    };
  </script>
</head>
<body>
  <div class="container">
    <header>
      <h1>${note.title}</h1>
      ${note.date ? `<p class="date">Last updated: ${note.date}</p>` : ''}
      <a href="/" class="home-link">← Back to Home</a>
    </header>
    <main class="content">
      ${note.html}
    </main>
    <footer>
      <p>Generated with Markdown Notes Static Site Generator</p>
    </footer>
  </div>
</body>
</html>
  `;
}

// Generate folder index page
function generateFolderIndexPage(folderPath, notes) {
    const folderName = getFolderName(folderPath);
    const folderDisplayName = getFolderDisplayName(folderPath);

    const notesLinks = notes.map(note => {
        const relativePath = path.relative(config.sourceDir, note.path);
        const htmlPath = relativePath.replace(/\.md$/, '.html');
        return `<li><a href="/${htmlPath}">${note.title}</a></li>`;
    }).join('\n      ');

    return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${folderDisplayName} | ${config.siteTitle}</title>
  <link rel="stylesheet" href="/css/style.css">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
</head>
<body>
  <div class="container">
    <header>
      <h1>${folderDisplayName}</h1>
      <a href="/" class="home-link">← Back to Home</a>
    </header>
    <main>
      <ul class="notes-list">
      ${notesLinks}
      </ul>
    </main>
    <footer>
      <p>Generated with Markdown Notes Static Site Generator</p>
    </footer>
  </div>
</body>
</html>
  `;
}

// Generate home page with folder cards
function generateHomePage(folders) {
    const folderCards = folders.map(folder => {
        const folderName = getFolderName(folder.path);
        const folderDisplayName = getFolderDisplayName(folder.path);
        const noteCount = folder.notes.length;

        return `
    <div class="card">
      <h2>${folderDisplayName}</h2>
      <p>${noteCount} note${noteCount !== 1 ? 's' : ''}</p>
      <a href="/${folderName}/index.html" class="card-link">View Notes</a>
    </div>`;
    }).join('\n    ');

    return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${config.siteTitle}</title>
  <link rel="stylesheet" href="/css/style.css">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
</head>
<body>
  <div class="container">
    <header>
      <h1>${config.siteTitle}</h1>
    </header>
    <main>
      <div class="cards-grid">
    ${folderCards}
      </div>
    </main>
    <footer>
      <p>Generated with Markdown Notes Static Site Generator</p>
    </footer>
  </div>
</body>
</html>
  `;
}

// CSS styles
const cssContent = `
:root {
  --primary-color: #000;
  --bg-color: #fff;
  --text-color: #333;
  --light-gray: #f5f5f5;
  --border-color: #ddd;
  --hover-color: #f0f0f0;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Inter', sans-serif;
  line-height: 1.6;
  color: var(--text-color);
  background-color: var(--bg-color);
}

.container {
  max-width: 900px;
  margin: 0 auto;
  padding: 2rem 1rem;
}

header {
  margin-bottom: 2rem;
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 1rem;
}

h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  color: var(--primary-color);
}

h2 {
  font-size: 1.8rem;
  margin-bottom: 0.5rem;
  color: var(--primary-color);
}

h3, h4, h5, h6 {
  margin-top: 1.5rem;
  margin-bottom: 0.5rem;
  color: var(--primary-color);
}

p {
  margin-bottom: 1.5rem;
}

a {
  color: var(--primary-color);
  text-decoration: none;
  border-bottom: 1px solid var(--border-color);
  transition: border-color 0.2s;
}

a:hover {
  border-color: var(--primary-color);
}

.home-link {
  display: inline-block;
  margin-top: 1rem;
  font-size: 0.9rem;
  border-bottom: none;
  padding: 0.5rem 0;
}

.home-link:hover {
  text-decoration: underline;
}

.date {
  font-size: 0.9rem;
  color: #666;
  margin-bottom: 1rem;
}

.content {
  line-height: 1.8;
}

.content h2 {
  margin-top: 2rem;
}

.content ul, .content ol {
  margin-bottom: 1.5rem;
  padding-left: 1.5rem;
}

.content img {
  max-width: 100%;
  height: auto;
  margin: 1.5rem 0;
}

.content code {
  background-color: var(--light-gray);
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
  font-family: monospace;
}

.content pre {
  background-color: var(--light-gray);
  padding: 1rem;
  border-radius: 3px;
  overflow-x: auto;
  margin-bottom: 1.5rem;
}

.content pre code {
  background-color: transparent;
  padding: 0;
}

.content blockquote {
  border-left: 3px solid var(--primary-color);
  padding-left: 1rem;
  font-style: italic;
  margin-bottom: 1.5rem;
}

footer {
  margin-top: 3rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border-color);
  text-align: center;
  font-size: 0.9rem;
  color: #666;
}

/* Cards for folders */
.cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 1.5rem;
}

.card {
  background-color: var(--light-gray);
  border-radius: 5px;
  padding: 1.5rem;
  transition: transform 0.2s;
}

.card:hover {
  transform: translateY(-3px);
}

.card h2 {
  margin-top: 0;
}

.card p {
  font-size: 0.9rem;
  margin-bottom: 1rem;
}

.card-link {
  display: inline-block;
  font-weight: 500;
  border-bottom: none;
}

.card-link:hover {
  text-decoration: underline;
}

/* Notes list */
.notes-list {
  list-style: none;
}

.notes-list li {
  margin-bottom: 0.75rem;
  padding: 0.75rem;
  border-radius: 3px;
}

.notes-list li:hover {
  background-color: var(--hover-color);
}

.notes-list a {
  display: block;
  border-bottom: none;
}

/* Responsive adjustments */
@media (max-width: 600px) {
  h1 {
    font-size: 2rem;
  }
  
  .cards-grid {
    grid-template-columns: 1fr;
  }
}
`;

// Main function to generate the site
async function generateSite() {
    console.log('Generating static site...');

    // Ensure output directory exists
    mkdirp.sync(config.outputDir);
    mkdirp.sync(path.join(config.outputDir, 'css'));

    // Create CSS file
    fs.writeFileSync(path.join(config.outputDir, 'css', 'style.css'), cssContent);
    console.log('Created CSS styles');

    // Find all markdown files in the source directory
    const markdownFiles = glob.sync(path.join(config.sourceDir, '**/*.md'));
    console.log(`Found ${markdownFiles.length} markdown files`);

    // Process all markdown files
    const notes = markdownFiles.map(processMarkdownFile);

    // Group notes by folder
    const folderMap = new Map();

    notes.forEach(note => {
        const relativePath = path.relative(config.sourceDir, note.path);
        const folderPath = path.dirname(relativePath);

        if (!folderMap.has(folderPath)) {
            folderMap.set(folderPath, {
                path: folderPath,
                notes: []
            });
        }

        folderMap.get(folderPath).notes.push(note);
    });

    const folders = Array.from(folderMap.values());
    console.log(`Found ${folders.length} folders`);

    // Generate note pages
    for (const note of notes) {
        const outputPath = note.outputPath;
        const outputDir = path.dirname(outputPath);

        mkdirp.sync(outputDir);

        const html = generateNotePage(note);
        fs.writeFileSync(outputPath, html);
        console.log(`Generated note page: ${outputPath}`);
    }

    // Generate folder index pages
    for (const folder of folders) {
        if (folder.path === '.') continue; // Skip root folder

        const outputPath = path.join(config.outputDir, folder.path, 'index.html');
        const outputDir = path.dirname(outputPath);

        mkdirp.sync(outputDir);

        const html = generateFolderIndexPage(folder.path, folder.notes);
        fs.writeFileSync(outputPath, html);
        console.log(`Generated folder index page: ${outputPath}`);
    }

    // Generate home page
    const nonRootFolders = folders.filter(folder => folder.path !== '.');
    const homePage = generateHomePage(nonRootFolders);
    fs.writeFileSync(path.join(config.outputDir, 'index.html'), homePage);
    console.log('Generated home page');

    console.log('Static site generation complete!');
}

// Run the generator
generateSite().catch(console.error);