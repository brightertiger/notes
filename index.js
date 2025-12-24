const fs = require('fs');
const path = require('path');
const marked = require('marked');
const matter = require('gray-matter');
const glob = require('glob');
const mkdirp = require('mkdirp');

const config = JSON.parse(fs.readFileSync('config.json', 'utf8'));

const buildConfig = {
    sourceDir: 'notes',
    outputDir: '.',
    dateFormat: { year: 'numeric', month: 'long', day: 'numeric' }
};

function createTitleFromFilename(filename) {
    const basename = path.basename(filename, path.extname(filename));
    const parts = basename.split('-');
    if (parts.length > 1 && /^\d+$/.test(parts[0])) {
        parts.shift();
    }
    return parts
        .join(' ')
        .replace(/_/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function processMarkdownFile(filePath) {
    const fileContent = fs.readFileSync(filePath, 'utf8');
    const { content, data } = matter(fileContent);
    const html = marked.parse(content);

    let titleFromContent = '';
    const headingMatch = content.match(/^#\s+(.+)$/m);
    if (headingMatch) {
        titleFromContent = headingMatch[1].trim();
    }

    const title = data.title || titleFromContent || createTitleFromFilename(filePath);
    const date = data.date ? new Date(data.date).toLocaleDateString('en-US', buildConfig.dateFormat) : null;

    const basename = path.basename(filePath, '.md');
    const orderMatch = basename.match(/^(\d+)/);
    const order = orderMatch ? parseInt(orderMatch[1]) : 999;

    return {
        title,
        date,
        html,
        order,
        path: filePath,
        relativePath: path.relative(buildConfig.sourceDir, filePath),
        outputPath: path.join(
            buildConfig.outputDir,
            path.relative(buildConfig.sourceDir, filePath).replace(/\.md$/, '.html')
        )
    };
}

function generateSidebar(folders, currentSourceKey = null, currentNotePath = null, isSourceIndex = false) {
    let navSections = '';

    folders.forEach(folder => {
        const sourceKey = folder.path.toLowerCase();
        const sourceConfig = config.sources[sourceKey] || {};
        const displayName = sourceConfig.shortName || sourceKey.toUpperCase();
        const isCurrentSource = sourceKey === currentSourceKey;

        const noteItems = folder.notes
            .sort((a, b) => a.order - b.order)
            .map(note => {
                const noteFilename = path.basename(note.path, '.md') + '.html';
                const isActive = currentNotePath === note.path;
                let href;
                if (currentNotePath) {
                    href = currentSourceKey === sourceKey ? noteFilename : `../${sourceKey}/${noteFilename}`;
                } else if (isSourceIndex) {
                    href = currentSourceKey === sourceKey ? noteFilename : `../${sourceKey}/${noteFilename}`;
                } else {
                    href = `${sourceKey}/${noteFilename}`;
                }
                return `<li class="nav-item"><a href="${href}"${isActive ? ' class="active"' : ''}>${note.title}</a></li>`;
            })
            .join('\n            ');

        navSections += `
        <div class="nav-section ${sourceKey}">
          <div class="nav-section-title" onclick="this.nextElementSibling.classList.toggle('collapsed')">
            <span class="icon">${sourceConfig.icon || '📄'}</span>
            ${displayName}
          </div>
          <ul class="nav-items${!isCurrentSource && currentSourceKey ? ' collapsed' : ''}">
            ${noteItems}
          </ul>
        </div>`;
    });

    const homeLink = (currentNotePath || isSourceIndex) ? '../index.html' : 'index.html';
    return `
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-header">
        <a href="${homeLink}" class="sidebar-logo">${config.siteTitle}</a>
      </div>
      <nav class="sidebar-nav">
        ${navSections}
      </nav>
    </aside>`;
}

function generateMobileHeader() {
    return `
    <header class="mobile-header">
      <a href="../index.html" class="sidebar-logo">${config.siteTitle}</a>
      <button class="mobile-menu-btn" onclick="document.getElementById('sidebar').classList.toggle('open'); document.getElementById('overlay').classList.toggle('visible')">
        <span></span>
        <span></span>
        <span></span>
      </button>
    </header>
    <div class="sidebar-overlay" id="overlay" onclick="document.getElementById('sidebar').classList.remove('open'); this.classList.remove('visible')"></div>`;
}

function generateNotePage(note, folders, sourceKey, prevNote, nextNote) {
    const sourceConfig = config.sources[sourceKey] || {};
    const sidebar = generateSidebar(folders, sourceKey, note.path);
    const mobileHeader = generateMobileHeader();

    let navigation = '';
    if (prevNote || nextNote) {
        navigation = '<nav class="page-navigation">';
        if (prevNote) {
            navigation += `
        <a href="${path.basename(prevNote.path, '.md')}.html" class="nav-link prev">
          <span class="nav-link-label">← Previous</span>
          <span class="nav-link-title">${prevNote.title}</span>
        </a>`;
        } else {
            navigation += '<div></div>';
        }
        if (nextNote) {
            navigation += `
        <a href="${path.basename(nextNote.path, '.md')}.html" class="nav-link next">
          <span class="nav-link-label">Next →</span>
          <span class="nav-link-title">${nextNote.title}</span>
        </a>`;
        }
        navigation += '</nav>';
    }

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${note.title} | ${config.siteTitle}</title>
  <link rel="stylesheet" href="../css/style.css">
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <script>
    MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      },
      options: { enableMenu: false }
    };
  </script>
</head>
<body>
  <div class="layout">
    ${sidebar}
    ${mobileHeader}
    <main class="main-content">
      <div class="content-wrapper">
        <header class="page-header">
          <div class="breadcrumb">
            <a href="../index.html">Home</a>
            <span>/</span>
            <a href="index.html">${sourceConfig.shortName || sourceKey.toUpperCase()}</a>
          </div>
          <h1 class="page-title">${note.title}</h1>
          ${note.date ? `<div class="page-meta"><span class="tag">${sourceConfig.shortName || sourceKey}</span><span>Updated ${note.date}</span></div>` : `<div class="page-meta"><span class="tag">${sourceConfig.shortName || sourceKey}</span></div>`}
        </header>
        <article class="content">
          ${note.html}
        </article>
        ${navigation}
      </div>
    </main>
  </div>
  <script>
    document.addEventListener('scroll', function() {
      const btn = document.querySelector('.back-to-top');
      if (btn) btn.classList.toggle('visible', window.scrollY > 300);
    });
  </script>
</body>
</html>`;
}

function generateSourceIndexPage(sourceKey, notes, folders) {
    const sourceConfig = config.sources[sourceKey] || {};
    const displayName = sourceConfig.name || sourceKey.toUpperCase();
    const sidebar = generateSidebar(folders, sourceKey, null, true);

    const sortedNotes = notes.sort((a, b) => a.order - b.order);

    const noteItems = sortedNotes.map((note, index) => {
        const noteFilename = path.basename(note.path, '.md') + '.html';
        return `
        <li class="notes-list-item">
          <a href="${noteFilename}">
            <span class="number">${String(index + 1).padStart(2, '0')}</span>
            <span class="title">${note.title}</span>
            <span class="arrow">→</span>
          </a>
        </li>`;
    }).join('');

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${displayName} | ${config.siteTitle}</title>
  <link rel="stylesheet" href="../css/style.css">
</head>
<body>
  <div class="layout">
    ${sidebar}
    ${generateMobileHeader()}
    <main class="main-content">
      <div class="content-wrapper">
        <header class="page-header">
          <div class="breadcrumb">
            <a href="../index.html">Home</a>
            <span>/</span>
            <span>${sourceConfig.shortName || sourceKey.toUpperCase()}</span>
          </div>
          <h1 class="page-title">${displayName}</h1>
          <p style="color: var(--text-secondary); margin-top: 0.5rem;">${sourceConfig.description || ''}</p>
        </header>
        <ul class="notes-list">
          ${noteItems}
        </ul>
      </div>
    </main>
  </div>
</body>
</html>`;
}

function generateHomePage(folders) {
    const sourceCards = folders.map(folder => {
        const sourceKey = folder.path.toLowerCase();
        const sourceConfig = config.sources[sourceKey] || {};
        const noteCount = folder.notes.length;

        return `
      <a href="${sourceKey}/index.html" class="source-card ${sourceKey}">
        <div class="source-card-icon">${sourceConfig.icon || '📄'}</div>
        <h2 class="source-card-title">${sourceConfig.name || sourceKey.toUpperCase()}</h2>
        <p class="source-card-description">${sourceConfig.description || ''}</p>
        <span class="source-card-count">${noteCount} note${noteCount !== 1 ? 's' : ''}</span>
      </a>`;
    }).join('');

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${config.siteTitle}</title>
  <link rel="stylesheet" href="css/style.css">
</head>
<body>
  <div class="home-container">
    <header class="home-header">
      <h1 class="home-title">${config.siteTitle}</h1>
      <p class="home-subtitle">${config.siteDescription}</p>
    </header>
    <div class="sources-grid">
      ${sourceCards}
    </div>
  </div>
</body>
</html>`;
}

async function generateSite() {
    console.log('🚀 Generating static site...\n');

    mkdirp.sync(buildConfig.outputDir);
    mkdirp.sync(path.join(buildConfig.outputDir, 'css'));

    const markdownFiles = glob.sync(path.join(buildConfig.sourceDir, '**/*.md'));
    console.log(`📝 Found ${markdownFiles.length} markdown files`);

    const notes = markdownFiles.map(processMarkdownFile);

    const folderMap = new Map();
    notes.forEach(note => {
        const relativePath = path.relative(buildConfig.sourceDir, note.path);
        const folderPath = path.dirname(relativePath);

        if (!folderMap.has(folderPath)) {
            folderMap.set(folderPath, { path: folderPath, notes: [] });
        }
        folderMap.get(folderPath).notes.push(note);
    });

    const folders = Array.from(folderMap.values()).filter(f => f.path !== '.');
    console.log(`📁 Found ${folders.length} source folders\n`);

    for (const folder of folders) {
        const sourceKey = folder.path.toLowerCase();
        const sortedNotes = folder.notes.sort((a, b) => a.order - b.order);

        for (let i = 0; i < sortedNotes.length; i++) {
            const note = sortedNotes[i];
            const prevNote = i > 0 ? sortedNotes[i - 1] : null;
            const nextNote = i < sortedNotes.length - 1 ? sortedNotes[i + 1] : null;

            const outputPath = note.outputPath;
            const outputDir = path.dirname(outputPath);

            mkdirp.sync(outputDir);

            const html = generateNotePage(note, folders, sourceKey, prevNote, nextNote);
            fs.writeFileSync(outputPath, html);
        }
        console.log(`  ✓ ${sourceKey}: ${sortedNotes.length} notes`);

        const indexPath = path.join(buildConfig.outputDir, folder.path, 'index.html');
        const indexHtml = generateSourceIndexPage(sourceKey, folder.notes, folders);
        fs.writeFileSync(indexPath, indexHtml);
    }

    const homePage = generateHomePage(folders);
    fs.writeFileSync(path.join(buildConfig.outputDir, 'index.html'), homePage);
    console.log('\n✅ Site generation complete!');
    console.log(`\n📖 Open index.html to view your notes`);
}

generateSite().catch(console.error);
