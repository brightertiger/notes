const fs = require('fs');
const path = require('path');
const marked = require('marked');
const cheerio = require('cheerio');

// Preserve LaTeX equations before markdown processing
function preserveLatex(markdown) {
    // Store all LaTeX expressions
    const latexExpressions = [];

    // Replace inline LaTeX with placeholders
    let processedMarkdown = markdown.replace(/\$([^\$]+)\$/g, (match, latex) => {
        latexExpressions.push(`$${latex}$`);
        return `LATEX_PLACEHOLDER_${latexExpressions.length - 1}`;
    });

    // Replace block LaTeX with placeholders
    processedMarkdown = processedMarkdown.replace(/\$\$([^\$]+)\$\$/g, (match, latex) => {
        latexExpressions.push(`$$${latex}$$`);
        return `LATEX_BLOCK_PLACEHOLDER_${latexExpressions.length - 1}`;
    });

    return { processedMarkdown, latexExpressions };
}

// Restore LaTeX expressions after markdown processing
function restoreLatex(html, latexExpressions) {
    let restoredHtml = html;

    // Restore inline LaTeX
    latexExpressions.forEach((latex, index) => {
        restoredHtml = restoredHtml.replace(
            `LATEX_PLACEHOLDER_${index}`,
            latex
        );
        restoredHtml = restoredHtml.replace(
            `LATEX_BLOCK_PLACEHOLDER_${index}`,
            latex
        );
    });

    return restoredHtml;
}

// Configure marked options for better rendering
marked.setOptions({
    gfm: true, // GitHub Flavored Markdown
    breaks: true, // Convert line breaks to <br>
    highlight: function (code, lang) {
        // You could add syntax highlighting here using highlight.js or prism.js
        return code;
    }
});

// Scan each directory and get all available HTML files
function getAllAvailableFiles() {
    const availableFiles = {};
    const directories = ['eslr', 'general', 'jurafsky', 'probml'];

    directories.forEach(dir => {
        const distDir = path.join('dist', dir);
        if (fs.existsSync(distDir)) {
            const files = fs.readdirSync(distDir)
                .filter(f => f.endsWith('.html'))
                .sort(); // Sort alphabetically
            availableFiles[dir] = files;
        } else {
            availableFiles[dir] = [];
        }
    });

    return availableFiles;
}

// Generate sidebar navigation for a specific section
function generateSectionNavigation(section, availableFiles) {
    const navHTML = `<div class="nav-section">
        <h3>${section.charAt(0).toUpperCase() + section.slice(1)} Notes</h3>
        <ul>
            ${availableFiles[section].map(file => {
        // Extract title from filename (e.g., "eslr-01-regression.html" -> "Regression")
        let title = file.replace(/\.html$/, '');

        // Extract part after first hyphen, if exists
        if (title.indexOf('-') !== -1) {
            const parts = title.split('-');
            if (parts.length > 1) {
                // For probml files, create more descriptive titles
                if (section === 'probml') {
                    // Special cases for probml files
                    if (file === 'probml-00.html') {
                        title = 'Introduction';
                    } else if (file === 'probml-01-introduction.html') {
                        title = 'Introduction (Detailed)';
                    } else if (file === 'probml-03-probability.html') {
                        title = 'Probability (Advanced)';
                    } else {
                        // Extract name after second hyphen if exists
                        const nameParts = parts.slice(1).join('-').split('_');
                        title = nameParts.map(word =>
                            word.charAt(0).toUpperCase() + word.slice(1)
                        ).join(' ');
                    }
                } else {
                    // For other sections, just use the part after the first hyphen
                    title = parts.slice(1).join('-');
                    // Convert underscores to spaces and capitalize
                    title = title.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                }
            }
        }

        // For probml-00.html, title should be Introduction
        if (file === `${section}-00.html`) {
            title = 'Introduction';
        }

        return `<li><a href="${file}">${title}</a></li>`;
    }).join('\n')}
        </ul>
    </div>`;

    return navHTML;
}

// Generate navigation for all sections
function generateNavigation(currentDir, currentFile) {
    const availableFiles = getAllAvailableFiles();
    let navHTML = '';

    // Add home link
    navHTML += `
    <a href="${currentDir ? '../index.html' : 'index.html'}" class="home-link">
        <svg class="home-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
            <polyline points="9 22 9 12 15 12 15 22"></polyline>
        </svg>
    </a>
    <h2>Navigation</h2>
    `;

    // Generate navigation for each section
    ['eslr', 'general', 'jurafsky', 'probml'].forEach(section => {
        if (availableFiles[section] && availableFiles[section].length > 0) {
            navHTML += generateSectionNavigation(section, availableFiles);
        }
    });

    return navHTML;
}

// Create a completely new HTML for each page with fixed navigation
function generateHTML(title, content, currentDir, currentFile) {
    // Get the full index.html content
    const indexHtml = fs.readFileSync('index.html', 'utf8');
    const $ = cheerio.load(indexHtml);

    // Update title and content
    $('title').text(title);
    $('.content').html(content);

    // Replace sidebar with newly generated navigation
    $('.sidebar').html(generateNavigation(currentDir, currentFile));

    // Fix links within the sidebar for proper relative paths
    if (currentDir) {
        $('.sidebar a').each(function () {
            const href = $(this).attr('href');
            if (!href) return;

            if (href === '../index.html' || href === 'index.html') {
                // Leave home link as is
                return;
            }

            // Process links
            if (href.includes('/')) {
                // Links to other sections need '../' prefix
                if (!href.startsWith('../') && !href.startsWith('./') &&
                    !href.startsWith('http') && !href.startsWith('#')) {
                    $(this).attr('href', '../' + href);
                }
            } else if (href !== currentFile) {
                // Links within the same section but to different files
                $(this).attr('href', href);
            } else {
                // Current file should be '#' to avoid page reload
                $(this).attr('href', '#');
                $(this).addClass('active');
            }
        });
    }

    // Add navigation arrows
    if (currentDir) {
        // Gather all HTML files in current directory
        const availableFiles = getAllAvailableFiles();
        const sectionFiles = availableFiles[currentDir] || [];

        // Find current index
        const currentIndex = sectionFiles.indexOf(currentFile);

        // Create navigation HTML
        let navHTML = '<div class="page-navigation">';

        // Previous link
        if (currentIndex > 0) {
            navHTML += `<a href="${sectionFiles[currentIndex - 1]}" class="nav-arrow prev">
                <svg class="arrow" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="19" y1="12" x2="5" y2="12"></line>
                    <polyline points="12 19 5 12 12 5"></polyline>
                </svg>
                Previous
            </a>`;
        } else {
            navHTML += `<span class="nav-arrow prev disabled">
                <svg class="arrow" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="19" y1="12" x2="5" y2="12"></line>
                    <polyline points="12 19 5 12 12 5"></polyline>
                </svg>
                Previous
            </span>`;
        }

        // Next link
        if (currentIndex < sectionFiles.length - 1 && currentIndex !== -1) {
            navHTML += `<a href="${sectionFiles[currentIndex + 1]}" class="nav-arrow next">
                Next
                <svg class="arrow" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                    <polyline points="12 5 19 12 12 19"></polyline>
                </svg>
            </a>`;
        } else {
            navHTML += `<span class="nav-arrow next disabled">
                <svg class="arrow" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                    <polyline points="12 5 19 12 12 19"></polyline>
                </svg>
            </span>`;
        }

        navHTML += '</div>';

        // Add navigation to content
        $('.content').append(navHTML);
    }

    // Update CSS and script paths
    if (currentDir) {
        $('link[rel="stylesheet"]').attr('href', '../styles.css');
        $('script[src="script.js"]').attr('src', '../script.js');
    }

    // Add Google Fonts
    $('head').prepend('<link rel="preconnect" href="https://fonts.googleapis.com">\n' +
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n' +
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">');

    // Add MathJax for equation rendering
    $('head').append(`
    <!-- MathJax for equation rendering -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
    // MathJax configuration
    window.MathJax = {
        tex: {
            inlineMath: [['$', '$']],
            displayMath: [['$$', '$$']]
        },
        svg: {
            fontCache: 'global'
        }
    };
    </script>
    `);

    return $.html();
}

// Process all markdown files
function processMarkdownFiles() {
    if (!fs.existsSync('dist')) {
        fs.mkdirSync('dist', { recursive: true });
    }

    const directories = [
        { source: 'notes/eslr', target: 'dist/eslr', prefix: 'eslr-' },
        { source: 'notes/general', target: 'dist/general', prefix: 'gen-' },
        { source: 'notes/jurafsky', target: 'dist/jurafsky', prefix: 'jfsky-' },
        { source: 'notes/probml', target: 'dist/probml', prefix: 'probml-' }
    ];

    // First, ensure all directories and files exist
    directories.forEach(dir => {
        const sourceDir = dir.source;
        const targetDir = dir.target;

        // Create output directory if it doesn't exist
        if (!fs.existsSync(targetDir)) {
            fs.mkdirSync(targetDir, { recursive: true });
        }

        // Check if source directory exists
        if (!fs.existsSync(sourceDir)) {
            console.warn(`⚠️ Source directory ${sourceDir} does not exist. Creating empty directory.`);
            fs.mkdirSync(sourceDir, { recursive: true });

            // Create placeholder file
            const placeholderPath = path.join(sourceDir, `${dir.prefix}00.md`);
            const placeholderContent = `---
title: "${dir.prefix.charAt(0).toUpperCase() + dir.prefix.slice(1)} Notes"
---

- Placeholder for ${dir.prefix.toUpperCase()} notes
- Content coming soon
`;
            fs.writeFileSync(placeholderPath, placeholderContent);
            console.log(`✅ Created placeholder file ${placeholderPath}`);
        }
    });

    // List all available files for debugging
    console.log("Markdown files available:");
    directories.forEach(dir => {
        const sourceDir = dir.source;
        if (fs.existsSync(sourceDir)) {
            const files = fs.readdirSync(sourceDir).filter(f => f.endsWith('.md'));
            console.log(`- ${dir.prefix}: ${files.join(', ')}`);
        } else {
            console.log(`- ${dir.prefix}: No source directory`);
        }
    });

    // Now process all files
    directories.forEach(dir => {
        const sourceDir = dir.source;
        const targetDir = dir.target;

        try {
            const files = fs.readdirSync(sourceDir).filter(file => file.endsWith('.md'));

            if (files.length === 0) {
                console.warn(`⚠️ No markdown files found in ${sourceDir}`);
            }

            files.forEach(file => {
                const sourcePath = path.join(sourceDir, file);
                const targetFile = file.replace('.md', '.html');
                const targetPath = path.join(targetDir, targetFile);

                try {
                    // Read markdown file
                    const rawMarkdown = fs.readFileSync(sourcePath, 'utf8');

                    // Extract title from markdown
                    const titleMatch = rawMarkdown.match(/^#\s+(.+)$/m) ||
                        rawMarkdown.match(/^---\s+title:\s*["']?([^"'\n]+)["']?/m);
                    let title = file.replace('.md', '');
                    if (titleMatch && titleMatch[1]) {
                        title = titleMatch[1].trim();
                    }

                    // Preserve LaTeX before markdown processing
                    const { processedMarkdown, latexExpressions } = preserveLatex(rawMarkdown);

                    // Convert markdown to HTML
                    let content = marked.parse(processedMarkdown);

                    // Restore LaTeX expressions
                    content = restoreLatex(content, latexExpressions);

                    // Generate final HTML with fully fixed navigation
                    const html = generateHTML(title, content, dir.prefix, targetFile);

                    // Write to file
                    fs.writeFileSync(targetPath, html);
                    console.log(`✅ Converted ${sourcePath} to ${targetPath}`);
                } catch (error) {
                    console.error(`❌ Error processing ${sourcePath}:`, error);
                }
            });
        } catch (error) {
            console.error(`❌ Error reading directory ${sourceDir}:`, error);
        }
    });

    // Also update the index.html with the proper links
    try {
        const indexPath = 'index.html';
        const indexContent = fs.readFileSync(indexPath, 'utf8');
        const $ = cheerio.load(indexContent);

        // Update the ProbML notes section in the sidebar
        const availableFiles = getAllAvailableFiles();

        if (availableFiles.probml && availableFiles.probml.length > 0) {
            // Find the ProbML section in the navigation
            $('.nav-section').each(function () {
                const sectionTitle = $(this).find('h3').text().trim();
                if (sectionTitle === 'ProbML Notes') {
                    // Replace the list with new links
                    const linksList = $(this).find('ul');
                    linksList.empty();

                    availableFiles.probml.forEach(file => {
                        // Extract title from filename
                        let title = file.replace(/\.html$/, '');

                        // Create proper descriptive titles
                        if (file === 'probml-00.html') {
                            title = 'Introduction';
                        } else if (file === 'probml-01-introduction.html') {
                            title = 'Introduction (Detailed)';
                        } else if (file === 'probml-02-probability.html') {
                            title = 'Probability';
                        } else if (file === 'probml-03-probability.html') {
                            title = 'Probability (Advanced)';
                        } else if (file === 'probml-04-statistics.html') {
                            title = 'Statistics';
                        } else if (file === 'probml-05-decision_theory.html') {
                            title = 'Decision Theory';
                        } else if (file === 'probml-06-information_theory.html') {
                            title = 'Information Theory';
                        } else if (file === 'probml-08-optimization.html') {
                            title = 'Optimization';
                        } else if (file === 'probml-09-discriminant_analysis.html') {
                            title = 'Discriminant Analysis';
                        } else if (file === 'probml-10-logistic_regression.html') {
                            title = 'Logistic Regression';
                        } else if (file === 'probml-11-linear_regression.html') {
                            title = 'Linear Regression';
                        } else if (file === 'probml-13-ffnn.html') {
                            title = 'Feed Forward Neural Networks';
                        } else if (file === 'probml-14-cnn.html') {
                            title = 'Convolutional Neural Networks';
                        } else if (file === 'probml-15-rnn.html') {
                            title = 'Recurrent Neural Networks';
                        } else if (file === 'probml-16-exemplar.html') {
                            title = 'Exemplar Methods';
                        } else if (file === 'probml-18-trees.html') {
                            title = 'Trees';
                        } else if (file === 'probml-19-ssl.html') {
                            title = 'Semi-Supervised Learning';
                        } else if (file === 'probml-21-recsys.html') {
                            title = 'Recommendation Systems';
                        } else {
                            // Extract name after second hyphen if exists - fallback for any unlisted files
                            const parts = title.split('-');
                            if (parts.length > 1) {
                                const nameParts = parts.slice(1).join('-').split('_');
                                title = nameParts.map(word =>
                                    word.charAt(0).toUpperCase() + word.slice(1)
                                ).join(' ');
                            }
                        }

                        linksList.append(`<li><a href="dist/probml/${file}">${title}</a></li>`);
                    });
                }
            });

            // Also update the link in the topic cards section
            $('.topic-card').each(function () {
                const cardTitle = $(this).find('h2').text().trim();
                if (cardTitle === 'ProbML Notes') {
                    $(this).find('.button').attr('href', 'dist/probml/probml-00.html');
                }
            });

            // Write the updated index.html
            fs.writeFileSync(indexPath, $.html());
            console.log('✅ Updated index.html with correct ProbML links');
        }

        // Copy the updated index.html to dist/
        fs.copyFileSync(indexPath, 'dist/index.html');
    } catch (error) {
        console.error('❌ Error updating index.html:', error);
    }

    // Copy static files
    try {
        fs.copyFileSync('styles.css', 'dist/styles.css');
        fs.copyFileSync('script.js', 'dist/script.js');
        console.log('✅ Copied static files to dist/');
    } catch (error) {
        console.error('❌ Error copying static files:', error);
    }
}

// Run the conversion
processMarkdownFiles(); 