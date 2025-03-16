const fs = require('fs');
const path = require('path');
const cheerio = require('cheerio');

const probmlDir = path.join('dist', 'probml');
const files = fs.readdirSync(probmlDir).filter(f => f.endsWith('.html'));

// First create a map of proper titles for all ProbML files
const titleMap = {
    'probml-00': 'Introduction',
    'probml-01-introduction': 'Introduction (Detailed)',
    'probml-02-probability': 'Probability',
    'probml-03-probability': 'Probability (Advanced)',
    'probml-04-statistics': 'Statistics',
    'probml-05-decision_theory': 'Decision Theory',
    'probml-06-information_theory': 'Information Theory',
    'probml-08-optimization': 'Optimization',
    'probml-09-discriminant_analysis': 'Discriminant Analysis',
    'probml-10-logistic_regression': 'Logistic Regression',
    'probml-11-linear_regression': 'Linear Regression',
    'probml-13-ffnn': 'Feed Forward Neural Networks',
    'probml-14-cnn': 'Convolutional Neural Networks',
    'probml-15-rnn': 'Recurrent Neural Networks',
    'probml-16-exemplar': 'Exemplar Methods',
    'probml-18-trees': 'Trees',
    'probml-19-ssl': 'Semi-Supervised Learning',
    'probml-21-recsys': 'Recommendation Systems'
};

files.forEach(file => {
    const filePath = path.join(probmlDir, file);
    console.log(`Processing ${filePath}`);
    
    let html = fs.readFileSync(filePath, 'utf8');
    const $ = cheerio.load(html);
    
    // Fix all navigation sections more aggressively
    $('.nav-section').each(function() {
        const sectionTitle = $(this).find('h3').text().trim();
        
        // Completely rebuild each section's links
        if (sectionTitle === 'Eslr Notes') {
            const list = $(this).find('ul');
            list.empty();
            list.append(`
                <li><a href="../eslr/eslr-00.html">Introduction</a></li>
                <li><a href="../eslr/eslr-01-regression.html">Regression</a></li>
                <li><a href="../eslr/eslr-02-classification.html">Classification</a></li>
                <li><a href="../eslr/eslr-03-kernel-methods.html">Kernel Methods</a></li>
                <li><a href="../eslr/eslr-04-model-assessment.html">Model Assessment</a></li>
                <li><a href="../eslr/eslr-08-model-selection.html">Model Selection</a></li>
                <li><a href="../eslr/eslr-09-additive-models.html">Additive Models</a></li>
                <li><a href="../eslr/eslr-10-boosting.html">Boosting</a></li>
                <li><a href="../eslr/eslr-15-random-forest.html">Random Forest</a></li>
            `);
        }
        
        if (sectionTitle === 'General Notes') {
            const list = $(this).find('ul');
            list.empty();
            list.append(`
                <li><a href="../general/gen-00.html">Introduction</a></li>
                <li><a href="../general/gen-01-basic-statistics.html">Basic Statistics</a></li>
                <li><a href="../general/gen-02-decision_trees.html">Decision Trees</a></li>
                <li><a href="../general/gen-03-boosting.html">Boosting</a></li>
                <li><a href="../general/gen-04-xgboost.html">XGBoost</a></li>
                <li><a href="../general/gen-05-clustering.html">Clustering</a></li>
                <li><a href="../general/gen-06-support_vector_machines.html">Support Vector Machines</a></li>
                <li><a href="../general/gen-07-dimensionality_reduction.html">Dimensionality Reduction</a></li>
                <li><a href="../general/gen-08-regression.html">Regression</a></li>
            `);
        }
        
        if (sectionTitle === 'Jurafsky Notes') {
            const list = $(this).find('ul');
            list.empty();
            list.append(`
                <li><a href="../jurafsky/jfsky-00.html">Introduction</a></li>
                <li><a href="../jurafsky/jfsky-01-regex.html">Regex</a></li>
                <li><a href="../jurafsky/jfsky-02-tokenization.html">Tokenization</a></li>
                <li><a href="../jurafsky/jfsky-03-vectors.html">Vectors</a></li>
                <li><a href="../jurafsky/jfsky-04-sequence.html">Sequence</a></li>
                <li><a href="../jurafsky/jfsky-05-encoder.html">Encoder</a></li>
                <li><a href="../jurafsky/jfsky-06-transfer.html">Transfer</a></li>
            `);
        }
        
        if (sectionTitle === 'Probml Notes') {
            // Create a formatted list of all ProbML links
            const currentFile = file.replace('.html', '');
            const list = $(this).find('ul');
            list.empty();
            
            // Add all ProbML links with proper titles
            Object.keys(titleMap).forEach(fileBase => {
                if (fileBase === currentFile) {
                    // Current page should have the 'active' class and href="#"
                    list.append(`<li><a href="#" class="active">${titleMap[fileBase]}</a></li>`);
                } else {
                    list.append(`<li><a href="${fileBase}.html">${titleMap[fileBase]}</a></li>`);
                }
            });
        }
    });
    
    // Fix the home link
    $('.home-link').attr('href', '../index.html');
    
    // Fix the previous/next navigation
    // This requires knowledge of the file order
    const allFiles = Object.keys(titleMap).map(base => base + '.html');
    const currentIndex = allFiles.indexOf(file);
    
    if (currentIndex !== -1) {
        const prevFile = currentIndex > 0 ? allFiles[currentIndex - 1] : null;
        const nextFile = currentIndex < allFiles.length - 1 ? allFiles[currentIndex + 1] : null;
        
        const pageNav = $('.page-navigation');
        pageNav.empty();
        
        // Add previous link or disabled button
        if (prevFile) {
            pageNav.append(`
                <a href="${prevFile}" class="nav-arrow prev">
                    <svg class="arrow" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="19" y1="12" x2="5" y2="12"></line>
                        <polyline points="12 19 5 12 12 5"></polyline>
                    </svg>
                    Previous
                </a>
            `);
        } else {
            pageNav.append(`
                <span class="nav-arrow prev disabled">
                    <svg class="arrow" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="19" y1="12" x2="5" y2="12"></line>
                        <polyline points="12 19 5 12 12 5"></polyline>
                    </svg>
                    Previous
                </span>
            `);
        }
        
        // Add next link or disabled button
        if (nextFile) {
            pageNav.append(`
                <a href="${nextFile}" class="nav-arrow next">
                    Next
                    <svg class="arrow" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                        <polyline points="12 5 19 12 12 19"></polyline>
                    </svg>
                </a>
            `);
        } else {
            pageNav.append(`
                <span class="nav-arrow next disabled">
                    Next
                    <svg class="arrow" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="5" y1="12" x2="19" y2="12"></line>
                        <polyline points="12 5 19 12 12 19"></polyline>
                    </svg>
                </span>
            `);
        }
    }

    // Write the updated HTML back to the file
    fs.writeFileSync(filePath, $.html());
    console.log(`Fixed ${file}`);
});

console.log('All ProbML HTML files updated!'); 