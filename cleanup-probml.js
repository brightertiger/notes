const fs = require('fs');
const path = require('path');

// Delete the probml directory if it exists
const probmlDir = path.join('dist', 'probml');
if (fs.existsSync(probmlDir)) {
    // Delete all files in probml directory
    const files = fs.readdirSync(probmlDir);
    files.forEach(file => {
        const filePath = path.join(probmlDir, file);
        fs.unlinkSync(filePath);
        console.log(`Deleted ${filePath}`);
    });
    
    // Remove the directory itself
    fs.rmdirSync(probmlDir);
    console.log(`Deleted directory ${probmlDir}`);
} else {
    console.log(`Directory ${probmlDir} does not exist, no cleanup needed`);
}

// Remove ProbML references from index.html
const indexPath = 'index.html';
if (fs.existsSync(indexPath)) {
    let indexContent = fs.readFileSync(indexPath, 'utf8');
    
    // Remove ProbML sidebar navigation
    indexContent = indexContent.replace(/<div class="nav-section">\s*<h3>ProbML Notes<\/h3>[\s\S]*?<\/div>/g, '');
    
    // Remove ProbML topic card
    indexContent = indexContent.replace(/<div class="topic-card">\s*<h2>ProbML Notes<\/h2>[\s\S]*?<\/div>/g, '');
    
    fs.writeFileSync(indexPath, indexContent);
    console.log(`Updated ${indexPath} - removed ProbML references`);
}

console.log('Cleanup complete. Run the conversion script to regenerate ProbML files.'); 