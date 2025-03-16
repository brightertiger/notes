document.addEventListener('DOMContentLoaded', function() {
    // Get current page path
    const currentPath = window.location.pathname;
    const filename = currentPath.substring(currentPath.lastIndexOf('/') + 1);
    
    // Highlight current page in navigation
    const navLinks = document.querySelectorAll('.sidebar a');
    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        if (linkPath === filename || linkPath.endsWith('/' + filename)) {
            link.style.color = 'white';
            link.style.fontWeight = 'bold';
            link.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
            link.style.borderRadius = '3px';
            link.style.padding = '5px 8px';
        }
    });
    
    // Add table of contents for content pages
    if (filename !== 'index.html' && filename !== '') {
        const contentHeadings = document.querySelectorAll('.content h2');
        if (contentHeadings.length > 0) {
            const toc = document.createElement('div');
            toc.className = 'table-of-contents';
            toc.innerHTML = '<h3>Contents</h3>';
            
            const tocList = document.createElement('ul');
            contentHeadings.forEach((heading, index) => {
                // Add ID to heading if it doesn't have one
                if (!heading.id) {
                    heading.id = 'section-' + index;
                }
                
                const listItem = document.createElement('li');
                const link = document.createElement('a');
                link.href = '#' + heading.id;
                link.textContent = heading.textContent;
                
                listItem.appendChild(link);
                tocList.appendChild(listItem);
            });
            
            toc.appendChild(tocList);
            
            // Insert ToC after the first h1
            const firstHeading = document.querySelector('.content h1');
            if (firstHeading) {
                firstHeading.parentNode.insertBefore(toc, firstHeading.nextSibling);
            }
        }
    }
}); 