const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Use pandoc to convert docx to markdown
const docxPath = process.argv[2];
if (!docxPath) {
    console.log('Usage: node read_proposal.js <path-to-docx>');
    process.exit(1);
}

try {
    const result = execSync(`pandoc -t markdown "${docxPath}"`, { encoding: 'utf8' });
    console.log(result);
} catch (e) {
    console.error('Error:', e.message);
}
