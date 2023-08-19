// contentScript.js

// Function to simulate web scraping behavior
function scrapeWebpage() {
    // Simulate opening a URL
}

// Function to get selected text
function getSelectedText() {
    var selectedText = window.getSelection().toString();
    return selectedText;
}

// Call the functions

// Wait for some time (similar to driver.implicitly_wait)
setTimeout(function() {
    console.log(document.title); // Print the page title

    var selectedText = getSelectedText();
    console.log(selectedText); // Print selected text

    // Here you can send the selected text back to the background script
    // or perform any other actions you need.
}, 5000); // Wait for 5 seconds (adjust the time as needed)
