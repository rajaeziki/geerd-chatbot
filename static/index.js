document.getElementById('chat-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const question = document.getElementById('question').value;

    // Display the user's question in the chat history
    addMessageToChat('You', question);

    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: new URLSearchParams({
            'question': question
        })
    })
    .then(response => response.json())
    .then(data => {
        // Display the chatbot's response in the chat history
        addMessageToChat('Chatbot', data.answer);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

// Function to add a message to the chat history
function addMessageToChat(sender, message) {
    const chatHistory = document.getElementById('chat-history');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message');
    messageElement.innerHTML = `<strong>${sender}:</strong> <p>${message}</p>`;
    chatHistory.appendChild(messageElement);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to the bottom
}
