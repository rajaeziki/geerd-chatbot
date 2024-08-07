document.getElementById('upload-form').addEventListener('submit', function(event) {
    event.preventDefault();

    let formData = new FormData();
    formData.append('document', document.getElementById('file-input').files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    }).then(response => response.json())
      .then(data => {
          console.log('Document uploaded successfully:', data);
          alert('Document uploaded successfully!');
      })
      .catch(error => {
          console.error('Error uploading document:', error);
      });
});

document.getElementById('send-btn').addEventListener('click', function() {
    let userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') {
        return;
    }

    let chatLog = document.getElementById('chat-log');
    chatLog.innerHTML += `<div><strong>You:</strong> ${userInput}</div>`;

    fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userInput })
    }).then(response => response.json())
      .then(data => {
          chatLog.innerHTML += `<div><strong>Chatbot:</strong> ${data.answer}</div>`;
          document.getElementById('user-input').value = '';
      })
      .catch(error => {
          console.error('Error asking question:', error);
      });
});
