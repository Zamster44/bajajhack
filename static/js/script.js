// DOM elements
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const suggestions = document.querySelectorAll('.suggestion-chip');

// Add message to chat
function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', 'p-3');
    
    if (isUser) {
        messageDiv.classList.add('user-message');
        messageDiv.innerHTML = `
            <div class="d-flex align-items-center mb-2">
                <div class="bg-secondary rounded-circle p-2 me-2">
                    <i class="fas fa-user text-white"></i>
                </div>
                <strong>You</strong>
            </div>
            <p>${content}</p>
        `;
    } else {
        messageDiv.classList.add('bot-message');
        messageDiv.innerHTML = `
            <div class="d-flex align-items-center mb-2">
                <div class="bg-primary rounded-circle p-2 me-2">
                    <i class="fas fa-robot text-white"></i>
                </div>
                <strong>Analyst Assistant</strong>
            </div>
            <div>${content}</div>
        `;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Show loading indicator
function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.classList.add('message', 'bot-message', 'p-3', 'loading-indicator');
    loadingDiv.id = 'loading-indicator';
    loadingDiv.innerHTML = `
        <div class="d-flex align-items-center">
            <div class="bg-primary rounded-circle p-2 me-2">
                <i class="fas fa-robot text-white"></i>
            </div>
            <strong>Analyst Assistant</strong>
        </div>
        <div class="mt-2 text-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2 mb-0">Analyzing your request...</p>
        </div>
    `;
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Hide loading indicator
function hideLoading() {
    const loading = document.getElementById('loading-indicator');
    if (loading) loading.remove();
}

// Handle API response
function handleResponse(response) {
    hideLoading();
    
    if (response.error) {
        addMessage(`Error: ${response.error}`);
        return;
    }
    
    switch(response.type) {
        case 'stock_metric':
            addMessage(`
                The <strong>${response.metric}</strong> stock price for ${response.period} was 
                <strong>₹${response.value.toFixed(2)}</strong>
            `);
            break;
            
        case 'comparison':
            const comp = response.comparison;
            const changeIcon = comp.change_pct >= 0 ? '▲' : '▼';
            const changeClass = comp.change_pct >= 0 ? 'text-success' : 'text-danger';
            
            addMessage(`
                <div class="comparison-card">
                    <div class="row">
                        <div class="col-md-6">
                            <h6>${comp.period1.label}</h6>
                            <h4>₹${comp.period1.avg.toFixed(2)}</h4>
                        </div>
                        <div class="col-md-6">
                            <h6>${comp.period2.label}</h6>
                            <h4>₹${comp.period2.avg.toFixed(2)}</h4>
                        </div>
                    </div>
                    <div class="mt-2 ${changeClass}">
                        <i class="fas fa-arrow-${comp.change_pct >= 0 ? 'up' : 'down'} me-1"></i>
                        Change: ${changeIcon} ${Math.abs(comp.change_pct).toFixed(2)}%
                    </div>
                </div>
            `);
            break;
            
        case 'commentary':
            addMessage(`
                <div class="bg-light p-3 border-start border-4 border-primary">
                    <h5><i class="fas fa-file-contract me-2"></i> CFO Commentary Draft</h5>
                    <div class="mt-2">${response.content}</div>
                </div>
            `);
            break;
            
        case 'table':
            addMessage(`
                <div class="bg-light p-3">
                    <h5><i class="fas fa-table me-2"></i> Requested Information</h5>
                    <div class="mt-2">${response.content}</div>
                </div>
            `);
            break;
            
        case 'answer':
            let contextBadges = '';
            if (response.context_sources) {
                contextBadges = `
                    <div class="mt-3">
                        <small>Sources: 
                            ${response.context_sources.map(q => 
                                `<span class="context-badge">${q}</span>`
                            ).join('')}
                        </small>
                    </div>
                `;
            }
            
            addMessage(`
                ${response.content}
                ${contextBadges}
            `);
            break;
            
        default:
            addMessage("I couldn't process your request. Please try again.");
    }
}

// Send query to backend
async function sendQuery() {
    const query = userInput.value.trim();
    if (!query) return;
    
    // Add user message to chat
    addMessage(query, true);
    userInput.value = '';
    showLoading();
    
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({query})
        });
        
        const data = await response.json();
        handleResponse(data);
    } catch (error) {
        hideLoading();
        addMessage("Error connecting to the server. Please try again.");
        console.error(error);
    }
}

// Event listeners
sendBtn.addEventListener('click', sendQuery);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendQuery();
});

// Insert suggestion
function insertSuggestion(text) {
    userInput.value = text;
    userInput.focus();
}

// Add click event to all suggestions
suggestions.forEach(suggestion => {
    suggestion.addEventListener('click', () => {
        insertSuggestion(suggestion.getAttribute('onclick').match(/'([^']+)'/)[1]);
    });
});