// Global Preference Tag Removal Handler
async function removePrefTag(prefText, btnElement) {
    const userId = localStorage.getItem('travel_user_id') || "traveler_alex";
    const tagSpan = btnElement.closest('.tag');
    
    if (tagSpan) {
        tagSpan.remove();
    }
    
    try {
        await fetch(`/api/memory/preferences/${encodeURIComponent(userId)}/${encodeURIComponent(prefText)}`, {
            method: 'DELETE'
        });
    } catch (err) {
        console.error("Error deleting preference:", err);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    console.log("✈️ TravelAI App Initialized.");
    const userId = localStorage.getItem('travel_user_id') || "traveler_alex";
    localStorage.setItem('travel_user_id', userId);

    // Tab Switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const target = btn.getAttribute('data-tab');
            const targetElem = document.getElementById(target);
            if (targetElem) {
                targetElem.classList.add('active');
            }
        });
    });

    // Elements
    const tripForm = document.getElementById('trip-form');
    const planBtn = document.getElementById('plan-btn');
    const destinationInput = document.getElementById('destination-input');
    const budgetInput = document.getElementById('budget-input');
    const itineraryOutput = document.getElementById('itinerary-output');
    const loadingSpinner = document.getElementById('loading-spinner');
    const logsList = document.getElementById('logs-list');
    const prefInput = document.getElementById('pref-input');
    const addPrefBtn = document.getElementById('add-pref-btn');
    const prefContainer = document.getElementById('pref-tags-container');

    // Add Preference Event Listener
    if (addPrefBtn && prefInput) {
        addPrefBtn.addEventListener('click', async () => {
            const pref = prefInput.value.trim();
            if (!pref) return;

            try {
                const res = await fetch('/api/memory/preferences', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_id: userId, preference: pref })
                });

                if (res.ok) {
                    const tag = document.createElement('span');
                    tag.className = 'tag';
                    tag.setAttribute('data-pref', pref);
                    tag.innerHTML = `✨ ${pref} <button type="button" class="remove-tag-btn" onclick="removePrefTag('${pref.replace(/'/g, "\\'")}', this)">×</button>`;
                    prefContainer.appendChild(tag);
                    prefInput.value = '';
                }
            } catch (err) {
                console.error("Preference save error:", err);
            }
        });

        prefInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                addPrefBtn.click();
            }
        });
    }

    // Direct Button Click Fallback
    if (planBtn && tripForm) {
        planBtn.addEventListener('click', (e) => {
            console.log("Generate button clicked directly.");
        });
    }

    // Form Submit: Invoke Agent API
    if (tripForm) {
        tripForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            console.log("Form submit triggered.");

            const query = destinationInput ? destinationInput.value.trim() : "";
            const budget = budgetInput ? (parseFloat(budgetInput.value) || null) : null;

            if (!query) {
                alert("Please enter a query or destination!");
                return;
            }

            // Ensure Itinerary Tab is active
            const itineraryTabBtn = document.querySelector('[data-tab="itinerary-tab"]');
            if (itineraryTabBtn) itineraryTabBtn.click();

            // UI state loading
            if (loadingSpinner) loadingSpinner.classList.remove('hidden');
            if (itineraryOutput) itineraryOutput.innerHTML = '';
            if (logsList) logsList.innerHTML = '<li class="log-item">🚀 Initializing LangGraph state graph...</li>';
            
            const rawMdStore = document.getElementById('raw-md-store');
            if (rawMdStore) rawMdStore.textContent = '';

            try {
                console.log("Posting to /api/chat:", { query, user_id: userId, budget });
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        user_id: userId,
                        budget: budget
                    })
                });

                if (!response.ok) {
                    const errText = await response.text();
                    throw new Error(`Server error: ${errText}`);
                }

                if (logsList) logsList.innerHTML = '';
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let done = false;
                
                while (!done) {
                    const { value, done: readerDone } = await reader.read();
                    done = readerDone;
                    
                    if (value) {
                        const chunkStr = decoder.decode(value, { stream: true });
                        const lines = chunkStr.split('\n');
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const jsonStr = line.replace('data: ', '').trim();
                                if (!jsonStr) continue;
                                
                                try {
                                    const data = JSON.parse(jsonStr);
                                    
                                    if (data.type === 'log') {
                                        if (logsList) {
                                            const li = document.createElement('li');
                                            li.className = 'log-item';
                                            li.textContent = data.content;
                                            logsList.appendChild(li);
                                        }
                                    } else if (data.type === 'content') {
                                        if (loadingSpinner) loadingSpinner.classList.add('hidden');
                                        if (itineraryOutput) {
                                            // Handle the placeholder removal on first token
                                            if (itineraryOutput.querySelector('.placeholder-state')) {
                                                itineraryOutput.innerHTML = '';
                                            }
                                            // Create a temporary hidden div to hold raw markdown state
                                            let rawMdStore = document.getElementById('raw-md-store');
                                            if (!rawMdStore) {
                                                rawMdStore = document.createElement('div');
                                                rawMdStore.id = 'raw-md-store';
                                                rawMdStore.style.display = 'none';
                                                document.body.appendChild(rawMdStore);
                                            }
                                            rawMdStore.textContent += data.chunk;
                                            
                                            // Parse and render the current accumulated markdown
                                            if (typeof marked !== 'undefined' && marked.parse) {
                                                itineraryOutput.innerHTML = marked.parse(rawMdStore.textContent);
                                            } else {
                                                itineraryOutput.innerText = rawMdStore.textContent;
                                            }
                                        }
                                    } else if (data.type === 'complete') {
                                        if (loadingSpinner) loadingSpinner.classList.add('hidden');
                                        if (itineraryOutput) {
                                            if (typeof marked !== 'undefined' && marked.parse) {
                                                itineraryOutput.innerHTML = marked.parse(data.itinerary);
                                            } else {
                                                itineraryOutput.innerText = data.itinerary;
                                            }
                                        }
                                    } else if (data.type === 'error') {
                                        if (loadingSpinner) loadingSpinner.classList.add('hidden');
                                        if (itineraryOutput) {
                                            itineraryOutput.innerHTML = `<div style="color:#ef4444; padding:20px; font-weight:bold;">⚠️ Error: ${data.detail}</div>`;
                                        }
                                    }
                                } catch (e) {
                                    console.error("Error parsing SSE JSON:", e, jsonStr);
                                }
                            }
                        }
                    }
                }

            } catch (err) {
                console.error("Fetch Exception:", err);
                if (loadingSpinner) loadingSpinner.classList.add('hidden');
                if (itineraryOutput) {
                    itineraryOutput.innerHTML = `<div style="color:#ef4444; padding:20px; font-weight:bold;">⚠️ Network error: ${err.message}</div>`;
                }
            }
        });
    }
});
