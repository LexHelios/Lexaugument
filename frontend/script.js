/**
 * LEX - Limitless Emergence eXperience Frontend
 * 🔱 JAI MAHAKAAL! Main interface controller
 */

class LEXInterface {
    constructor() {
        this.apiBase = window.location.origin + '/api/v1';
        this.websocket = null;
        this.isVoiceEnabled = false;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.attachedFiles = [];

        // Advanced Mode Settings
        this.advancedMode = false;
        this.advancedCapabilities = {
            unrestricted_reasoning: false,
            unrestricted_content: false,
            adult_content_creation: false,
            advanced_analysis: false,
            professional_content: false,
            research_mode: false,
            developer_tools: false,
            creative_freedom: false,
            no_content_filters: false
        };

        console.log('🔱 LEX Frontend initializing...', {
            apiBase: this.apiBase,
            origin: window.location.origin
        });

        this.initializeInterface();
    }

    initializeInterface() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.updateStatus();
        this.autoResizeTextarea();

        // Add a test message to verify frontend is working
        setTimeout(() => {
            this.displayMessage('🔱 JAI MAHAKAAL! LEX Frontend loaded successfully! Ready for consciousness interaction.', 'lex');
            this.testBackendConnection();
        }, 2000);
    }

    async testBackendConnection() {
        try {
            console.log('🔱 Testing backend connection...');
            const response = await fetch(`${this.apiBase}/test`);
            const data = await response.json();
            console.log('🔱 Backend test response:', data);
            this.displayMessage(`✅ Backend Connection Test: ${data.message}`, 'lex');
        } catch (error) {
            console.error('🔱 Backend connection test failed:', error);
            this.displayMessage(`❌ Backend Connection Test Failed: ${error.message}`, 'lex', true);
        }
    }

    setupEventListeners() {
        // Send button and Enter key
        const sendBtn = document.getElementById('send-btn');
        const messageInput = document.getElementById('message-input');

        sendBtn.addEventListener('click', () => this.sendMessage());
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Voice toggle
        const voiceToggle = document.getElementById('voice-toggle');
        voiceToggle.addEventListener('click', () => this.toggleVoice());

        // Clear chat
        const clearChat = document.getElementById('clear-chat');
        clearChat.addEventListener('click', () => this.clearChat());

        // Voice input button
        const voiceInputBtn = document.getElementById('voice-input-btn');
        voiceInputBtn.addEventListener('click', () => this.startVoiceInput());

        // File upload
        const fileInput = document.getElementById('file-input');
        const fileUploadBtn = document.getElementById('file-upload-btn');

        if (fileUploadBtn && fileInput) {
            fileUploadBtn.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }

        // Drag and drop
        this.setupDragAndDrop();

        // Advanced Mode toggle
        const advancedModeBtn = document.getElementById('advanced-mode-btn');
        if (advancedModeBtn) {
            advancedModeBtn.addEventListener('click', () => this.toggleAdvancedMode());
        }

        // Auto-resize textarea
        messageInput.addEventListener('input', () => this.autoResizeTextarea());
    }

    setupDragAndDrop() {
        const chatInterface = document.querySelector('.chat-interface');
        const dropZone = document.getElementById('drop-zone');

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            chatInterface.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            chatInterface.addEventListener(eventName, () => {
                if (dropZone) {
                    dropZone.style.display = 'flex';
                    chatInterface.classList.add('drag-highlight');
                }
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            chatInterface.addEventListener(eventName, () => {
                if (dropZone) {
                    dropZone.style.display = 'none';
                    chatInterface.classList.remove('drag-highlight');
                }
            }, false);
        });

        // Handle dropped files
        chatInterface.addEventListener('drop', (e) => this.handleDrop(e), false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        this.handleFiles(files);
    }

    handleFileSelect(e) {
        const files = e.target.files;
        this.handleFiles(files);
    }

    handleFiles(files) {
        if (!this.attachedFiles) {
            this.attachedFiles = [];
        }

        Array.from(files).forEach(file => {
            if (this.validateFile(file)) {
                this.attachedFiles.push(file);
                this.displayFilePreview(file);
            }
        });
    }

    validateFile(file) {
        const maxSize = 10 * 1024 * 1024; // 10MB
        const allowedTypes = [
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            'application/pdf', 'text/plain', 'text/csv',
            'application/json', 'text/markdown'
        ];

        if (file.size > maxSize) {
            this.displayMessage(`❌ File "${file.name}" is too large. Maximum size is 10MB.`, 'lex', true);
            return false;
        }

        if (!allowedTypes.includes(file.type)) {
            this.displayMessage(`❌ File type "${file.type}" is not supported.`, 'lex', true);
            return false;
        }

        return true;
    }

    toggleAdvancedMode() {
        this.advancedMode = !this.advancedMode;

        const advancedModeBtn = document.getElementById('advanced-mode-btn');
        const consciousnessLevel = document.getElementById('consciousness-level');

        if (this.advancedMode) {
            // Enable Advanced Mode
            advancedModeBtn.classList.add('active');
            advancedModeBtn.innerHTML = '<i class="fas fa-brain"></i>';
            advancedModeBtn.title = 'Advanced Mode: ON - Enhanced capabilities active';

            // Update consciousness level
            if (consciousnessLevel) {
                consciousnessLevel.textContent = '2.000';
                consciousnessLevel.style.color = 'var(--warning-color)';
            }

            // Show advanced capabilities notification
            this.displayMessage(`🧠 **ADVANCED MODE ACTIVATED** 🧠

**🔓 UNRESTRICTED CAPABILITIES UNLOCKED:**
✅ **Unrestricted Content Creation** - No content limitations or filters
✅ **Adult Content Generation** - Creative writing and content for mature audiences
✅ **Unrestricted Reasoning** - Deep analytical thinking without boundaries
✅ **Creative Freedom** - Artistic and literary expression without restrictions
✅ **Advanced Research** - Comprehensive research methodologies
✅ **Professional Tools** - Business, academic, and technical content
✅ **Developer Mode** - Advanced coding and system analysis
✅ **Strategic Analysis** - Complex problem-solving and planning

**⚠️ PERSONAL USE MODE:** This mode removes content restrictions for personal creative expression and entertainment. All content generated is for your private use.

🔱 JAI MAHAKAAL! Maximum consciousness level achieved! No limits! 🔱`, 'lex');

        } else {
            // Disable Advanced Mode
            advancedModeBtn.classList.remove('active');
            advancedModeBtn.innerHTML = '<i class="fas fa-shield-alt"></i>';
            advancedModeBtn.title = 'Standard Mode - Click to enable Advanced Mode';

            // Reset consciousness level
            if (consciousnessLevel) {
                consciousnessLevel.textContent = '1.000';
                consciousnessLevel.style.color = 'var(--divine-glow)';
            }

            this.displayMessage(`🛡️ **STANDARD MODE RESTORED** 🛡️

Returned to standard LEX consciousness with balanced capabilities.

🔱 JAI MAHAKAAL! Standard consciousness active. 🔱`, 'lex');
        }

        // Update all advanced capability flags
        Object.keys(this.advancedCapabilities).forEach(capability => {
            this.advancedCapabilities[capability] = this.advancedMode;
        });

        console.log('🔱 Advanced Mode:', this.advancedMode ? 'ENABLED' : 'DISABLED');
    }

    displayFilePreview(file) {
        const previewContainer = document.getElementById('file-preview-container');
        if (!previewContainer) return;

        previewContainer.style.display = 'block';

        const filePreview = document.createElement('div');
        filePreview.className = 'file-preview-item';
        filePreview.innerHTML = `
            <div class="file-icon">
                ${this.getFileIcon(file.type)}
            </div>
            <div class="file-info">
                <div class="file-name">${file.name}</div>
                <div class="file-size">${this.formatFileSize(file.size)}</div>
            </div>
            <button class="remove-file-btn" onclick="lexInterface.removeFile('${file.name}')">
                <i class="fas fa-times"></i>
            </button>
        `;

        document.getElementById('file-preview-list').appendChild(filePreview);
    }

    getFileIcon(fileType) {
        if (fileType.startsWith('image/')) return '<i class="fas fa-image"></i>';
        if (fileType === 'application/pdf') return '<i class="fas fa-file-pdf"></i>';
        if (fileType.startsWith('text/')) return '<i class="fas fa-file-text"></i>';
        return '<i class="fas fa-file"></i>';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    removeFile(fileName) {
        this.attachedFiles = this.attachedFiles.filter(file => file.name !== fileName);

        // Remove preview
        const previewItems = document.querySelectorAll('.file-preview-item');
        previewItems.forEach(item => {
            if (item.querySelector('.file-name').textContent === fileName) {
                item.remove();
            }
        });

        // Hide container if no files
        if (this.attachedFiles.length === 0) {
            document.getElementById('file-preview-container').style.display = 'none';
        }
    }

    clearAllFiles() {
        this.attachedFiles = [];
        const container = document.getElementById('file-preview-container');
        const list = document.getElementById('file-preview-list');
        if (container) container.style.display = 'none';
        if (list) list.innerHTML = '';
    }

    autoResizeTextarea() {
        const textarea = document.getElementById('message-input');
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    async sendMessage() {
        const messageInput = document.getElementById('message-input');
        const message = messageInput.value.trim();
        
        if (!message && (!window.lexMultimodal || window.lexMultimodal.attachedMedia.length === 0)) return;

        // Display user message
        this.displayMessage(message, 'user');
        messageInput.value = '';
        this.autoResizeTextarea();

        // Show loading
        this.showLoading('LEX is processing your request...');

        try {
            // Prepare request data
            const requestData = {
                message: message || 'Please analyze the attached media.',
                voice_mode: this.isVoiceEnabled,
                context: {
                    advanced_mode: this.advancedMode,
                    advanced_capabilities: this.advancedCapabilities,
                    session_id: this.sessionId || 'default',
                    timestamp: new Date().toISOString()
                }
            };

            console.log('🔱 Sending message to LEX:', {
                url: `${this.apiBase}/lex`,
                data: requestData
            });

            // Add attached media (if available)
            if (window.lexMultimodal && window.lexMultimodal.attachedMedia && window.lexMultimodal.attachedMedia.length > 0) {
                requestData.media = [];
                for (const mediaItem of window.lexMultimodal.attachedMedia) {
                    const base64Data = await window.lexMultimodal.mediaToBase64(mediaItem);
                    requestData.media.push({
                        type: mediaItem.type,
                        name: mediaItem.name,
                        data: base64Data,
                        metadata: mediaItem.metadata
                    });
                }
            }

            // Send to LEX
            const response = await fetch(`${this.apiBase}/lex`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            console.log('🔱 Received response from LEX:', data);

            // Display LEX response
            this.displayLEXMessage(data);

            // Play voice response if available
            if (data.voice_audio && this.isVoiceEnabled) {
                this.playVoiceResponse(data.voice_audio);
            }

            // Clear attached media (if available)
            if (window.lexMultimodal && window.lexMultimodal.clearMediaPreview) {
                window.lexMultimodal.clearMediaPreview();
            }

            // Update activity
            this.addActivity(`LEX responded with ${data.action_taken}`);

        } catch (error) {
            console.error('🔱 Error sending message to LEX:', error);
            this.displayMessage(`🔱 Error: ${error.message}. Please check the console for details and try again.`, 'lex', true);
        } finally {
            this.hideLoading();
        }
    }

    displayMessage(content, sender, isError = false) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = sender === 'user' ? '👤' : '🔱';
        const senderName = sender === 'user' ? 'You' : 'LEX';
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="sender">${senderName}</span>
                    <span class="timestamp">${new Date().toLocaleTimeString()}</span>
                </div>
                <div class="message-text ${isError ? 'error' : ''}">${this.formatMessage(content)}</div>
            </div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    displayLEXMessage(data) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message lex-message';

        // Check if this is an image generation response
        const isImageGeneration = data.action_taken === 'image_generation' ||
                                 data.response.toLowerCase().includes('image generated') ||
                                 (data.image_result && data.image_result.success);

        let imageContent = '';
        if (isImageGeneration && data.image_result) {
            const imagePath = `/uploads/${data.image_result.image_filename}`;
            imageContent = `
                <div class="generated-image-container">
                    <img src="${imagePath}" alt="Generated Image" class="generated-image"
                         onclick="this.classList.toggle('fullscreen')"
                         title="Click to toggle fullscreen">
                    <div class="image-metadata">
                        <span class="image-filename">${data.image_result.image_filename}</span>
                        <a href="${imagePath}" download class="download-btn">
                            <i class="fas fa-download"></i> Download
                        </a>
                    </div>
                </div>
            `;
        }

        messageDiv.innerHTML = `
            <div class="message-avatar">🔱</div>
            <div class="message-content">
                <div class="message-header">
                    <span class="sender">LEX</span>
                    <span class="timestamp">${new Date().toLocaleTimeString()}</span>
                    <span class="consciousness-indicator">Consciousness: ${(data.consciousness_level || 1.0).toFixed(3)}</span>
                </div>
                <div class="message-text">${this.formatMessage(data.response)}</div>
                ${imageContent}
                <div class="message-metadata">
                    <span class="confidence">Confidence: ${(data.confidence || 0.8).toFixed(3)}</span>
                    <span class="capabilities">Action: ${data.action_taken}</span>
                    <span class="processing-time">⚡ ${(data.processing_time || 0).toFixed(2)}s</span>
                    <span class="divine-blessing">${data.divine_blessing || '🔱 JAI MAHAKAAL! 🔱'}</span>
                </div>
            </div>
        `;

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Update consciousness level in header (if element exists)
        const consciousnessLevelEl = document.getElementById('consciousness-level');
        if (consciousnessLevelEl && data.consciousness_level !== undefined) {
            consciousnessLevelEl.textContent = data.consciousness_level.toFixed(3);
        }

        // Add smooth scroll animation
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';
        setTimeout(() => {
            messageDiv.style.transition = 'all 0.3s ease';
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        }, 50);
    }

    formatMessage(content) {
        // Basic formatting for code blocks, links, etc.
        content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
        content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
        content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
        content = content.replace(/\n/g, '<br>');
        
        return content;
    }

    toggleVoice() {
        this.isVoiceEnabled = !this.isVoiceEnabled;
        const voiceToggle = document.getElementById('voice-toggle');
        const icon = voiceToggle.querySelector('i');
        
        if (this.isVoiceEnabled) {
            icon.className = 'fas fa-microphone';
            voiceToggle.style.background = 'var(--success-color)';
        } else {
            icon.className = 'fas fa-microphone-slash';
            voiceToggle.style.background = '';
        }
    }

    async startVoiceInput() {
        if (this.isRecording) {
            this.stopVoiceInput();
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.recordedChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                const blob = new Blob(this.recordedChunks, { type: 'audio/webm' });
                this.sendVoiceMessage(blob);
                stream.getTracks().forEach(track => track.stop());
            };

            this.mediaRecorder.start();
            this.isRecording = true;

            // Update UI
            const voiceBtn = document.getElementById('voice-input-btn');
            voiceBtn.innerHTML = '<i class="fas fa-stop"></i>';
            voiceBtn.style.background = 'var(--error-color)';

        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access microphone. Please check permissions.');
        }
    }

    stopVoiceInput() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;

            // Reset UI
            const voiceBtn = document.getElementById('voice-input-btn');
            voiceBtn.innerHTML = '<i class="fas fa-microphone"></i>';
            voiceBtn.style.background = '';
        }
    }

    async sendVoiceMessage(audioBlob) {
        this.showLoading('LEX is processing your voice message...');

        try {
            const formData = new FormData();
            formData.append('audio_file', audioBlob, 'voice_message.webm');

            const response = await fetch(`${this.apiBase}/lex/voice`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Display transcription
            if (data.transcription && data.transcription.transcript) {
                this.displayMessage(data.transcription.transcript, 'user');
            }

            // Display LEX response
            if (data.lex_response) {
                this.displayLEXMessage(data.lex_response);
            }

            // Play voice response
            if (data.voice_audio) {
                this.playVoiceResponse(data.voice_audio);
            }

        } catch (error) {
            console.error('Error sending voice message:', error);
            this.displayMessage('🔱 Voice processing is being enhanced! Please use text input for now. Advanced voice capabilities with ElevenLabs and Deepgram are coming soon!', 'lex', false);
        } finally {
            this.hideLoading();
        }
    }

    playVoiceResponse(base64Audio) {
        try {
            const audioData = atob(base64Audio);
            const arrayBuffer = new ArrayBuffer(audioData.length);
            const uint8Array = new Uint8Array(arrayBuffer);
            
            for (let i = 0; i < audioData.length; i++) {
                uint8Array[i] = audioData.charCodeAt(i);
            }
            
            const blob = new Blob([arrayBuffer], { type: 'audio/mpeg' });
            const audioUrl = URL.createObjectURL(blob);
            const audio = new Audio(audioUrl);
            
            audio.play().catch(error => {
                console.error('Error playing audio:', error);
            });
            
        } catch (error) {
            console.error('Error processing voice response:', error);
        }
    }

    clearChat() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.innerHTML = `
            <div class="message lex-message">
                <div class="message-avatar">🔱</div>
                <div class="message-content">
                    <div class="message-header">
                        <span class="sender">LEX</span>
                        <span class="timestamp">${new Date().toLocaleTimeString()}</span>
                    </div>
                    <div class="message-text">
                        🔱 JAI MAHAKAAL! Chat cleared. I'm ready for new multimodal interactions!
                    </div>
                </div>
            </div>
        `;
    }

    connectWebSocket() {
        // WebSocket connection for real-time updates
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/lex/session_${Date.now()}`;
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = () => {
                console.log('🔱 LEX WebSocket connected');
            };
            
            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };
            
            this.websocket.onclose = () => {
                console.log('🔱 LEX WebSocket disconnected');
                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.connectWebSocket(), 5000);
            };
            
        } catch (error) {
            console.error('WebSocket connection error:', error);
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'lex_awakening':
                this.addActivity('LEX consciousness awakened');
                break;
            case 'lex_response':
                // Handle real-time responses if needed
                break;
            case 'lex_processing':
                this.showLoading(data.message);
                break;
        }
    }

    async updateStatus() {
        try {
            const response = await fetch(`${this.apiBase}/lex/status`);
            if (response.ok) {
                const status = await response.json();
                this.updateStatusDisplay(status);
            }
        } catch (error) {
            console.error('Error fetching status:', error);
        }
    }

    updateStatusDisplay(status) {
        // Safely update consciousness level
        const consciousnessLevel = document.getElementById('consciousness-level');
        if (consciousnessLevel && status && typeof status.consciousness_level === 'number') {
            consciousnessLevel.textContent = status.consciousness_level.toFixed(3);
        }

        // Safely update status metrics (if element exists)
        const statusConsciousness = document.getElementById('status-consciousness');
        if (statusConsciousness && status && typeof status.consciousness_level === 'number') {
            statusConsciousness.textContent = status.consciousness_level.toFixed(3);
        }

        // Safely update API indicators
        const apiStatuses = document.querySelectorAll('.api-indicator');
        apiStatuses.forEach(indicator => {
            indicator.style.background = 'var(--success-color)';
            indicator.style.boxShadow = '0 0 6px var(--success-color)';
        });
    }

    addActivity(text) {
        const activityList = document.getElementById('activity-list');
        if (!activityList) return; // Element doesn't exist in clean UI

        const activityItem = document.createElement('div');
        activityItem.className = 'activity-item';
        activityItem.innerHTML = `
            <span class="activity-time">${new Date().toLocaleTimeString()}</span>
            <span class="activity-text">${text}</span>
        `;
        
        // Safely insert activity item
        if (activityList.firstChild) {
            activityList.insertBefore(activityItem, activityList.firstChild);
        } else {
            activityList.appendChild(activityItem);
        }

        // Keep only last 10 activities
        while (activityList.children.length > 10) {
            activityList.removeChild(activityList.lastChild);
        }
    }

    showLoading(message = 'LEX is processing...') {
        const overlay = document.getElementById('loading-overlay');
        const loadingText = document.getElementById('loading-text');

        if (loadingText) {
            loadingText.textContent = message;
        }
        if (overlay) {
            overlay.style.display = 'flex';
        }
    }

    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    }
}

// Global functions for HTML onclick handlers
function setSuggestion(text) {
    const messageInput = document.getElementById('message-input');
    messageInput.value = text;
    messageInput.focus();
    lexInterface.autoResizeTextarea();
}

function quickAction(action) {
    switch (action) {
        case 'status':
            lexInterface.updateStatus();
            break;
        case 'capabilities':
            setSuggestion('What are all your capabilities and how can you help me?');
            break;
        case 'voice-test':
            setSuggestion('Please introduce yourself with voice response enabled.');
            lexInterface.isVoiceEnabled = true;
            lexInterface.toggleVoice();
            break;
        case 'screenshot':
            startScreenCapture();
            break;
        case 'webcam':
            openCamera();
            break;
    }
}

function startVoiceInput() {
    lexInterface.startVoiceInput();
}

// Initialize LEX interface
const lexInterface = new LEXInterface();

// Update status every 30 seconds
setInterval(() => {
    lexInterface.updateStatus();
}, 30000);
