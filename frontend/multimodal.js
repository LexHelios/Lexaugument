/**
 * LEX Multimodal Interface
 * 🔱 JAI MAHAKAAL! - Handles all media types and interactions
 */

class LEXMultimodal {
    constructor() {
        this.attachedMedia = [];
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.cameraStream = null;
        this.drawingCanvas = null;
        this.drawingContext = null;
        this.isDrawing = false;
        
        this.initializeMultimodal();
    }

    initializeMultimodal() {
        this.setupFileInputs();
        this.setupCamera();
        this.setupDrawingPad();
        this.setupDragAndDrop();
        this.setupMediaGallery();
    }

    setupFileInputs() {
        const fileInputs = [
            'file-input-image',
            'file-input-video', 
            'file-input-audio',
            'file-input-document',
            'file-input-code',
            'file-input-3d',
            'file-input-any'
        ];

        fileInputs.forEach(inputId => {
            const input = document.getElementById(inputId);
            if (input) {
                input.addEventListener('change', (e) => this.handleFileSelect(e));
            }
        });
    }

    setupDragAndDrop() {
        const chatMessages = document.getElementById('chat-messages');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            chatMessages.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            chatMessages.addEventListener(eventName, this.highlight.bind(this), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            chatMessages.addEventListener(eventName, this.unhighlight.bind(this), false);
        });

        chatMessages.addEventListener('drop', this.handleDrop.bind(this), false);
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlight(e) {
        e.target.classList.add('drag-highlight');
    }

    unhighlight(e) {
        e.target.classList.remove('drag-highlight');
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        this.handleFiles(files);
    }

    handleFileSelect(e) {
        this.handleFiles(e.target.files);
    }

    handleFiles(files) {
        Array.from(files).forEach(file => {
            this.processFile(file);
        });
    }

    async processFile(file) {
        const mediaItem = {
            id: Date.now() + Math.random(),
            file: file,
            name: file.name,
            type: file.type,
            size: file.size,
            url: URL.createObjectURL(file),
            processed: false
        };

        this.attachedMedia.push(mediaItem);
        this.displayMediaPreview(mediaItem);
        
        // Process based on file type
        if (file.type.startsWith('image/')) {
            await this.processImage(mediaItem);
        } else if (file.type.startsWith('video/')) {
            await this.processVideo(mediaItem);
        } else if (file.type.startsWith('audio/')) {
            await this.processAudio(mediaItem);
        } else if (file.type.includes('pdf') || file.type.includes('document')) {
            await this.processDocument(mediaItem);
        } else {
            await this.processGenericFile(mediaItem);
        }
    }

    displayMediaPreview(mediaItem) {
        const previewArea = document.getElementById('media-preview-area');
        const previewContent = document.getElementById('media-preview-content');
        
        previewArea.style.display = 'block';
        
        const previewElement = document.createElement('div');
        previewElement.className = 'media-preview-item';
        previewElement.dataset.mediaId = mediaItem.id;
        
        let mediaHTML = '';
        
        if (mediaItem.type.startsWith('image/')) {
            mediaHTML = `
                <div class="media-thumbnail">
                    <img src="${mediaItem.url}" alt="${mediaItem.name}">
                </div>
            `;
        } else if (mediaItem.type.startsWith('video/')) {
            mediaHTML = `
                <div class="media-thumbnail">
                    <video src="${mediaItem.url}" controls></video>
                </div>
            `;
        } else if (mediaItem.type.startsWith('audio/')) {
            mediaHTML = `
                <div class="media-thumbnail">
                    <audio src="${mediaItem.url}" controls></audio>
                </div>
            `;
        } else {
            mediaHTML = `
                <div class="media-thumbnail file-thumbnail">
                    <i class="fas fa-file"></i>
                </div>
            `;
        }
        
        previewElement.innerHTML = `
            ${mediaHTML}
            <div class="media-info">
                <span class="media-name">${mediaItem.name}</span>
                <span class="media-size">${this.formatFileSize(mediaItem.size)}</span>
            </div>
            <button class="remove-media-btn" onclick="lexMultimodal.removeMedia('${mediaItem.id}')">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        previewContent.appendChild(previewElement);
    }

    async processImage(mediaItem) {
        // Extract image metadata
        const img = new Image();
        img.onload = () => {
            mediaItem.metadata = {
                width: img.width,
                height: img.height,
                aspectRatio: img.width / img.height
            };
            mediaItem.processed = true;
        };
        img.src = mediaItem.url;
    }

    async processVideo(mediaItem) {
        // Extract video metadata
        const video = document.createElement('video');
        video.onloadedmetadata = () => {
            mediaItem.metadata = {
                width: video.videoWidth,
                height: video.videoHeight,
                duration: video.duration,
                aspectRatio: video.videoWidth / video.videoHeight
            };
            mediaItem.processed = true;
        };
        video.src = mediaItem.url;
    }

    async processAudio(mediaItem) {
        // Extract audio metadata
        const audio = document.createElement('audio');
        audio.onloadedmetadata = () => {
            mediaItem.metadata = {
                duration: audio.duration
            };
            mediaItem.processed = true;
        };
        audio.src = mediaItem.url;
    }

    async processDocument(mediaItem) {
        // For documents, we'll send to LEX for processing
        mediaItem.metadata = {
            needsProcessing: true,
            type: 'document'
        };
        mediaItem.processed = true;
    }

    async processGenericFile(mediaItem) {
        mediaItem.metadata = {
            needsProcessing: true,
            type: 'generic'
        };
        mediaItem.processed = true;
    }

    removeMedia(mediaId) {
        this.attachedMedia = this.attachedMedia.filter(item => item.id != mediaId);
        
        const previewElement = document.querySelector(`[data-media-id="${mediaId}"]`);
        if (previewElement) {
            previewElement.remove();
        }
        
        if (this.attachedMedia.length === 0) {
            document.getElementById('media-preview-area').style.display = 'none';
        }
    }

    clearMediaPreview() {
        this.attachedMedia = [];
        document.getElementById('media-preview-area').style.display = 'none';
        document.getElementById('media-preview-content').innerHTML = '';
    }

    setupCamera() {
        const cameraVideo = document.getElementById('camera-video');
        const captureBtn = document.getElementById('capture-photo');
        const startRecordBtn = document.getElementById('start-video-record');
        const stopRecordBtn = document.getElementById('stop-video-record');
        
        if (captureBtn) {
            captureBtn.addEventListener('click', () => this.capturePhoto());
        }
        
        if (startRecordBtn) {
            startRecordBtn.addEventListener('click', () => this.startVideoRecording());
        }
        
        if (stopRecordBtn) {
            stopRecordBtn.addEventListener('click', () => this.stopVideoRecording());
        }
    }

    async openCamera() {
        try {
            this.cameraStream = await navigator.mediaDevices.getUserMedia({ 
                video: true, 
                audio: true 
            });
            
            const video = document.getElementById('camera-video');
            video.srcObject = this.cameraStream;
            
            document.getElementById('camera-modal').style.display = 'flex';
        } catch (error) {
            console.error('Error accessing camera:', error);
            alert('Could not access camera. Please check permissions.');
        }
    }

    capturePhoto() {
        const video = document.getElementById('camera-video');
        const canvas = document.getElementById('camera-canvas');
        const context = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0);
        
        canvas.toBlob(blob => {
            const file = new File([blob], `photo_${Date.now()}.png`, { type: 'image/png' });
            this.processFile(file);
            this.closeCameraModal();
        });
    }

    startVideoRecording() {
        this.recordedChunks = [];
        this.mediaRecorder = new MediaRecorder(this.cameraStream);
        
        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.recordedChunks.push(event.data);
            }
        };
        
        this.mediaRecorder.onstop = () => {
            const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
            const file = new File([blob], `video_${Date.now()}.webm`, { type: 'video/webm' });
            this.processFile(file);
            this.closeCameraModal();
        };
        
        this.mediaRecorder.start();
        
        document.getElementById('start-video-record').disabled = true;
        document.getElementById('stop-video-record').disabled = false;
    }

    stopVideoRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
        
        document.getElementById('start-video-record').disabled = false;
        document.getElementById('stop-video-record').disabled = true;
    }

    closeCameraModal() {
        if (this.cameraStream) {
            this.cameraStream.getTracks().forEach(track => track.stop());
            this.cameraStream = null;
        }
        
        document.getElementById('camera-modal').style.display = 'none';
    }

    setupDrawingPad() {
        const canvas = document.getElementById('drawing-canvas');
        if (!canvas) return;
        
        this.drawingCanvas = canvas;
        this.drawingContext = canvas.getContext('2d');
        
        // Set up drawing events
        canvas.addEventListener('mousedown', this.startDrawing.bind(this));
        canvas.addEventListener('mousemove', this.draw.bind(this));
        canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        canvas.addEventListener('mouseout', this.stopDrawing.bind(this));
        
        // Touch events for mobile
        canvas.addEventListener('touchstart', this.handleTouch.bind(this));
        canvas.addEventListener('touchmove', this.handleTouch.bind(this));
        canvas.addEventListener('touchend', this.stopDrawing.bind(this));
        
        // Set default drawing style
        this.drawingContext.lineCap = 'round';
        this.drawingContext.lineJoin = 'round';
        this.drawingContext.lineWidth = 5;
        this.drawingContext.strokeStyle = '#ffffff';
    }

    startDrawing(e) {
        this.isDrawing = true;
        this.draw(e);
    }

    draw(e) {
        if (!this.isDrawing) return;
        
        const rect = this.drawingCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        this.drawingContext.lineTo(x, y);
        this.drawingContext.stroke();
        this.drawingContext.beginPath();
        this.drawingContext.moveTo(x, y);
    }

    stopDrawing() {
        if (this.isDrawing) {
            this.drawingContext.beginPath();
            this.isDrawing = false;
        }
    }

    handleTouch(e) {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                         e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        this.drawingCanvas.dispatchEvent(mouseEvent);
    }

    clearCanvas() {
        this.drawingContext.clearRect(0, 0, this.drawingCanvas.width, this.drawingCanvas.height);
    }

    saveDrawing() {
        this.drawingCanvas.toBlob(blob => {
            const file = new File([blob], `drawing_${Date.now()}.png`, { type: 'image/png' });
            this.processFile(file);
            this.closeDrawingModal();
        });
    }

    openDrawingPad() {
        document.getElementById('drawing-modal').style.display = 'flex';
        // Clear canvas when opening
        if (this.drawingContext) {
            this.clearCanvas();
        }
    }

    closeDrawingModal() {
        document.getElementById('drawing-modal').style.display = 'none';
    }

    setupMediaGallery() {
        // Initialize gallery tabs
        const galleryTabs = document.querySelectorAll('.gallery-tab');
        galleryTabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                galleryTabs.forEach(t => t.classList.remove('active'));
                e.target.classList.add('active');
                this.filterGallery(e.target.dataset.tab);
            });
        });
    }

    filterGallery(filter) {
        // Implementation for filtering gallery content
        console.log('Filtering gallery by:', filter);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Screen capture functionality
    async startScreenCapture() {
        try {
            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: true,
                audio: true
            });
            
            // Create video element to capture frame
            const video = document.createElement('video');
            video.srcObject = stream;
            video.play();
            
            video.addEventListener('loadedmetadata', () => {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const context = canvas.getContext('2d');
                context.drawImage(video, 0, 0);
                
                canvas.toBlob(blob => {
                    const file = new File([blob], `screenshot_${Date.now()}.png`, { type: 'image/png' });
                    this.processFile(file);
                });
                
                // Stop the stream
                stream.getTracks().forEach(track => track.stop());
            });
        } catch (error) {
            console.error('Error capturing screen:', error);
            alert('Could not capture screen. Please check permissions.');
        }
    }

    // Get all attached media for sending to LEX
    getAttachedMedia() {
        return this.attachedMedia;
    }

    // Convert media to base64 for API transmission
    async mediaToBase64(mediaItem) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.readAsDataURL(mediaItem.file);
        });
    }
}

// Global functions for HTML onclick handlers
function triggerFileInput(type) {
    const input = document.getElementById(`file-input-${type}`);
    if (input) {
        input.click();
    }
}

function openCamera() {
    lexMultimodal.openCamera();
}

function closeCameraModal() {
    lexMultimodal.closeCameraModal();
}

function openDrawingPad() {
    lexMultimodal.openDrawingPad();
}

function closeDrawingModal() {
    lexMultimodal.closeDrawingModal();
}

function clearCanvas() {
    lexMultimodal.clearCanvas();
}

function saveDrawing() {
    lexMultimodal.saveDrawing();
}

function clearMediaPreview() {
    lexMultimodal.clearMediaPreview();
}

function startScreenCapture() {
    lexMultimodal.startScreenCapture();
}

// Initialize multimodal functionality
const lexMultimodal = new LEXMultimodal();
