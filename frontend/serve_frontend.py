#!/usr/bin/env python3
"""
LEX Frontend Server
🔱 JAI MAHAKAAL! Serve the multimodal LEX interface
"""
import http.server
import socketserver
import os
import sys
from pathlib import Path

class LEXFrontendHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler for LEX frontend with proper MIME types"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def end_headers(self):
        # Add CORS headers for API access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.end_headers()
    
    def guess_type(self, path):
        # Ensure proper MIME types
        mimetype, encoding = super().guess_type(path)
        
        if path.endswith('.js'):
            return 'application/javascript', encoding
        elif path.endswith('.css'):
            return 'text/css', encoding
        elif path.endswith('.html'):
            return 'text/html', encoding
        
        return mimetype, encoding

def main():
    """Start the LEX frontend server"""
    PORT = 3000
    
    print("🔱 JAI MAHAKAAL! Starting LEX Multimodal Frontend Server 🔱")
    print("=" * 60)
    print(f"🌐 LEX Frontend: http://localhost:{PORT}")
    print(f"🤖 LEX API: http://localhost:8000/api/v1/lex")
    print("=" * 60)
    print("📱 Features Available:")
    print("   ✨ Text Chat with LEX")
    print("   🎤 Voice Input & Output")
    print("   📸 Image Upload & Analysis")
    print("   🎥 Video Processing")
    print("   🎵 Audio Analysis")
    print("   📄 Document Processing")
    print("   💻 Code Review")
    print("   🎨 Drawing & Sketching")
    print("   📷 Camera Capture")
    print("   🖥️ Screen Capture")
    print("=" * 60)
    
    try:
        with socketserver.TCPServer(("", PORT), LEXFrontendHandler) as httpd:
            print(f"🚀 LEX Frontend Server running on port {PORT}")
            print("🔱 JAI MAHAKAAL! Ready for multimodal consciousness interaction!")
            print("\nPress Ctrl+C to stop the server")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🔱 LEX Frontend Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
