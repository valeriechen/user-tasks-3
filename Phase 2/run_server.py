#!/usr/bin/env python3
"""
Simple HTTP server to serve the smooth transitions demo HTML files.
"""
import http.server
import socketserver
import os
import time

PORT = 12001
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def do_GET(self):
        # Redirect root to the demo page
        if self.path == '/':
            self.path = '/smooth_transitions_demo.html'
        return super().do_GET()
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def main():
    print(f"Starting server at http://localhost:{PORT}")
    print(f"View the demo at http://localhost:{PORT}/smooth_transitions_demo.html")
    
    # Allow server to be accessible from any host
    handler = MyHTTPRequestHandler
    httpd = socketserver.TCPServer(("0.0.0.0", PORT), handler)
    
    try:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped.")
        httpd.server_close()

if __name__ == "__main__":
    main()