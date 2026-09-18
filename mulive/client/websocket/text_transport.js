export class TextTransport {
    constructor(url) {
      this.url = url;
      this.ws = null;
      this.onMessageHandler = () => {};
    }
  
    connect() {
      this.ws = new WebSocket(this.url);
      this.ws.onopen = () => console.log("[WS] Connected");
      this.ws.onclose = () => console.log("[WS] Disconnected");
      this.ws.onerror = (err) => console.error("[WS] Error:", err);
      this.ws.onmessage = (event) => {
        console.log("[WS] Message:", event.data);
        this.onMessageHandler(event.data);
      };
    }
  
    sendMessage(text) {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        console.log("[WS] Sending:", text);
        this.ws.send(text);
      }
    }
  
    onMessage(callback) {
      this.onMessageHandler = callback;
    }
  }
  
