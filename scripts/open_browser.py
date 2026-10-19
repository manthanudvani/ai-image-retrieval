import time
import socket
import webbrowser

end = time.time() + 300
opened = False
while time.time() < end:
    try:
        s = socket.create_connection(('127.0.0.1', 8000), 1)
        s.close()
        print("Server is up! Opening browser...")
        webbrowser.open('http://127.0.0.1:8000')
        opened = True
        break
    except:
        time.sleep(2)

if not opened:
    print("Timeout waiting for server to start.")
