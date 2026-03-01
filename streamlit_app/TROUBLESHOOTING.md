# Troubleshooting Remote Access

## HTTP ERROR 502

This error means the server is not responding. Common causes:

### 1. Container Not Running

Check if the container is running:
```bash
docker ps | grep twi-translator
```

If not running, check logs:
```bash
docker logs twi-translator
```

### 2. Port Not Exposed

Check port mapping:
```bash
docker port twi-translator
```

Should show: `8501/tcp -> 0.0.0.0:8501`

### 3. Firewall Blocking

Check if port 8501 is open:
```bash
# On the server
sudo ufw status
sudo ufw allow 8501/tcp

# Or iptables
sudo iptables -L | grep 8501
sudo iptables -A INPUT -p tcp --dport 8501 -j ACCEPT
```

### 4. Wrong Server Address

Ensure Streamlit binds to `0.0.0.0` not `localhost`:

In `.streamlit/config.toml`:
```toml
[server]
address = "0.0.0.0"
port = 8501
```

### 5. Test Local Access First

On the server itself:
```bash
curl http://localhost:8501/_stcore/health
```

Should return: `{"status": "ok"}`

### 6. Check Network Interface

Find your server's IP:
```bash
ip addr show
# or
hostname -I
```

Access using the IP:
```
http://<SERVER_IP>:8501
```

### 7. Docker Network Issues

Try using host network mode:
```bash
docker run -d \
    --name twi-translator \
    --network host \
    -v "$(pwd)/../Attention_is_All_You_Need/results:/app/models:ro" \
    -v "$(pwd)/../Attention_is_All_You_Need/data:/app/data:ro" \
    twi-chinese-translator:latest
```

### 8. SELinux (RHEL/CentOS/Fedora)

If using SELinux, add the `:Z` flag:
```bash
-v "$(pwd)/../Attention_is_All_You_Need/results:/app/models:ro,Z"
```

## Quick Fix Script

Run this on the server:
```bash
#!/bin/bash
# Fix permissions and restart

docker stop twi-translator 2>/dev/null
docker rm twi-translator 2>/dev/null

docker run -d \
    --name twi-translator \
    -p 0.0.0.0:8501:8501 \
    -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    -e STREAMLIT_SERVER_PORT=8501 \
    -e STREAMLIT_SERVER_HEADLESS=true \
    -v "$(pwd)/../Attention_is_All_You_Need/results:/app/models:ro" \
    -v "$(pwd)/../Attention_is_All_You_Need/data:/app/data:ro" \
    twi-chinese-translator:latest

sleep 3
docker logs --tail 20 twi-translator
```

## Using Nginx Reverse Proxy

If behind Nginx:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

## Test Commands

```bash
# 1. Test inside container
docker exec -it twi-translator curl http://localhost:8501/_stcore/health

# 2. Test from host
curl http://localhost:8501/_stcore/health

# 3. Test from another machine
curl http://<SERVER_IP>:8501/_stcore/health

# 4. Check what's listening
sudo netstat -tlnp | grep 8501
# or
sudo ss -tlnp | grep 8501
```
