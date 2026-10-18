# AWS deployment (EC2 + S3/CloudFront)

## 0. Before anything
- Create an AWS Budget with alerts at $50, $100 and $150.
- Stop the EC2 instance whenever you are not demoing.

## 1. Backend on EC2
1. Launch Ubuntu 22.04 or 24.04 on a `t3.large` (or `c6i.large`), 30 GB disk.
2. Security group: allow 22 (your IP only), 80 and 443.
3. Install Docker:
   ```bash
   sudo apt update && sudo apt install -y docker.io docker-compose-v2 nginx certbot python3-certbot-nginx
   sudo usermod -aG docker $USER
   ```
4. Copy the repo and weights to the instance (`git clone` plus `scp` or `aws s3 cp` for `models/`).
5. `cp .env.example .env`, set `CORS_ORIGINS` to your frontend origin, then:
   ```bash
   docker compose up -d --build backend
   curl localhost:8000/health
   curl localhost:8000/models
   ```

## 2. HTTPS (required for webcam access)
Point a domain or subdomain at the instance's Elastic IP, then add an Nginx server block:

```nginx
server {
    server_name api.example.com;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 120s;
    }
}
```

```bash
sudo certbot --nginx -d api.example.com
```

## 3. Frontend on S3 + CloudFront
1. Set `window.API_BASE = "https://api.example.com"` in `frontend/config.js`.
2. Create an S3 bucket, upload `frontend/*`, and serve it through CloudFront (origin access control, default root object `index.html`).
3. Set `CORS_ORIGINS` on the backend to the CloudFront domain, then restart the container.

## 4. Optional extras
- Model weights in S3: `aws s3 cp s3://<bucket>/models/ ./models/ --recursive` at deploy time.
- CloudWatch agent for container logs.
- GitHub Actions to build the backend image and push it to ECR.

## 5. Benchmark for the report
Record end-to-end FPS and `latency_ms` for each model on each instance type you try, and note the
approximate cost per hour.
