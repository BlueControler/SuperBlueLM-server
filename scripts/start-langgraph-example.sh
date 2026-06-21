#!/bin/sh
. .venv/bin/activate
langgraph dev --ssl-keyfile .local/ssl/your-address/privkey.pem \
    --ssl-certfile .local/ssl/your-address/fullchain.pem \
    --no-browser \
    --host 0.0.0.0 \
    --port 2024