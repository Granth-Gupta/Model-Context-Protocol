FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# 2. Serve with Nginx
FROM nginx:stable-alpine
COPY --from=builder /app/dist /usr/share/nginx/html

# Replace Nginx default config with one that listens on $PORT (default 8080)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Set default environment variable if not provided
ENV REACT_APP_MCP_CLIENT_URL=https://mcp-client-service-514638629767.us-central1.run.app

EXPOSE 8080

# Use nginx with automatic template processing
CMD ["nginx", "-g", "daemon off;"]
