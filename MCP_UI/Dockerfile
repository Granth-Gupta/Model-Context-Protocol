FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci

# Copy source code and build
COPY . .
RUN npm run build

FROM nginx:alpine AS production

# Install gettext for envsubst (environment variable substitution)
RUN apk add --no-cache gettext

# Copy built app to nginx - React builds to 'build', not 'dist'
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx template for environment variable processing
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

# Set default environment variable if not provided
ENV REACT_APP_MCP_CLIENT_URL=https://mcp-client-273927490120.us-central1.run.app

EXPOSE 8080

# Use nginx with automatic template processing
CMD ["nginx", "-g", "daemon off;"]
