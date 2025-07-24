FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci

# Copy source code and build
COPY . .
RUN npm run build

FROM nginx:alpine AS production

# Install envsubst for environment variable substitution
RUN apk add --no-cache gettext

# Copy built app to nginx (React builds to 'build', not 'dist')
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx template
COPY nginx.conf.template /etc/nginx/templates/nginx.conf.template

EXPOSE 8080

# Use nginx with template processing
CMD ["nginx", "-g", "daemon off;"]
