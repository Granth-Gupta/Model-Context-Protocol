FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Expose port 3000 (default for development server)
EXPOSE 3000

# Set environment to development for npm run dev
ENV NODE_ENV=development

# Run development server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
