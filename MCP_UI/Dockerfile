FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Expose the port your dev server uses (check package.json)
EXPOSE 5173

# Run development server
CMD ["npm", "run", "dev"]
