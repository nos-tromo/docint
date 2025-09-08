# ---------- Builder ----------
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first (better caching)
COPY package*.json ./
RUN npm ci --include=dev

# Copy source files
COPY . .

# Build the app
RUN npm run build

# ---------- Runtime ----------
FROM nginx:stable-alpine AS runtime

# Clean out default nginx html
RUN rm -rf /usr/share/nginx/html/*

# Copy built artifacts only
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
