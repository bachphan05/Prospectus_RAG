# 1. Use Node.js 22 (required by Vite 6.x)
FROM node:22-alpine

# 2. Set the working directory
WORKDIR /app

# 3. Copy package.json and package-lock.json first
COPY package.json package-lock.json ./

# 4. Install dependencies
RUN npm install

# 5. Copy the rest of the frontend code
COPY . .

# 6. Expose the port Vite runs on (default is 5173)
EXPOSE 5173

# 7. Run the development server
# --host 0.0.0.0 is crucial! It lets the container accept connections from outside.
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]