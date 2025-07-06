#!/bin/bash
# Create necessary directories and files

# Create data directory if it doesn't exist
mkdir -p data
mkdir -p models

# Create style templates file if it doesn't exist
if [ ! -f "data/style_templates.json" ]; then
    echo "Creating style templates file..."
    cat > data/style_templates.json << 'EOF'
{
  "cinematic": "Create a cinematic scene with dramatic lighting and composition",
  "anime": "Design an anime-style illustration with vibrant colors",
  "photorealistic": "Generate a photorealistic image with high detail",
  "fantasy": "Create a fantasy-themed illustration with magical elements",
  "abstract": "Design an abstract artistic composition",
  "cyberpunk": "Create a cyberpunk-themed image with neon lights, high technology, and urban dystopia",
  "sci-fi": "Generate a science fiction scene with futuristic technology and cosmic elements",
  "cartoon": "Design a cartoon-style illustration with exaggerated features and bright colors",
  "oil-painting": "Create an image in the style of classical oil painting with rich textures and depth",
  "watercolor": "Design a watercolor-style illustration with soft edges and translucent colors"
}
EOF
    echo "Style templates file created successfully."
else
    echo "Style templates file already exists."
fi

echo "Directories and files created successfully."