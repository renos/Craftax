#!/usr/bin/env python3
"""
Asset generation script for Fabrax textures.
Uses OpenAI GPT-4 with image generation to create missing game assets based on reference images.
"""

import yaml
import os
import base64
from openai import OpenAI
from PIL import Image
import io
import time

# Load OpenAI API key
def load_openai_key():
    """Load OpenAI API key from ~/.openai/openai.key"""
    key_path = os.path.expanduser("~/.openai/openai.key")
    try:
        with open(key_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"OpenAI API key not found at {key_path}. Please create this file with your API key.")

# Initialize OpenAI client
api_key = load_openai_key()
client = OpenAI(api_key=api_key)

def load_config(config_path):
    """Load the assets configuration YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def encode_image_to_base64(image_path):
    """Encode reference image to base64 for OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_file(file_path):
    """Upload file to OpenAI and return file ID."""
    with open(file_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="assistants")
        return file_obj.id

def generate_asset(asset_name, asset_config, reference_assets_dir, output_dir):
    """Generate a single asset using GPT-4 image generation with reference image."""

    reference_path = os.path.join(reference_assets_dir, asset_config['reference_asset'])
    output_path = os.path.join(output_dir, f"{asset_name}.png")

    # Skip if already exists
    if os.path.exists(output_path):
        print(f"✓ {asset_name}.png already exists, skipping...")
        return True

    if not os.path.exists(reference_path):
        print(f"❌ Reference image not found: {reference_path}")
        return False

    print(f"🎨 Generating {asset_name}.png using reference {asset_config['reference_asset']}...")

    try:
        # Encode reference image
        reference_b64 = encode_image_to_base64(reference_path)

        # Create prompt with reference context
        full_prompt = f"""Generate a minecraft-style 16x16 pixel art texture based on the reference image provided, but modified as follows: {asset_config['prompt']}

IMPORTANT: Keep the exact same pixel art style, color palette approach, and aesthetic as the reference image. Only change the specific elements mentioned while maintaining the 16x16 pixel art format with sharp, blocky pixels and no anti-aliasing."""

        # Generate image using GPT-4 with image generation
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": full_prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{reference_b64}",
                        },
                    ],
                }
            ],
            tools=[{"type": "image_generation"}],
        )

        # Extract generated image
        image_generation_calls = [
            output
            for output in response.output
            if output.type == "image_generation_call"
        ]

        if not image_generation_calls:
            print(f"❌ No image generated for {asset_name}")
            return False

        image_data = [output.result for output in image_generation_calls]
        if not image_data:
            print(f"❌ No image data returned for {asset_name}")
            return False

        # Decode and save the image
        image_base64 = image_data[0]
        image_bytes = base64.b64decode(image_base64)

        # Open with PIL and resize to exactly 16x16
        image = Image.open(io.BytesIO(image_bytes))
        # Resize to 16x16 using nearest neighbor to preserve pixel art style
        image_16x16 = image.resize((16, 16), Image.NEAREST)

        # Save as PNG
        image_16x16.save(output_path, "PNG")
        print(f"✅ Successfully generated and saved {asset_name}.png")

        # Add delay to respect API rate limits
        time.sleep(1)
        return True

    except Exception as e:
        print(f"❌ Failed to generate {asset_name}: {str(e)}")
        return False

def main():
    """Main function to generate all missing assets."""

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "assets_config.yaml")
    reference_assets_dir = os.path.join(script_dir, "../../craftax_classic/assets")
    output_dir = os.path.join(script_dir, "assets_generated")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load configuration
    print("📋 Loading asset configuration...")
    config = load_config(config_path)

    assets = config['assets']
    total_assets = len(assets)
    generated_count = 0
    failed_count = 0
    skipped_count = 0

    print(f"🚀 Starting generation of {total_assets} assets...")
    print("=" * 60)

    # Generate each asset
    for asset_name, asset_config in assets.items():
        print(f"\n[{generated_count + failed_count + skipped_count + 1}/{total_assets}] Processing {asset_name}...")

        output_path = os.path.join(output_dir, f"{asset_name}.png")
        if os.path.exists(output_path):
            print(f"✓ {asset_name}.png already exists, skipping...")
            skipped_count += 1
            continue

        success = generate_asset(asset_name, asset_config, reference_assets_dir, output_dir)

        if success:
            generated_count += 1
        else:
            failed_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("📊 GENERATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully generated: {generated_count}")
    print(f"⏭️  Already existed (skipped): {skipped_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📈 Total processed: {generated_count + failed_count + skipped_count}")

    if failed_count == 0:
        print("\n🎉 All assets generated successfully!")
    else:
        print(f"\n⚠️  {failed_count} assets failed to generate. Check errors above.")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())