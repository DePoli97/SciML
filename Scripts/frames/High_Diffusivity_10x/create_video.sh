#!/bin/bash
# Script to create video from frame sequence
# Requires ffmpeg to be installed

# Navigate to the script's directory to ensure we find the files
cd "$(dirname "$0")"
echo "Working directory: $(pwd)"
echo "Creating videos from $(ls frame_*.png | wc -l) frames..."
ls -l frame_0000.png 2>/dev/null || echo "Warning: Cannot find first frame!"

# Create MP4 video (high quality, 10 fps)
ffmpeg -y -r 10 -i frame_%04d.png -vf "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2:color=white" -c:v libx264 -pix_fmt yuv420p -crf 18 High_Diffusivity_10x_wave_propagation.mp4

# Create WebM video (web-friendly, 10 fps)
# ffmpeg -y -r 10 -i frame_%04d.png -vf "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2:color=white" -c:v libvpx-vp9 -crf 30 -b:v 0 High_Diffusivity_10x_wave_propagation.webm

# Create animated GIF (lower quality, smaller file)
# ffmpeg -y -r 5 -i frame_%04d.png -vf "scale=640:512,pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2:color=white" High_Diffusivity_10x_wave_propagation.gif

echo "Video compilation completed!"
echo "Generated files:"
echo "  High_Diffusivity_10x_wave_propagation.mp4 (high quality)"
# echo "  High_Diffusivity_10x_wave_propagation.webm (web format)"
# echo "  High_Diffusivity_10x_wave_propagation.gif (animated GIF)"
