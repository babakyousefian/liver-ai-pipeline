"""
Image Extractor Script
Extracts all images from the NewLiver folder structure and copies them to NewLiverData

Author: AI Assistant
Date: 2026
Purpose: Extract images from nested folder structure into a single directory
"""

import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path

ABSOLUTE_PATH = r"C:\Users\liana\Documents\pythonProjects\FinalTasks\packages\data\my_Data.backup"


class ImageExtractor:
    """
    A class to extract all images from a nested folder structure
    and copy them to a single destination folder.
    """

    def __init__(self, base_path=ABSOLUTE_PATH):
        """
        Initialize the ImageExtractor with paths

        Args:
            base_path (str): Path to the base directory containing the data
        """
        self.base_path = Path(base_path)
        self.source_path = self.base_path / "NewSonoDr.Roshanaee"
        self.destination_path = self.base_path / "NewLiverData2026"

        # Supported image formats
        self.image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
        }

        # Store image paths and results
        self.all_images = []
        self.copied_images = []
        self.errors = []

    def find_all_images(self):
        """
        Recursively find all images in the source directory

        Returns:
            list: List of Path objects for all found images
        """
        print("🔍 Searching for images...")

        if not self.source_path.exists():
            raise FileNotFoundError(f"Path {self.source_path} does not exist!")

        # Walk through all directories and collect image files
        for root, dirs, files in os.walk(self.source_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.image_extensions:
                    self.all_images.append(file_path)

        print(f"📸 Found {len(self.all_images)} images")
        return self.all_images

    def get_file_hash(self, file_path):
        """
        Calculate MD5 hash of a file for duplicate detection

        Args:
            file_path (Path): Path to the file

        Returns:
            str: MD5 hash of the file, or None if error
        """
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None

    def extract_images(self, check_duplicates=True):
        """
        Extract and copy all images to the destination folder

        Args:
            check_duplicates (bool): Whether to check for duplicate files

        Returns:
            list: List of dictionaries with information about copied files
        """

        # Create destination folder
        self.destination_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Destination folder: {self.destination_path}")

        # Find images if not already found
        if not self.all_images:
            self.find_all_images()

        print(f"🚀 Starting to copy {len(self.all_images)} images...")

        # Track duplicates using hash values
        seen_hashes = set()

        for idx, img_path in enumerate(self.all_images, 1):
            try:
                # Create unique filename
                original_name = img_path.name
                base_name, ext = os.path.splitext(original_name)

                # Handle duplicate filenames by adding counter
                new_name = original_name
                counter = 1
                while (self.destination_path / new_name).exists():
                    new_name = f"{base_name}_{counter}{ext}"
                    counter += 1

                # Optional: Check if file content is duplicate (by hash)
                if check_duplicates:
                    file_hash = self.get_file_hash(img_path)
                    if file_hash in seen_hashes:
                        # Skip duplicate content
                        print(f"⏭️ Skipping duplicate: {img_path.name}")
                        continue
                    seen_hashes.add(file_hash)

                # Copy the file preserving metadata
                dest_path = self.destination_path / new_name
                shutil.copy2(img_path, dest_path)

                # Store file information
                self.copied_images.append(
                    {
                        "source": img_path,
                        "destination": dest_path,
                        "name": new_name,
                        "size": img_path.stat().st_size,
                        "modified": datetime.fromtimestamp(img_path.stat().st_mtime),
                    }
                )

                # Show progress every 20 files
                if idx % 20 == 0:
                    print(f"📊 Progress: {idx}/{len(self.all_images)}")

            except Exception as e:
                error_msg = f"Error in {img_path.name}: {str(e)}"
                self.errors.append(error_msg)
                print(f"⚠️ {error_msg}")

        return self.copied_images

    def generate_report(self):
        """
        Generate and display a comprehensive report of the extraction

        Returns:
            list: List of copied images information
        """
        print("\n" + "=" * 60)
        print("📊 Final Report - Image Extraction")
        print("=" * 60)

        total = len(self.copied_images)
        print(f"✅ Number of images copied: {total}")

        # Show errors if any
        if self.errors:
            print(f"⚠️ Number of errors: {len(self.errors)}")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"   - {error}")

        # Calculate and format total size
        total_size = sum(img["size"] for img in self.copied_images)
        if total_size > 1024**3:
            size_str = f"{total_size/1024**3:.2f} GB"
        elif total_size > 1024**2:
            size_str = f"{total_size/1024**2:.2f} MB"
        else:
            size_str = f"{total_size/1024:.2f} KB"

        print(f"📦 Total size: {size_str}")

        # Display sample of copied files
        print("\n📁 Sample of copied files:")
        for img in self.copied_images[:5]:
            print(f"   - {img['name']} ({img['size']/1024:.1f} KB)")

        print(f"\n📂 All images in: {self.destination_path}")
        return self.copied_images


# Main execution block
if __name__ == "__main__":
    try:
        # Initialize the extractor
        extractor = ImageExtractor()

        # Find all images
        extractor.find_all_images()

        # Extract and copy all images
        results = extractor.extract_images()

        # Generate and display final report
        extractor.generate_report()

        print("\n🎉 Operation completed successfully!")
        print(f"📁 Images saved to: {extractor.destination_path}")

    except Exception as e:
        print(f"❌ General error: {e}")
        print("Please make sure the path is correct and you have sufficient permissions.")
