
from pathlib import Path
import uuid

folder = Path(r"C:\Users\liana\Downloads\cropped_raw")

extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

images = sorted(
    [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ],
    key=lambda f: f.name.lower()
)

start_number = 2642

# مرحله اول: تغییر نام تمام تصاویر به نام موقت
temp_files = []

for image in images:
    temp_name = folder / f"temp_{uuid.uuid4().hex}.tmp"
    image.rename(temp_name)
    temp_files.append(temp_name)

# مرحله دوم: نام‌گذاری نهایی
for index, temp_file in enumerate(temp_files):
    new_name = folder / f"Liver{start_number + index}.png"
    temp_file.rename(new_name)

print("=" * 45)
print(f"Successfully renamed {len(images)} images!")
print(f"First image: Liver{start_number}.png")
print(f"Last image: Liver{start_number + len(images) - 1}.png")
print("=" * 45)
