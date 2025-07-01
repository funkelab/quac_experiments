# Download and extract all the zip files in "urls.txt" to "/nrs/funke/adjavond/data/bbbc021"
# Usage: bash download.sh

# Check if the directory exists, if not create it
if [ ! -d "/nrs/funke/adjavond/data/bbbc021" ]; then
    mkdir -p /nrs/funke/adjavond/data/bbbc021
fi
# Copy urls.txt to the directory
cp urls.txt /nrs/funke/adjavond/data/bbbc021
# Change to the directory
cd /nrs/funke/adjavond/data/bbbc021

# Download the files listed in urls.txt
while IFS= read -r url; do
    # Extract the filename from the URL
    filename=$(basename "$url")
    
    # Check if the file already exists
    if [ ! -f "$filename" ]; then
        echo "Downloading $filename..."
        wget "$url"
    else
        echo "$filename already exists, skipping download."
    fi
    
    # Unzip the file if it is not already unzipped
    if [ ! -d "${filename%.zip}" ]; then
        echo "Unzipping $filename..."
        unzip "$filename"
    else
        echo "${filename%.zip} already exists, skipping unzip."
    fi
done < urls.txt
# Remove the zip files after unzipping
for file in *.zip; do
    if [ -f "$file" ]; then
        echo "Removing $file..."
        rm "$file"
    fi
done